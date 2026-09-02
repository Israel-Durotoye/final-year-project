import os
import time
import random
from math import cos, radians, sin
from datetime import datetime
from typing import Any, Callable, Dict, List

import httpx
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions
from backend.utils.season import get_nigerian_season

# Load environment variables
load_dotenv()

TABLE_NAME = os.environ.get("FARM_DATA_TABLE", "capstone_dataset")
TRANSMISSION_INTERVAL_SECONDS = float(os.environ.get("SIMULATOR_INTERVAL_SECONDS", "60"))
MAX_INSERT_ATTEMPTS = max(1, int(os.environ.get("SIMULATOR_MAX_INSERT_ATTEMPTS", "4")))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("SIMULATOR_RETRY_BASE_DELAY_SECONDS", "1"))
HEXAGON_CENTER_LATITUDE = float(os.environ.get("SIMULATOR_CENTER_LATITUDE", "8.48225"))
HEXAGON_CENTER_LONGITUDE = float(os.environ.get("SIMULATOR_CENTER_LONGITUDE", "4.54225"))
HEXAGON_RADIUS_METERS = float(os.environ.get("SIMULATOR_HEX_RADIUS_METERS", "140"))

# FUT Minna's Gidan Kwano main campus lies within approximately
# 9.5281-9.5369 N and 6.4386-6.4664 E. NODE_04-NODE_06 are placed in a
# compact triangle near the centre of those published campus bounds.
FUT_MINNA_CENTER_LATITUDE = float(
    os.environ.get("FUT_MINNA_CENTER_LATITUDE", "9.53250")
)
FUT_MINNA_CENTER_LONGITUDE = float(
    os.environ.get("FUT_MINNA_CENTER_LONGITUDE", "6.45250")
)
FUT_MINNA_NODE_RADIUS_METERS = float(
    os.environ.get("FUT_MINNA_NODE_RADIUS_METERS", "140")
)
FUT_MINNA_LATITUDE_BOUNDS = (9.5280556, 9.5369444)
FUT_MINNA_LONGITUDE_BOUNDS = (6.4386111, 6.4663889)

# --- Crop Profiles for Realistic Simulated Telemetry ---
CROP_PROFILES = {
    "Maize": {
        "temp": (20.0, 30.0), "moisture": (50.0, 70.0), "humidity": (55.0, 75.0),
        "n": (100, 150), "p": (30, 50), "k": (80, 120)
    },
    "Cassava": {
        "temp": (25.0, 35.0), "moisture": (40.0, 60.0), "humidity": (50.0, 70.0),
        "n": (50, 90), "p": (10, 30), "k": (100, 150)
    },
    "Rice": {
        "temp": (22.0, 32.0), "moisture": (80.0, 100.0), "humidity": (70.0, 90.0),
        "n": (80, 120), "p": (20, 40), "k": (30, 60)
    }
}

NODE_CROPS = ("Maize", "Maize", "Cassava", "Cassava", "Rice", "Rice")
SIMULATED_NODE_IDS = ("NODE_03", "NODE_04", "NODE_05", "NODE_06")


def build_hexagon_nodes(
    center_latitude: float = HEXAGON_CENTER_LATITUDE,
    center_longitude: float = HEXAGON_CENTER_LONGITUDE,
    radius_meters: float = HEXAGON_RADIUS_METERS,
) -> Dict[str, Dict[str, Any]]:
    """Place six nodes clockwise around a geographic center."""
    latitude_degrees_per_meter = 1.0 / 111_320.0
    longitude_degrees_per_meter = 1.0 / (
        111_320.0 * cos(radians(center_latitude))
    )
    nodes: Dict[str, Dict[str, Any]] = {}

    for index, crop in enumerate(NODE_CROPS):
        angle = radians(30 + index * 60)
        nodes[f"NODE_{index + 1:02d}"] = {
            "lat": round(
                center_latitude + radius_meters * sin(angle) * latitude_degrees_per_meter,
                7,
            ),
            "lng": round(
                center_longitude + radius_meters * cos(angle) * longitude_degrees_per_meter,
                7,
            ),
            "crop": crop,
        }

    return nodes


def build_fut_minna_node_coordinates(
    center_latitude: float = FUT_MINNA_CENTER_LATITUDE,
    center_longitude: float = FUT_MINNA_CENTER_LONGITUDE,
    radius_meters: float = FUT_MINNA_NODE_RADIUS_METERS,
) -> Dict[str, Dict[str, float]]:
    """Place NODE_04-NODE_06 inside FUT Minna's Gidan Kwano campus."""
    latitude_degrees_per_meter = 1.0 / 111_320.0
    longitude_degrees_per_meter = 1.0 / (
        111_320.0 * cos(radians(center_latitude))
    )
    coordinates: Dict[str, Dict[str, float]] = {}

    for node_id, angle_degrees in zip(
        ("NODE_04", "NODE_05", "NODE_06"),
        (30, 150, 270),
    ):
        angle = radians(angle_degrees)
        coordinates[node_id] = {
            "lat": round(
                center_latitude
                + radius_meters * sin(angle) * latitude_degrees_per_meter,
                7,
            ),
            "lng": round(
                center_longitude
                + radius_meters * cos(angle) * longitude_degrees_per_meter,
                7,
            ),
        }

    return coordinates


NODES = build_hexagon_nodes()
for fut_node_id, fut_coordinates in build_fut_minna_node_coordinates().items():
    NODES[fut_node_id].update(fut_coordinates)


def create_simulator_client() -> tuple[Client, httpx.Client]:
    url = os.environ.get("VITE_SUPABASE_URL")
    key = os.environ.get("VITE_SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials missing from .env file.")

    # The simulator is long-lived. Use HTTP/1.1 explicitly so a damaged HTTP/2
    # stream cannot terminate the process, and keep transport retries bounded.
    limits = httpx.Limits(
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=30.0,
    )
    http_client = httpx.Client(
        http1=True,
        http2=False,
        timeout=httpx.Timeout(30.0, connect=15.0),
        limits=limits,
        transport=httpx.HTTPTransport(
            http1=True,
            http2=False,
            limits=limits,
            retries=1,
        ),
    )
    options = ClientOptions(httpx_client=http_client, postgrest_client_timeout=30)
    return create_client(url, key, options), http_client


def build_telemetry_batch(timestamp: datetime | None = None) -> List[Dict[str, Any]]:
    reading_timestamp = timestamp or datetime.now()
    reading_time = reading_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    batch: List[Dict[str, Any]] = []

    for node_id in SIMULATED_NODE_IDS:
        config = NODES[node_id]
        profile = CROP_PROFILES[config["crop"]]
        batch.append({
            "Timestamp": reading_time,
            "Node_ID": node_id,
            "Moisture_%": round(random.uniform(*profile["moisture"]), 1),
            "Temperature_C": round(random.uniform(*profile["temp"]), 1),
            "Humidity_%": round(random.uniform(*profile["humidity"]), 1),
            "Nitrogen_mg_k": round(random.uniform(*profile["n"]), 2),
            "Phosphorus_m": round(random.uniform(*profile["p"]), 2),
            "Potassium_mg_": round(random.uniform(*profile["k"]), 2),
            "Latitude": config["lat"],
            "Longitude": config["lng"],
            "Altitude_m": round(random.uniform(295.0, 305.0), 1),
            "Satellites": random.choice([4, 5, 6, 7, 8, "ERR"]),
            "Season": get_nigerian_season(reading_timestamp),
            "Target_Crop": config["crop"],
        })

    return batch


def telemetry_batch_exists(client: Client, batch: List[Dict[str, Any]]) -> bool:
    """Check whether a batch was committed when its HTTP response was lost."""
    if not batch:
        return True

    timestamp = batch[0]["Timestamp"]
    expected_node_ids = {str(item["Node_ID"]) for item in batch}
    response = (
        client.table(TABLE_NAME)
        .select("Node_ID")
        .eq("Timestamp", timestamp)
        .in_("Node_ID", sorted(expected_node_ids))
        .execute()
    )
    persisted_node_ids = {
        str(item.get("Node_ID"))
        for item in (response.data or [])
        if isinstance(item, dict)
    }
    return expected_node_ids.issubset(persisted_node_ids)


def insert_telemetry_batch(
    client: Client,
    batch: List[Dict[str, Any]],
    max_attempts: int = MAX_INSERT_ATTEMPTS,
    base_delay_seconds: float = RETRY_BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Insert one telemetry cycle, retrying transport failures only."""
    for attempt in range(1, max_attempts + 1):
        try:
            client.table(TABLE_NAME).insert(batch).execute()
            return
        except httpx.TransportError as exc:
            try:
                if telemetry_batch_exists(client, batch):
                    print("✅ Telemetry was committed before the response connection failed.")
                    return
            except Exception:
                # Verification is best-effort; the bounded retry below remains
                # responsible for recovering from a fully failed request.
                pass

            if attempt >= max_attempts:
                raise

            delay = base_delay_seconds * (2 ** (attempt - 1))
            print(
                f"⚠️  Supabase transport error ({type(exc).__name__}); "
                f"retrying in {delay:.1f}s ({attempt}/{max_attempts - 1})..."
            )
            sleep(delay)


def run_simulator() -> None:
    client, http_client = create_simulator_client()
    print("🌱 Starting capstone hardware simulation... Press Ctrl+C to stop.")
    print(
        f"⬡ Simulating {', '.join(SIMULATED_NODE_IDS)}. "
        f"NODE_04-NODE_06 use FUT Minna Gidan Kwano GPS positions around "
        f"({FUT_MINNA_CENTER_LATITUDE:.5f}, {FUT_MINNA_CENTER_LONGITUDE:.5f})."
    )

    try:
        while True:
            batch = build_telemetry_batch()
            insert_telemetry_batch(client, batch)

            for data in batch:
                print(f"📡 Sent {data['Target_Crop']} telemetry for {data['Node_ID']}")

            print(
                f"⏳ Waiting {TRANSMISSION_INTERVAL_SECONDS:g} seconds "
                "for next transmission..."
            )
            time.sleep(TRANSMISSION_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:
        http_client.close()


if __name__ == "__main__":
    run_simulator()
