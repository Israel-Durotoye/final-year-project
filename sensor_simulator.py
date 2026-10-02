import os
import time
import random
from dataclasses import dataclass
from math import cos, pi, radians, sin
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
# 9.5281-9.5369 N and 6.4386-6.4664 E. NODE_04-NODE_07 are placed in a
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

NODE_CROPS = ("Maize", "Maize", "Cassava", "Cassava", "Rice", "Rice", "Rice")
SIMULATED_NODE_IDS = ("NODE_04", "NODE_05", "NODE_06", "NODE_07")


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
    """Place all simulator nodes inside FUT Minna's Gidan Kwano campus."""
    latitude_degrees_per_meter = 1.0 / 111_320.0
    longitude_degrees_per_meter = 1.0 / (
        111_320.0 * cos(radians(center_latitude))
    )
    coordinates: Dict[str, Dict[str, float]] = {}

    for node_id, angle_degrees in zip(
        SIMULATED_NODE_IDS,
        (330, 30, 150, 270),
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


# --- Stateful telemetry model -------------------------------------------------
#
# Each reading continues from the previous one instead of being drawn at random.
# Real soil does not jump between unrelated values every minute, and a sequence
# model can only learn from data that has a pattern over time:
#
#   temperature  follows a daily cycle (coolest before dawn, warmest mid-afternoon)
#   humidity     moves opposite to temperature
#   moisture     dries slowly, faster when warm, and jumps up when the field is
#                irrigated or it rains
#   N, P, K      drift slowly around a field-specific level and dip slightly
#                when water is added
#
# Values stay close to the crop's reference range in CROP_PROFILES.

DRYING_HOURS = float(os.environ.get("SIMULATOR_DRYING_HOURS", "6"))
RAIN_CHANCE_PER_MINUTE = {"rainy": 0.004, "dry": 0.0005}
MAX_STEP_MINUTES = 60.0


@dataclass
class NodeState:
    """The simulated field condition at one node."""

    moisture: float
    temperature_offset: float
    humidity_offset: float
    nitrogen: float
    phosphorus: float
    potassium: float
    nitrogen_level: float
    phosphorus_level: float
    potassium_level: float
    irrigation_point: float
    timestamp: datetime


_node_states: Dict[str, NodeState] = {}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _inner(bounds: tuple[float, float], rng: random.Random, margin: float = 0.25) -> float:
    """Pick a level away from the edges of a reference range."""
    low, high = bounds
    span = high - low
    return rng.uniform(low + span * margin, high - span * margin)


def _new_state(profile: Dict[str, tuple], timestamp: datetime, rng: random.Random) -> NodeState:
    moisture_low, moisture_high = profile["moisture"]
    nitrogen, phosphorus, potassium = (_inner(profile[key], rng) for key in ("n", "p", "k"))
    return NodeState(
        moisture=rng.uniform(moisture_low + 0.3 * (moisture_high - moisture_low), moisture_high),
        temperature_offset=0.0,
        humidity_offset=0.0,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        nitrogen_level=nitrogen,
        phosphorus_level=phosphorus,
        potassium_level=potassium,
        irrigation_point=moisture_low + rng.uniform(0.0, 2.0),
        timestamp=timestamp,
    )


def _daily_cycle(timestamp: datetime) -> float:
    """-1 at about 03:00, +1 at about 15:00."""
    hour = timestamp.hour + timestamp.minute / 60.0
    return sin(2 * pi * (hour - 9.0) / 24.0)


def _revert(value: float, target: float, rate: float, noise: float, minutes: float, rng: random.Random) -> float:
    """Move a value part of the way back to its level, plus a small random step."""
    pull = 1.0 - (1.0 - rate) ** minutes
    return value + (target - value) * pull + rng.gauss(0.0, noise) * minutes ** 0.5


def advance_node_state(
    state: NodeState,
    profile: Dict[str, tuple],
    timestamp: datetime,
    rng: random.Random,
) -> Dict[str, float]:
    """Move one node forward to ``timestamp`` and return its sensor values."""
    minutes = (timestamp - state.timestamp).total_seconds() / 60.0
    minutes = _clamp(minutes, 0.0, MAX_STEP_MINUTES)
    state.timestamp = timestamp
    cycle = _daily_cycle(timestamp)

    temp_low, temp_high = profile["temp"]
    temp_mid, temp_span = (temp_low + temp_high) / 2.0, temp_high - temp_low
    state.temperature_offset = _revert(state.temperature_offset, 0.0, 0.08, 0.12, minutes, rng)
    temperature = _clamp(
        temp_mid + 0.35 * temp_span * cycle + state.temperature_offset,
        temp_low,
        temp_high,
    )

    humidity_low, humidity_high = profile["humidity"]
    humidity_mid, humidity_span = (humidity_low + humidity_high) / 2.0, humidity_high - humidity_low
    state.humidity_offset = _revert(state.humidity_offset, 0.0, 0.08, 0.35, minutes, rng)
    humidity = _clamp(
        humidity_mid - 0.30 * humidity_span * cycle + state.humidity_offset,
        humidity_low,
        humidity_high,
    )

    moisture_low, moisture_high = profile["moisture"]
    moisture_span = moisture_high - moisture_low
    drying_per_minute = moisture_span / (DRYING_HOURS * 60.0)
    warmth = 1.0 + 0.6 * (temperature - temp_mid) / temp_span
    state.moisture -= drying_per_minute * warmth * minutes
    state.moisture += rng.gauss(0.0, 0.08) * minutes ** 0.5

    season = get_nigerian_season(timestamp).lower()
    rain_chance = RAIN_CHANCE_PER_MINUTE["rainy" if "rainy" in season else "dry"]
    watered = False
    if state.moisture <= state.irrigation_point:
        # The field is irrigated when it reaches the bottom of its range.
        state.moisture += rng.uniform(0.6, 0.9) * moisture_span
        state.irrigation_point = moisture_low + rng.uniform(0.0, 2.0)
        watered = True
    elif minutes > 0 and rng.random() < 1.0 - (1.0 - rain_chance) ** minutes:
        state.moisture += rng.uniform(0.25, 0.6) * moisture_span
        watered = True
    state.moisture = _clamp(state.moisture, moisture_low - 3.0, moisture_high)

    for name, level_name, key, rate, noise in (
        ("nitrogen", "nitrogen_level", "n", 0.01, 0.35),
        ("phosphorus", "phosphorus_level", "p", 0.01, 0.12),
        ("potassium", "potassium_level", "k", 0.01, 0.30),
    ):
        low, high = profile[key]
        value = _revert(getattr(state, name), getattr(state, level_name), rate, noise, minutes, rng)
        if watered and name != "phosphorus":
            # Added water dilutes the mobile nutrients for a while.
            value -= rng.uniform(0.01, 0.03) * (high - low)
        setattr(state, name, _clamp(value, low, high))

    return {
        "Moisture_%": round(state.moisture, 1),
        "Temperature_C": round(temperature, 1),
        "Humidity_%": round(humidity, 1),
        "Nitrogen_mg_k": round(state.nitrogen, 2),
        "Phosphorus_m": round(state.phosphorus, 2),
        "Potassium_mg_": round(state.potassium, 2),
    }


def reset_node_states() -> None:
    """Forget every simulated field condition (used by tests)."""
    _node_states.clear()


def build_telemetry_batch(
    timestamp: datetime | None = None,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    reading_timestamp = timestamp or datetime.now()
    reading_time = reading_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    generator = rng or random
    batch: List[Dict[str, Any]] = []

    for node_id in SIMULATED_NODE_IDS:
        config = NODES[node_id]
        profile = CROP_PROFILES[config["crop"]]
        state = _node_states.get(node_id)
        if state is None or reading_timestamp < state.timestamp:
            state = _node_states[node_id] = _new_state(profile, reading_timestamp, generator)
        batch.append({
            "Timestamp": reading_time,
            "Node_ID": node_id,
            **advance_node_state(state, profile, reading_timestamp, generator),
            "Latitude": config["lat"],
            "Longitude": config["lng"],
            "Altitude_m": round(generator.uniform(295.0, 305.0), 1),
            "Satellites": generator.choice([4, 5, 6, 7, 8, "ERR"]),
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
        f"NODE_04-NODE_07 use FUT Minna Gidan Kwano GPS positions around "
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
