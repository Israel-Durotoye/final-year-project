"""Read and normalize the Firebase telemetry emitted by the INO gateway."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


PHYSICAL_NODE_IDS = {"NODE_01", "NODE_02"}
FIREBASE_PUSH_ALPHABET = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
DEFAULT_FIREBASE_URL = "https://capstone-2e26e-default-rtdb.firebaseio.com"
MAX_FIREBASE_ROWS = 5000


def is_physical_node(node_id: str) -> bool:
    return str(node_id).strip().upper() in PHYSICAL_NODE_IDS


def firebase_push_timestamp(push_id: str) -> str | None:
    if len(push_id) < 8:
        return None

    timestamp_ms = 0
    for character in push_id[:8]:
        value = FIREBASE_PUSH_ALPHABET.find(character)
        if value < 0:
            return None
        timestamp_ms = timestamp_ms * 64 + value

    try:
        return datetime.fromtimestamp(
            timestamp_ms / 1000,
            tz=timezone.utc,
        ).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_firebase_row(push_id: str, row: dict[str, Any]) -> dict[str, Any] | None:
    node_id = str(row.get("node_id") or "").strip().upper()
    timestamp = firebase_push_timestamp(push_id)
    if node_id not in PHYSICAL_NODE_IDS or timestamp is None:
        return None

    return {
        "Node_ID": node_id,
        "Timestamp": timestamp,
        "Nitrogen_mg_k": _number(row.get("nitrogen")),
        "Phosphorus_m": _number(row.get("phosphorus")),
        "Potassium_mg_": _number(row.get("potassium")),
        "Moisture_%": _number(row.get("moisture")),
        "Temperature_C": _number(row.get("temp")),
        "Humidity_%": _number(row.get("humidity")),
        "Soil_pH": _number(row.get("ph")),
        "Latitude": _number(row.get("latitude")),
        "Longitude": _number(row.get("longitude")),
        "Altitude_m": _number(row.get("altitude")),
        "Satellites": row.get("satellites"),
        "Season": row.get("season"),
        "GPS_Source": row.get("gps_source"),
        "Device_Uptime_Seconds": row.get("timestamp"),
        "Data_Source": "hardware",
    }


def fetch_hardware_rows(
    node_id: str | None = None,
    *,
    limit: int = 100,
    timeout_seconds: float = 12,
) -> list[dict[str, Any]]:
    """Return normalized hardware rows in chronological order."""
    cleaned_node = str(node_id or "").strip().upper()
    if cleaned_node and cleaned_node not in PHYSICAL_NODE_IDS:
        return []

    firebase_url = (
        os.getenv("HARDWARE_FIREBASE_URL")
        or os.getenv("VITE_HARDWARE_FIREBASE_URL")
        or DEFAULT_FIREBASE_URL
    ).rstrip("/")
    requested_rows = min(max(max(limit, 1) * 2, 100), MAX_FIREBASE_ROWS)
    query = urlencode({"orderBy": '"$key"', "limitToLast": requested_rows})
    url = f"{firebase_url}/readings/log.json?{query}"

    with urlopen(url, timeout=timeout_seconds) as response:  # nosec B310 - configured Firebase URL
        payload = json.loads(response.read().decode("utf-8"))

    if payload is None:
        return []
    if not isinstance(payload, dict):
        raise ValueError("Hardware Firebase returned an unexpected telemetry payload.")

    rows: list[dict[str, Any]] = []
    for push_id, value in payload.items():
        if not isinstance(value, dict):
            continue
        normalized = normalize_firebase_row(str(push_id), value)
        if normalized is None:
            continue
        if cleaned_node and normalized["Node_ID"] != cleaned_node:
            continue
        rows.append(normalized)

    rows.sort(key=lambda row: str(row["Timestamp"]))
    return rows[-max(limit, 1):]
