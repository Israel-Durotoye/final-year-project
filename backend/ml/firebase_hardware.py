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


def _firebase_base_url() -> str:
    return (
        os.getenv("HARDWARE_FIREBASE_URL")
        or os.getenv("VITE_HARDWARE_FIREBASE_URL")
        or DEFAULT_FIREBASE_URL
    ).rstrip("/")


def _read_firebase_payload(url: str, timeout_seconds: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:  # nosec B310 - configured Firebase URL
        payload = json.loads(response.read().decode("utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Hardware Firebase returned an unexpected telemetry payload.")
    return payload


def fetch_all_hardware_rows(
    *,
    page_size: int = 1000,
    timeout_seconds: float = 30,
) -> list[dict[str, Any]]:
    """Fetch the complete physical-node log using Firebase key pagination."""
    if page_size < 1:
        raise ValueError("page_size must be at least 1.")

    firebase_url = _firebase_base_url()
    rows: list[dict[str, Any]] = []
    last_key: str | None = None

    while True:
        request_limit = page_size + (1 if last_key else 0)
        query_values: dict[str, Any] = {
            "orderBy": '"$key"',
            "limitToFirst": request_limit,
        }
        if last_key:
            query_values["startAt"] = json.dumps(last_key)
        url = f"{firebase_url}/readings/log.json?{urlencode(query_values)}"
        payload = _read_firebase_payload(url, timeout_seconds)
        ordered_items = sorted(payload.items())
        if last_key:
            ordered_items = [item for item in ordered_items if item[0] > last_key]

        for push_id, value in ordered_items:
            if not isinstance(value, dict):
                continue
            normalized = normalize_firebase_row(str(push_id), value)
            if normalized is not None:
                rows.append(normalized)

        if len(payload) < request_limit or not ordered_items:
            break
        last_key = ordered_items[-1][0]

    rows.sort(key=lambda row: (str(row["Timestamp"]), str(row["Node_ID"])))
    return rows


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

    firebase_url = _firebase_base_url()
    requested_rows = min(max(max(limit, 1) * 2, 100), MAX_FIREBASE_ROWS)
    query = urlencode({"orderBy": '"$key"', "limitToLast": requested_rows})
    url = f"{firebase_url}/readings/log.json?{query}"

    payload = _read_firebase_payload(url, timeout_seconds)

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
