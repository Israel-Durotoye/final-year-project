"""Read hardware-node telemetry from the dedicated Supabase project."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - optional dependency
    Client = Any
    create_client = None

HARDWARE_NODE_IDS = {"NODE_01", "NODE_02", "NODE_03"}
DEFAULT_TABLE = "capstone_dataset"


def _credentials() -> tuple[str | None, str | None]:
    return (
        os.getenv("HARDWARE_SUPABASE_URL") or os.getenv("VITE_HARDWARE_SUPABASE_URL"),
        os.getenv("HARDWARE_SUPABASE_ANON_KEY") or os.getenv("VITE_HARDWARE_SUPABASE_ANON_KEY"),
    )


def is_configured() -> bool:
    url, key = _credentials()
    return bool(url and key and create_client)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    timestamp = row.get("Timestamp") or row.get("timestamp") or row.get("created_at")
    if isinstance(timestamp, (int, float)):
        timestamp = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()
    return {
        **row,
        "Node_ID": str(row.get("Node_ID") or row.get("node_id") or "").strip().upper(),
        "Timestamp": str(timestamp or ""),
        "Nitrogen_mg_k": row.get("Nitrogen_mg_k", row.get("nitrogen")),
        "Phosphorus_m": row.get("Phosphorus_m", row.get("phosphorus")),
        "Potassium_mg_": row.get("Potassium_mg_", row.get("potassium")),
        "Moisture_%": row.get("Moisture_%", row.get("moisture")),
        "Temperature_C": row.get("Temperature_C", row.get("temperature", row.get("temp"))),
        "Humidity_%": row.get("Humidity_%", row.get("humidity")),
        "Latitude": _number(row.get("Latitude", row.get("latitude"))),
        "Longitude": _number(row.get("Longitude", row.get("longitude"))),
        "Altitude_m": _number(row.get("Altitude_m", row.get("altitude"))),
        "Data_Source": "hardware",
    }


def fetch_hardware_rows(node_id: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
    """Return the latest hardware rows in chronological order."""
    if not is_configured():
        return []
    cleaned = str(node_id or "").strip().upper()
    if cleaned and cleaned not in HARDWARE_NODE_IDS:
        return []

    url, key = _credentials()
    table = os.getenv("HARDWARE_SUPABASE_TABLE") or os.getenv("VITE_HARDWARE_SUPABASE_TABLE") or DEFAULT_TABLE
    client: Client = create_client(url, key)
    query = client.table(table).select("*").in_("Node_ID", sorted(HARDWARE_NODE_IDS))
    if cleaned:
        query = query.eq("Node_ID", cleaned)
    rows = query.order("Timestamp", desc=True).limit(max(limit, 1)).execute().data or []
    normalized = [normalize_row(row) for row in rows if isinstance(row, dict)]
    normalized.sort(key=lambda row: str(row.get("Timestamp") or ""))
    return normalized


def fetch_all_hardware_rows(*, page_size: int = 1000) -> list[dict[str, Any]]:
    return fetch_hardware_rows(limit=page_size)
