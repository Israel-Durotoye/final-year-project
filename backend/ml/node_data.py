"""
node_data.py — Shared Supabase access for per-node sensor windows.

Both the ML API route (classify-suitability) and the chatbot tool need the same
thing: the most recent N readings for one node, in chronological order, plus that
node's Target_Crop. This module is the single place that talks to Supabase for
that purpose so the query shape is not duplicated.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - optional dependency
    Client = None
    create_client = None

from backend.ml import firebase_hardware, soil_health

logger = logging.getLogger(__name__)

FARM_DATA_TABLE = os.getenv("FARM_DATA_TABLE", "capstone_dataset")

# Window length required by the suitability classifier.
DEFAULT_WINDOW = 24


def _resolve_credentials() -> tuple[str | None, str | None]:
    url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")
    return url, key


def fetch_node_window(node_id: str, limit: int = DEFAULT_WINDOW) -> dict[str, Any]:
    """
    Fetch the latest ``limit`` readings for a node in chronological order.

    Returns a status dict:
        {"status": "ok", "rows": [...oldest..newest...], "crop": <Target_Crop>,
         "latest": <newest row>, "count": <n>}
    or a failure dict with status in
        {"unavailable", "insufficient_data"} and a "reason"/"message".
    """

    cleaned_node = str(node_id).strip().upper()
    if firebase_hardware.is_physical_node(cleaned_node):
        try:
            rows = firebase_hardware.fetch_hardware_rows(cleaned_node, limit=limit)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Hardware node window query failed for %s: %s", cleaned_node, exc)
            return {"status": "unavailable", "reason": "Unable to retrieve sensor data."}

        if not rows:
            return {
                "status": "insufficient_data",
                "message": f"No sensor data found for {cleaned_node}.",
                "count": 0,
            }
        return {
            "status": "ok",
            "rows": rows,
            "latest": rows[-1],
            "crop": None,
            "count": len(rows),
        }

    if create_client is None:
        return {"status": "unavailable", "reason": "Supabase package is not installed."}

    url, key = _resolve_credentials()
    if not url or not key:
        return {"status": "unavailable", "reason": "Supabase credentials are not configured."}

    try:
        client: Client = create_client(url, key)
        result = (
            client.table(FARM_DATA_TABLE)
            .select("*")
            .eq("Node_ID", cleaned_node)
            .order("Timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        rows = getattr(result, "data", None) or []
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("Node window query failed for %s: %s", cleaned_node, exc)
        return {"status": "unavailable", "reason": "Unable to retrieve sensor data."}

    if not rows:
        return {
            "status": "insufficient_data",
            "message": f"No sensor data found for {cleaned_node}.",
            "count": 0,
        }

    # Supabase returned newest-first; reverse to chronological (oldest-first).
    ordered = list(reversed(rows))
    latest = ordered[-1]

    return {
        "status": "ok",
        "rows": ordered,
        "latest": latest,
        "crop": latest.get("Target_Crop"),
        "count": len(ordered),
    }


def build_feature_matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
    """Convert reading rows into a [n, 6] matrix in soil_health.FEATURES order."""

    matrix: list[list[float]] = []
    for r in rows:
        vec: list[float] = []
        for f in soil_health.FEATURES:
            val = r.get(f)
            vec.append(float(val) if val is not None else 0.0)
        matrix.append(vec)
    return matrix
