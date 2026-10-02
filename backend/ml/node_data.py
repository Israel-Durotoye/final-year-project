"""
node_data.py — Single telemetry gateway for per-node sensor windows.

Every backend consumer (the ML routes, the temporal service and the chatbot)
needs the same thing: recent readings for one node, in chronological order.
This module is the one place that decides which data source owns a node, reuses
one database client, and briefly caches results so a single chat request does
not download the same telemetry several times.

Node ownership
--------------
    NODE_01–NODE_03  physical hardware. Read from the hardware Supabase project
                     when it is configured; otherwise NODE_01/NODE_02 fall back
                     to the gateway's Firebase log and NODE_03 is unavailable.
    everything else  simulator Supabase project (``capstone_dataset``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - optional dependency
    Client = None
    create_client = None

from backend.ml import firebase_hardware, soil_health, supabase_hardware

logger = logging.getLogger(__name__)

FARM_DATA_TABLE = os.getenv("FARM_DATA_TABLE", "capstone_dataset")

# Window length required by the crop and anomaly models.
DEFAULT_WINDOW = 24

# Telemetry arrives at most once a minute, so a short cache removes duplicate
# downloads inside one request without hiding new readings.
CACHE_TTL_SECONDS = float(os.getenv("TELEMETRY_CACHE_SECONDS", "20"))

_cache_lock = threading.Lock()
_window_cache: dict[str, tuple[float, int, dict[str, Any]]] = {}
_active_nodes_cache: tuple[float, list[str]] | None = None
_client_lock = threading.Lock()
_client: Any | None = None
_client_credentials: tuple[str, str] | None = None


def _resolve_credentials() -> tuple[str | None, str | None]:
    url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")
    return url, key


def get_supabase_client() -> Any | None:
    """Return one shared simulator-project client, or None when unconfigured."""
    global _client, _client_credentials
    if create_client is None:
        return None
    url, key = _resolve_credentials()
    if not url or not key:
        return None
    with _client_lock:
        if _client is None or _client_credentials != (url, key):
            _client = create_client(url, key)
            _client_credentials = (url, key)
        return _client


def clear_cache() -> None:
    """Drop cached telemetry (used by tests and after training)."""
    global _active_nodes_cache
    with _cache_lock:
        _window_cache.clear()
    _active_nodes_cache = None
    firebase_hardware.clear_cache()


def telemetry_source(node_id: str) -> str:
    """Name the data source that owns ``node_id``."""
    cleaned = str(node_id).strip().upper()
    if cleaned in supabase_hardware.HARDWARE_NODE_IDS:
        if supabase_hardware.is_configured():
            return "hardware_supabase"
        if firebase_hardware.is_physical_node(cleaned):
            return "hardware_firebase"
        return "hardware_unconfigured"
    return "simulator"


def _cache_get(node_id: str, limit: int) -> dict[str, Any] | None:
    if CACHE_TTL_SECONDS <= 0:
        return None
    with _cache_lock:
        entry = _window_cache.get(node_id)
    if entry is None:
        return None
    stored_at, stored_limit, window = entry
    if time.monotonic() - stored_at > CACHE_TTL_SECONDS:
        return None
    if window.get("status") != "ok":
        return window if stored_limit >= limit else None
    # A cached larger window can serve a smaller request; a cached window that
    # held every available row can serve any request.
    if stored_limit < limit and window.get("count", 0) >= stored_limit:
        return None
    rows = window["rows"][-limit:]
    return {**window, "rows": rows, "latest": rows[-1], "count": len(rows)}


def _cache_put(node_id: str, limit: int, window: dict[str, Any]) -> None:
    if CACHE_TTL_SECONDS <= 0:
        return
    with _cache_lock:
        _window_cache[node_id] = (time.monotonic(), limit, window)


def _ok(rows: list[dict[str, Any]], crop: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "rows": rows,
        "latest": rows[-1],
        "crop": crop,
        "count": len(rows),
    }


def _no_data(node_id: str) -> dict[str, Any]:
    return {
        "status": "insufficient_data",
        "message": f"No sensor data found for {node_id}.",
        "count": 0,
    }


def _fetch_uncached(cleaned_node: str, limit: int) -> dict[str, Any]:
    source = telemetry_source(cleaned_node)

    if source == "hardware_supabase":
        try:
            rows = supabase_hardware.fetch_hardware_rows(cleaned_node, limit=limit)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Hardware Supabase window query failed for %s: %s", cleaned_node, exc)
            return {"status": "unavailable", "reason": "Unable to retrieve sensor data."}
        return _ok(rows, rows[-1].get("Target_Crop")) if rows else _no_data(cleaned_node)

    if source == "hardware_unconfigured":
        return {"status": "unavailable", "reason": "Hardware Supabase credentials are not configured."}

    if source == "hardware_firebase":
        try:
            rows = firebase_hardware.fetch_hardware_rows(cleaned_node, limit=limit)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Hardware node window query failed for %s: %s", cleaned_node, exc)
            return {"status": "unavailable", "reason": "Unable to retrieve sensor data."}
        # The gateway firmware does not record a planting.
        return _ok(rows, None) if rows else _no_data(cleaned_node)

    if create_client is None:
        return {"status": "unavailable", "reason": "Supabase package is not installed."}
    client = get_supabase_client()
    if client is None:
        return {"status": "unavailable", "reason": "Supabase credentials are not configured."}

    try:
        # select("*") is intentional: PostgREST needs special quoting for the
        # production columns containing '%'.
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
        return _no_data(cleaned_node)

    # Supabase returned newest-first; reverse to chronological (oldest-first).
    ordered = list(reversed(rows))
    return _ok(ordered, ordered[-1].get("Target_Crop"))


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
    if not cleaned_node:
        return {"status": "unavailable", "reason": "node_id is required."}
    limit = max(int(limit), 1)

    cached = _cache_get(cleaned_node, limit)
    if cached is not None:
        return cached

    window = _fetch_uncached(cleaned_node, limit)
    _cache_put(cleaned_node, limit, window)
    return window


def prefetch_simulator_windows(node_ids: list[str], limit: int) -> None:
    """Warm the cache for several simulator nodes with one database query.

    Simulator nodes report on the same schedule, so the newest
    ``limit × nodes`` rows contain about ``limit`` rows for each of them. A
    node that turns out to have fewer rows in the shared page is left for
    ``fetch_node_window`` to query individually.
    """
    wanted = [
        node
        for node in dict.fromkeys(str(value).strip().upper() for value in node_ids)
        if node and telemetry_source(node) == "simulator" and _cache_get(node, limit) is None
    ]
    if len(wanted) < 2:
        return
    client = get_supabase_client()
    if client is None:
        return
    page_size = limit * len(wanted)
    try:
        result = (
            client.table(FARM_DATA_TABLE)
            .select("*")
            .in_("Node_ID", wanted)
            .order("Timestamp", desc=True)
            .limit(page_size)
            .execute()
        )
        rows = getattr(result, "data", None) or []
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("Shared simulator window query failed: %s", exc)
        return

    grouped: dict[str, list[dict[str, Any]]] = {node: [] for node in wanted}
    for row in rows:
        node = str(row.get("Node_ID") or "").strip().upper()
        if node in grouped and len(grouped[node]) < limit:
            grouped[node].append(row)

    page_was_full = len(rows) >= page_size
    for node, newest_first in grouped.items():
        if len(newest_first) < limit and page_was_full:
            continue  # The shared page may have cut this node's history short.
        if not newest_first:
            _cache_put(node, limit, _no_data(node))
            continue
        ordered = list(reversed(newest_first))
        _cache_put(node, limit, _ok(ordered, ordered[-1].get("Target_Crop")))


def list_active_node_ids(recent_rows: int = 500) -> list[str]:
    """Return every node that currently has telemetry, hardware nodes first."""
    global _active_nodes_cache
    cached = _active_nodes_cache
    if cached is not None and time.monotonic() - cached[0] <= CACHE_TTL_SECONDS:
        return list(cached[1])
    nodes = _list_active_node_ids(recent_rows)
    if CACHE_TTL_SECONDS > 0:
        _active_nodes_cache = (time.monotonic(), nodes)
    return list(nodes)


def _list_active_node_ids(recent_rows: int) -> list[str]:
    nodes: list[str] = []
    for node in sorted(supabase_hardware.HARDWARE_NODE_IDS):
        if telemetry_source(node) != "hardware_unconfigured":
            nodes.append(node)

    client = get_supabase_client()
    if client is not None:
        try:
            result = (
                client.table(FARM_DATA_TABLE)
                .select("Node_ID")
                .order("Timestamp", desc=True)
                .limit(recent_rows)
                .execute()
            )
            seen = {
                str(row.get("Node_ID") or "").strip().upper()
                for row in (getattr(result, "data", None) or [])
            }
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Active node query failed: %s", exc)
            seen = set()
        # Old simulated rows for hardware node IDs are not live hardware data.
        nodes.extend(sorted(seen - set(supabase_hardware.HARDWARE_NODE_IDS) - {""}))
    return nodes


def declared_crop(node_id: str) -> str | None:
    """Return the crop declared for a node in ``crop_node_labels.json``.

    Hardware readings carry no planting, so the project's node-to-crop label
    file is the recorded planting for those nodes.
    """
    from backend.ml import crop_training_data  # local import avoids a cycle

    try:
        return crop_training_data.load_node_crop_map().get(str(node_id).strip().upper())
    except (FileNotFoundError, ValueError):
        return None


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
