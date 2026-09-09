"""Load the complete mixed-source dataset used for crop model training."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from backend.ml import firebase_hardware


DEFAULT_NODE_CROP_MAP = {
    "NODE_01": "Maize",
    "NODE_02": "Maize",
    "NODE_03": "Cassava",
    "NODE_04": "Cassava",
    "NODE_05": "Rice",
    "NODE_06": "Rice",
}
DEFAULT_CROP_MAP_PATH = Path(__file__).with_name("crop_node_labels.json")
DEFAULT_HISTORICAL_DATA_PATH = Path(__file__).resolve().parents[2] / "pipeline_ready_sensor_data.csv"
DEFAULT_TABLE = os.getenv("FARM_DATA_TABLE", "capstone_dataset")


def load_node_crop_map(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Load the editable node-to-crop mapping used to label training rows."""
    configured_path = path or os.getenv("CROP_NODE_LABELS_PATH")
    mapping_path = Path(configured_path) if configured_path else DEFAULT_CROP_MAP_PATH

    if mapping_path.exists():
        with mapping_path.open(encoding="utf-8") as mapping_file:
            raw_mapping = json.load(mapping_file)
    elif configured_path:
        raise FileNotFoundError(f"Crop label mapping not found: {mapping_path}")
    else:
        raw_mapping = DEFAULT_NODE_CROP_MAP

    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("The crop label mapping must be a non-empty JSON object.")

    mapping: dict[str, str] = {}
    for raw_node_id, raw_crop in raw_mapping.items():
        node_id = str(raw_node_id).strip().upper()
        crop = str(raw_crop).strip()
        if not node_id or not crop:
            raise ValueError("Every crop label mapping entry needs a node ID and crop name.")
        mapping[node_id] = crop
    return mapping


NODE_CROP_MAP = load_node_crop_map()
SIMULATOR_NODE_IDS = set(NODE_CROP_MAP) - set(firebase_hardware.PHYSICAL_NODE_IDS)


def load_historical_crop_rows(
    path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Load the local labelled crop dataset into the application schema."""
    configured_path = path or os.getenv("CROP_HISTORICAL_DATA_PATH")
    data_path = Path(configured_path) if configured_path else DEFAULT_HISTORICAL_DATA_PATH
    if not data_path.exists():
        if configured_path:
            raise FileNotFoundError(f"Historical crop dataset not found: {data_path}")
        return []

    column_map = {
        "timestamp": "Timestamp",
        "temperature": "Temperature_C",
        "humidity": "Humidity_%",
        "moisture": "Moisture_%",
        "N": "Nitrogen_mg_k",
        "P": "Phosphorus_m",
        "K": "Potassium_mg_",
        "pH_Value": "Soil_pH",
        "lat": "Latitude",
        "lon": "Longitude",
    }
    rows: list[dict[str, Any]] = []
    with data_path.open(newline="", encoding="utf-8") as data_file:
        for source_row in csv.DictReader(data_file):
            crop = str(source_row.get("Crop") or "").strip()
            if not crop:
                continue
            slug = re.sub(r"[^A-Z0-9]+", "_", crop.upper()).strip("_")
            row = {
                destination: source_row.get(source)
                for source, destination in column_map.items()
            }
            row.update({
                "Node_ID": f"HIST_{slug}",
                "Target_Crop": crop,
                "Data_Source": "historical_csv",
            })
            rows.append(row)
    return rows


def _response_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    return [row for row in (data or []) if isinstance(row, dict)]


def _create_supabase_client() -> Any:
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("The supabase package is required to load simulator training data.") from exc

    url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials are missing.")
    return create_client(url, key)


def fetch_all_simulator_rows(
    *,
    client: Any | None = None,
    table: str = DEFAULT_TABLE,
    page_size: int = 1000,
    node_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch every simulator row using PostgREST range pagination."""
    if page_size < 1:
        raise ValueError("page_size must be at least 1.")
    client = client or _create_supabase_client()
    selected_node_ids = node_ids if node_ids is not None else SIMULATOR_NODE_IDS
    if not selected_node_ids:
        return []
    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        response = (
            client.table(table)
            .select("*")
            .in_("Node_ID", sorted(selected_node_ids))
            .order("Timestamp")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = _response_rows(response)
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)

    return rows


def fetch_all_crop_training_rows(
    *,
    supabase_client: Any | None = None,
    page_size: int = 1000,
    node_crop_map: dict[str, str] | None = None,
    include_historical: bool = False,
    historical_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Merge all physical and simulated records into the application schema."""
    crop_map = node_crop_map or load_node_crop_map()
    simulator_node_ids = set(crop_map) - set(firebase_hardware.PHYSICAL_NODE_IDS)
    hardware_node_ids = set(crop_map) & set(firebase_hardware.PHYSICAL_NODE_IDS)
    simulator_rows = fetch_all_simulator_rows(
        client=supabase_client,
        page_size=page_size,
        node_ids=simulator_node_ids,
    )
    hardware_rows = firebase_hardware.fetch_all_hardware_rows(page_size=page_size)

    merged: list[dict[str, Any]] = []
    for source, source_rows, allowed_nodes in (
        ("simulator", simulator_rows, simulator_node_ids),
        ("hardware", hardware_rows, hardware_node_ids),
    ):
        for original in source_rows:
            row = dict(original)
            node_id = str(row.get("Node_ID") or "").strip().upper()
            if node_id not in allowed_nodes:
                continue
            row["Node_ID"] = node_id
            row["Target_Crop"] = crop_map[node_id]
            row["Data_Source"] = source
            merged.append(row)

    if include_historical:
        merged.extend(load_historical_crop_rows(historical_path))

    merged.sort(key=lambda row: (str(row.get("Timestamp") or ""), str(row["Node_ID"])))
    return merged
