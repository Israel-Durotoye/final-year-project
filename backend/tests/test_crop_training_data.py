from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.ml import crop_training_data


class _PagedSupabaseClient:
    def __init__(self, pages: list[list[dict[str, object]]]) -> None:
        self.pages = iter(pages)
        self.ranges: list[tuple[int, int]] = []
        self.node_filter: list[str] = []

    def table(self, _name: str) -> "_PagedSupabaseClient":
        return self

    def select(self, _columns: str) -> "_PagedSupabaseClient":
        return self

    def in_(self, _column: str, values: list[str]) -> "_PagedSupabaseClient":
        self.node_filter = values
        return self

    def order(self, _column: str) -> "_PagedSupabaseClient":
        return self

    def range(self, start: int, end: int) -> "_PagedSupabaseClient":
        self.ranges.append((start, end))
        return self

    def execute(self) -> SimpleNamespace:
        return SimpleNamespace(data=next(self.pages))


class CropTrainingDataTests(unittest.TestCase):
    def test_historical_csv_adds_labelled_crop_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "crops.csv"
            data_path.write_text(
                "timestamp,temperature,humidity,moisture,N,P,K,pH_Value,lat,lon,Crop\n"
                "2026-01-01 00:00:00,25,70,45,80,40,60,6.5,9.5,6.4,Groundnut\n",
                encoding="utf-8",
            )

            rows = crop_training_data.load_historical_crop_rows(data_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Node_ID"], "HIST_GROUNDNUT")
        self.assertEqual(rows[0]["Target_Crop"], "Groundnut")
        self.assertEqual(rows[0]["Nitrogen_mg_k"], "80")
        self.assertEqual(rows[0]["Data_Source"], "historical_csv")

    def test_crop_mapping_can_be_extended_without_code_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mapping_path = Path(temporary_directory) / "labels.json"
            mapping_path.write_text(
                json.dumps({"node_01": "Maize", "node_07": "Groundnut"}),
                encoding="utf-8",
            )

            mapping = crop_training_data.load_node_crop_map(mapping_path)

        self.assertEqual(mapping, {"NODE_01": "Maize", "NODE_07": "Groundnut"})

    def test_supabase_loader_pages_until_every_simulator_row_is_read(self) -> None:
        client = _PagedSupabaseClient([
            [
                {"Node_ID": "NODE_04", "Timestamp": "2026-09-01T00:01:00Z"},
                {"Node_ID": "NODE_05", "Timestamp": "2026-09-01T00:02:00Z"},
            ],
            [{"Node_ID": "NODE_06", "Timestamp": "2026-09-01T00:03:00Z"}],
        ])

        rows = crop_training_data.fetch_all_simulator_rows(client=client, page_size=2)

        self.assertEqual(len(rows), 3)
        self.assertEqual(client.ranges, [(0, 1), (2, 3)])
        self.assertEqual(set(client.node_filter), crop_training_data.SIMULATOR_NODE_IDS)

    def test_combined_loader_labels_and_identifies_both_sources(self) -> None:
        simulator = [{
            "Node_ID": "NODE_04",
            "Timestamp": "2026-09-02T09:28:00Z",
            "Target_Crop": "Outdated database label",
        }]
        hardware = [{
            "Node_ID": "NODE_01",
            "Timestamp": "2026-09-02T09:27:59Z",
            "Nitrogen_mg_k": 120.0,
        }]
        with (
            patch.object(crop_training_data, "fetch_all_simulator_rows", return_value=simulator),
            patch.object(
                crop_training_data.firebase_hardware,
                "fetch_all_hardware_rows",
                return_value=hardware,
            ),
        ):
            rows = crop_training_data.fetch_all_crop_training_rows()

        self.assertEqual([row["Node_ID"] for row in rows], ["NODE_01", "NODE_04"])
        self.assertEqual(rows[0]["Target_Crop"], "Maize")
        self.assertEqual(rows[0]["Data_Source"], "hardware")
        self.assertEqual(rows[1]["Target_Crop"], "Cassava")
        self.assertEqual(rows[1]["Data_Source"], "simulator")


if __name__ == "__main__":
    unittest.main()
