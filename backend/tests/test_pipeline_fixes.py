"""Regression tests for the telemetry gateway, screening scales and API fixes."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from backend.api.routes import chat as chat_route
from backend.ml import keras_compat, node_data, soil_health
from backend.rag import chat_llm
from backend.rag.prescriptions import FarmRecommendationPlanner
from backend.rag.rag_engine import RAGEngine
from backend.utils.season import get_nigerian_season


def sim_rows(node_id: str, count: int) -> list[dict]:
    """Newest first, as Supabase returns them."""
    return [
        {"Node_ID": node_id, "Timestamp": f"2026-09-01T10:{minute:02d}:00", "Target_Crop": "Rice", "Moisture_%": 90.0}
        for minute in range(count - 1, -1, -1)
    ]


class _Query:
    """Minimal Supabase query-builder stand-in that records each execute()."""

    def __init__(self, client: "_Client") -> None:
        self.client = client
        self.nodes: list[str] | None = None
        self.row_limit = 0

    def select(self, _columns):
        return self

    def eq(self, _column, value):
        self.nodes = [value]
        return self

    def in_(self, _column, values):
        self.nodes = list(values)
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.row_limit = value
        return self

    def execute(self):
        self.client.queries.append((tuple(self.nodes or ()), self.row_limit))
        rows = [row for node in (self.nodes or []) for row in self.client.rows.get(node, [])]
        rows.sort(key=lambda row: row["Timestamp"], reverse=True)
        return SimpleNamespace(data=rows[: self.row_limit])


class _Client:
    def __init__(self, rows: dict[str, list[dict]]) -> None:
        self.rows = rows
        self.queries: list[tuple[tuple[str, ...], int]] = []

    def table(self, _name):
        return _Query(self)


class TelemetryGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _Client({"NODE_04": sim_rows("NODE_04", 120), "NODE_05": sim_rows("NODE_05", 120)})
        for target, value in (("get_supabase_client", lambda: self.client), ("create_client", object())):
            patcher = patch.object(node_data, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_repeated_and_smaller_requests_reuse_one_query(self) -> None:
        first = node_data.fetch_node_window("NODE_04", limit=100)
        second = node_data.fetch_node_window("node_04", limit=24)
        self.assertEqual(len(self.client.queries), 1)
        self.assertEqual(first["count"], 100)
        self.assertEqual(second["count"], 24)
        self.assertEqual(second["latest"], first["latest"])
        self.assertLess(second["rows"][0]["Timestamp"], second["rows"][-1]["Timestamp"])

    def test_larger_request_is_not_served_from_a_smaller_cached_window(self) -> None:
        node_data.fetch_node_window("NODE_04", limit=24)
        self.assertEqual(node_data.fetch_node_window("NODE_04", limit=100)["count"], 100)
        self.assertEqual(len(self.client.queries), 2)

    def test_one_shared_query_serves_every_simulator_node(self) -> None:
        node_data.prefetch_simulator_windows(["NODE_04", "NODE_05"], 100)
        for node in ("NODE_04", "NODE_05"):
            window = node_data.fetch_node_window(node, limit=100)
            self.assertEqual(window["count"], 100)
            self.assertEqual({row["Node_ID"] for row in window["rows"]}, {node})
        self.assertEqual(self.client.queries, [(("NODE_04", "NODE_05"), 200)])

    def test_hardware_node_without_a_source_is_unavailable_not_simulated(self) -> None:
        self.client.rows["NODE_03"] = sim_rows("NODE_03", 50)  # Old simulated rows under a hardware ID.
        with patch.object(node_data.supabase_hardware, "is_configured", return_value=False):
            self.assertEqual(node_data.telemetry_source("NODE_03"), "hardware_unconfigured")
            self.assertEqual(node_data.fetch_node_window("NODE_03")["status"], "unavailable")
            self.assertEqual(node_data.telemetry_source("NODE_01"), "hardware_firebase")
        self.assertEqual(self.client.queries, [])


class ScreeningScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = FarmRecommendationPlanner()

    @staticmethod
    def node(**overrides):
        reading = {
            "node_id": "NODE_05", "timestamp_utc": "2026-09-01T10:00:00+00:00",
            "currently_planted_crop": "Rice", "season": "Peak Rainy",
            "nitrogen_mg_kg": 100.0, "phosphorus_mg_kg": 30.0, "potassium_mg_kg": 45.0,
            "moisture_pct": 92.0, "temperature_c": 27.0, "humidity_pct": 80.0,
        }
        reading.update(overrides)
        return reading

    def test_flooded_paddy_rice_is_not_reported_as_waterlogged(self) -> None:
        brief = self.planner.build_node_brief(self.node())
        self.assertIn("rice", brief["screening_basis"])
        self.assertEqual(brief["priorities"], [])
        self.assertTrue(brief["healthy_sensor_state"])

    def test_dry_paddy_rice_is_a_water_deficit(self) -> None:
        brief = self.planner.build_node_brief(self.node(moisture_pct=55.0))
        self.assertEqual([item["domain"] for item in brief["priorities"]], ["water_deficit"])

    def test_cassava_in_its_reference_range_needs_no_correction(self) -> None:
        brief = self.planner.build_node_brief(self.node(
            currently_planted_crop="Cassava", nitrogen_mg_kg=70.0, phosphorus_mg_kg=20.0,
            potassium_mg_kg=125.0, moisture_pct=50.0, temperature_c=30.0, humidity_pct=60.0,
        ))
        self.assertIn("cassava", brief["screening_basis"])
        self.assertEqual(brief["priorities"], [])

    def test_unknown_crop_is_screened_on_the_sensor_moisture_scale(self) -> None:
        normal = self.planner.build_node_brief(self.node(currently_planted_crop=None, moisture_pct=55.0))
        self.assertTrue(normal["screening_basis"].startswith("generic"))
        self.assertNotIn("excess_water", [item["domain"] for item in normal["priorities"]])
        wet = self.planner.build_node_brief(self.node(currently_planted_crop=None, moisture_pct=96.0))
        self.assertIn("excess_water", [item["domain"] for item in wet["priorities"]])

    def test_crop_names_resolve_to_profiles(self) -> None:
        self.assertEqual(soil_health.resolve_crop_profile("Paddy Rice"), "rice")
        self.assertEqual(soil_health.resolve_crop_profile(" cassava "), "cassava")
        self.assertEqual(soil_health.resolve_crop_profile("Maize/Corn"), "maize_corn")
        self.assertIsNone(soil_health.resolve_crop_profile("Sorghum"))
        self.assertIsNone(soil_health.resolve_crop_profile(None))


class SeasonTests(unittest.TestCase):
    def test_fractional_second_timestamps_use_the_reading_date(self) -> None:
        # These used to fall back silently to today's season.
        self.assertEqual(get_nigerian_season("2026-01-18T10:07:04.368000Z"), "Dry (Harmattan)")
        self.assertEqual(get_nigerian_season("2026-08-18T10:07:04.368+00:00"), "Peak Rainy")
        self.assertEqual(get_nigerian_season("2026-04-02 09:00:00"), "Early Rainy")


class SnapshotTests(unittest.TestCase):
    def test_snapshot_uses_reading_time_season_and_declared_crop(self) -> None:
        row = {
            "Node_ID": "NODE_01", "Timestamp": "2026-08-18T10:07:04.368000Z", "Season": "Dry",
            "Nitrogen_mg_k": 40.0, "Phosphorus_m": 20.0, "Potassium_mg_": 150.0,
            "Moisture_%": 51.0, "Temperature_C": 25.3, "Humidity_%": 60.0, "Soil_pH": 6.4,
        }
        window = {"status": "ok", "rows": [row], "latest": row, "crop": None, "count": 1}
        with (
            patch.object(node_data, "list_active_node_ids", return_value=["NODE_01"]),
            patch.object(node_data, "prefetch_simulator_windows"),
            patch.object(node_data, "fetch_node_window", return_value=window),
            patch.object(node_data, "declared_crop", return_value="Maize"),
        ):
            snapshot = chat_llm._get_farm_snapshot()
        node = snapshot["nodes"][0]
        self.assertEqual(snapshot["status"], "online")
        self.assertEqual(node["season"], "Peak Rainy")
        self.assertEqual(node["currently_planted_crop"], "Maize")
        self.assertEqual(node["crop_source"], "declared_node_label")
        self.assertEqual(node["soil_ph"], 6.4)
        self.assertNotIn("ai_predicted_ideal_crop", node)  # One reading is not a model window.

    def test_snapshot_reports_unavailable_when_no_node_can_be_read(self) -> None:
        failure = {"status": "unavailable", "reason": "Supabase credentials are not configured."}
        with (
            patch.object(node_data, "list_active_node_ids", return_value=["NODE_04"]),
            patch.object(node_data, "prefetch_simulator_windows"),
            patch.object(node_data, "fetch_node_window", return_value=failure),
        ):
            snapshot = chat_llm._get_farm_snapshot()
        self.assertEqual(snapshot["status"], "unavailable")
        self.assertIn("credentials", snapshot["reason"])


class RetrievalTests(unittest.TestCase):
    def test_keyword_search_returns_nothing_when_no_term_matches(self) -> None:
        engine = RAGEngine()
        engine._bm25_index = MagicMock()
        engine._bm25_index.get_scores.return_value = np.zeros(3)
        engine._bm25_doc_ids = ["a", "b", "c"]
        engine._collection = MagicMock()
        self.assertEqual(engine._sparse_search("zzzz", n_candidates=2), [])
        engine._collection.get.assert_not_called()

    def test_keyword_search_drops_documents_with_no_shared_term(self) -> None:
        engine = RAGEngine()
        engine._bm25_index = MagicMock()
        engine._bm25_index.get_scores.return_value = np.array([0.0, 2.0, 0.0])
        engine._bm25_doc_ids = ["a", "b", "c"]
        engine._collection = MagicMock()
        engine._collection.get.return_value = {"ids": ["b"], "documents": ["text"], "metadatas": [{"source_name": "Guide"}]}
        results = engine._sparse_search("lime", n_candidates=3)
        self.assertEqual([item["chunk_id"] for item in results], ["b"])
        self.assertEqual(engine._collection.get.call_args.kwargs["ids"], ["b"])


class ChatRouteTests(unittest.TestCase):
    def test_long_stored_answers_do_not_break_history(self) -> None:
        long_answer = "x" * 6000  # A field report is longer than the old 4000-character limit.
        message = chat_route.ChatMessage(role="assistant", content=long_answer)
        self.assertEqual(len(message.content), 6000)
        request = chat_route.ChatRequest(query="Next step?", history=[message])
        self.assertEqual(len(request.history), 1)

    def test_history_is_limited_to_fifty_messages(self) -> None:
        messages = [chat_route.ChatMessage(role="user", content="hi")] * 51
        with self.assertRaises(ValueError):
            chat_route.ChatRequest(query="hello", history=messages)

    def test_chat_endpoint_runs_in_a_worker_thread(self) -> None:
        import inspect

        self.assertFalse(inspect.iscoroutinefunction(chat_route.post_chat))


class KerasCompatTests(unittest.TestCase):
    def test_loader_uses_the_api_present_in_keras_2_and_3(self) -> None:
        fake_keras = SimpleNamespace(models=SimpleNamespace(load_model=MagicMock(return_value="model")))
        with patch.dict("sys.modules", {"keras": fake_keras}):
            self.assertEqual(keras_compat.load_keras_model("model.keras"), "model")
        fake_keras.models.load_model.assert_called_once_with("model.keras", compile=False)


if __name__ == "__main__":
    unittest.main()
