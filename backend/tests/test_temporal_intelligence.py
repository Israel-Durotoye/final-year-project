"""Acceptance tests for temporal history, forecasting fallback, and chat evidence."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np

from backend.ml import temporal_service
from backend.ml.temporal_analysis import analyze_history
from backend.ml.temporal_data import FEATURE_COLUMNS, prepare_temporal_rows
from backend.ml.train_lstm_forecaster import _windows_for_partition
from backend.rag.chat_llm import (
    SYSTEM_INSTRUCTION,
    TOOLS,
    _build_user_content,
    _get_automatic_temporal_context,
)


def rows_for(
    node_id: str,
    moisture: list[float],
    *,
    nitrogen: list[float] | None = None,
    start: datetime | None = None,
    intervals_minutes: list[int] | None = None,
) -> list[dict]:
    start = start or datetime(2026, 8, 1, tzinfo=timezone.utc)
    nitrogen = nitrogen or [90.0] * len(moisture)
    timestamps = [start]
    for index in range(1, len(moisture)):
        minutes = intervals_minutes[index - 1] if intervals_minutes else 60
        timestamps.append(timestamps[-1] + timedelta(minutes=minutes))
    return [
        {
            "Timestamp": timestamp.isoformat(),
            "Node_ID": node_id,
            "Target_Crop": "Maize",
            "Nitrogen_mg_k": nitrogen[index],
            "Phosphorus_m": 25.0,
            "Potassium_mg_": 45.0,
            "Moisture_%": value,
            "Temperature_C": 27.0,
            "Humidity_%": 65.0,
        }
        for index, (timestamp, value) in enumerate(zip(timestamps, moisture))
    ]


class TemporalAcceptanceTests(unittest.TestCase):
    def test_a_stable_history_has_no_event_and_can_carry_stable_forecast(self) -> None:
        rows = rows_for("NODE_01", [60.0] * 60)
        fake_forecast = {
            "status": "success",
            "forecast": {"6h": {"moisture_pct": {"predicted": 60.1}}},
            "forecast_trends": {"moisture_pct": "stable"},
            "model": {"status": "available", "sequence_length": 48},
        }
        with (
            patch.object(
                temporal_service.lstm_forecaster,
                "artifact_status",
                return_value={"status": "available", "sequence_length": 48},
            ),
            patch.object(
                temporal_service.lstm_forecaster,
                "forecast",
                return_value=fake_forecast,
            ),
        ):
            result = temporal_service.get_temporal_farm_intelligence("NODE_01", rows=rows)

        self.assertEqual(result["historical_analysis"]["moisture"]["trend"], "stable")
        self.assertEqual(result["events"], [])
        self.assertAlmostEqual(result["forecast"]["6h"]["moisture_pct"]["predicted"], 60.1)

    def test_b_sustained_wetting_does_not_claim_rainfall(self) -> None:
        prepared = prepare_temporal_rows(
            rows_for("NODE_02", [48, 49, 51, 79, 82, 84, 83, 82]),
            node_id="NODE_02",
        )
        result = analyze_history(prepared)
        wetting = next(event for event in result["events"] if event["type"] == "sustained_wetting")
        self.assertEqual(wetting["likely_cause"], "external_water_input")
        self.assertEqual(wetting["cause_detail"], "rain_or_irrigation_unknown")
        self.assertNotIn("confirmed", json.dumps(wetting).lower())

    def test_c_isolated_moisture_spike_is_an_anomaly_not_wetting(self) -> None:
        prepared = prepare_temporal_rows(
            rows_for("NODE_03", [49, 50, 51, 99, 50, 51]),
            node_id="NODE_03",
        )
        result = analyze_history(prepared)
        self.assertTrue(prepared.data_quality["possible_anomalies"])
        self.assertFalse(any(event["type"] == "sustained_wetting" for event in result["events"]))

    def test_d_continuous_decrease_is_drying(self) -> None:
        prepared = prepare_temporal_rows(
            rows_for("NODE_04", [82, 80, 76, 70, 64, 58]),
            node_id="NODE_04",
        )
        result = analyze_history(prepared)
        self.assertIn(result["historical_analysis"]["moisture"]["trend"], {"falling", "strongly_falling"})
        self.assertTrue(any("drying" in event["type"] for event in result["events"]))

    def test_e_nitrogen_decline_has_no_fertilizer_recommendation(self) -> None:
        prepared = prepare_temporal_rows(
            rows_for(
                "NODE_05",
                [60] * 7,
                nitrogen=[102, 101, 100, 97, 95, 94, 92],
            ),
            node_id="NODE_05",
        )
        result = analyze_history(prepared)
        nitrogen = result["historical_analysis"]["nitrogen"]
        self.assertIn(nitrogen["trend"], {"falling", "strongly_falling"})
        self.assertEqual(nitrogen["event"], "sustained_decline")
        self.assertNotIn("fertil", json.dumps(result).lower())

    def test_f_irregular_timestamps_are_sorted_deduplicated_and_gaps_reported(self) -> None:
        rows = rows_for("NODE_06", [50, 51, 52, 53], intervals_minutes=[60, 60, 600])
        duplicate = dict(rows[1])
        duplicate["Nitrogen_mg_k"] = None
        mixed = [rows[3], duplicate, rows[0], rows[2], rows[1]]
        prepared = prepare_temporal_rows(mixed, node_id="NODE_06")
        timestamps = [row["Timestamp"] for row in prepared.rows]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(prepared.history["samples_used"], 4)
        self.assertEqual(prepared.data_quality["duplicate_timestamps_removed"], 1)
        self.assertEqual(prepared.data_quality["gap_count"], 1)
        self.assertEqual(prepared.history["contiguous_tail_samples"], 1)

    def test_g_insufficient_history_keeps_deterministic_analysis(self) -> None:
        result = temporal_service.get_temporal_farm_intelligence(
            "NODE_01", rows=rows_for("NODE_01", list(range(40, 71)))
        )
        self.assertEqual(result["status"], "historical_only")
        self.assertEqual(result["forecast_status"], "insufficient_history")
        self.assertIsNone(result["forecast"])
        self.assertTrue(result["historical_analysis"])

    def test_h_multi_node_histories_never_mix(self) -> None:
        row_sets = {
            "NODE_01": rows_for("NODE_01", [40, 41, 42, 43]),
            "NODE_02": rows_for("NODE_02", [80, 79, 78, 77]),
        }
        result = temporal_service.get_multi_node_temporal_intelligence(
            ["NODE_01", "NODE_02"], row_sets=row_sets
        )
        node_1 = result["nodes"]["NODE_01"]["historical_analysis"]["moisture"]
        node_2 = result["nodes"]["NODE_02"]["historical_analysis"]["moisture"]
        self.assertEqual(node_1["current"], 43.0)
        self.assertEqual(node_2["current"], 77.0)
        self.assertGreater(node_1["slope_per_hour"], 0)
        self.assertLess(node_2["slope_per_hour"], 0)

    def test_i_normal_current_and_predicted_deterioration_are_distinct_evidence(self) -> None:
        temporal = {
            "status": "success",
            "nodes": {
                "NODE_01": {
                    "status": "success",
                    "history": {"samples_used": 100},
                    "historical_analysis": {"moisture": {"current": 62, "trend": "rising"}},
                    "events": [],
                    "data_quality": {},
                    "forecast_status": "success",
                    "forecast": {"24h": {"moisture_pct": {"predicted": 75}}},
                    "forecast_trends": {"moisture_pct": "rising"},
                    "model": {"status": "available"},
                }
            },
        }
        prompt = _build_user_content(
            "Analyse NODE_01",
            "Retrieved agronomic evidence",
            True,
            {"status": "online", "nodes": [], "node_count": 0},
            temporal_context=temporal,
        )
        self.assertIn('"current": 62', prompt)
        self.assertIn('"predicted": 75', prompt)
        self.assertLess(prompt.index("TEMPORAL FARM ANALYSIS"), prompt.index("FUTURE SENSOR FORECAST"))
        self.assertIn("latest value is acceptable", SYSTEM_INSTRUCTION)

    def test_j_poor_current_and_predicted_recovery_remain_uncertain(self) -> None:
        lowered = SYSTEM_INSTRUCTION.lower()
        self.assertIn("forecast to recover", lowered)
        self.assertIn("uncertain recovery", lowered)
        self.assertIn("never be reported as events that already happened", lowered)

    def test_walk_forward_window_builder_keeps_future_out_of_inputs(self) -> None:
        matrix = np.arange(30, dtype=np.float32).reshape(15, 2)
        x_values, y_values = _windows_for_partition(
            matrix,
            target_start=8,
            target_end=15,
            sequence_length=4,
            forecast_steps=3,
        )
        self.assertTrue(x_values)
        for x_window, y_window in zip(x_values, y_values):
            self.assertLess(float(x_window[-1, 0]), float(y_window[0, 0]))
            self.assertEqual(x_window.shape, (4, 2))
            self.assertEqual(y_window.shape, (3, 2))

    def test_exact_production_feature_order(self) -> None:
        self.assertEqual(
            FEATURE_COLUMNS,
            (
                "Nitrogen_mg_k",
                "Phosphorus_m",
                "Potassium_mg_",
                "Moisture_%",
                "Temperature_C",
                "Humidity_%",
            ),
        )

    def test_active_agent_tools_use_temporal_analysis_not_legacy_classifiers(self) -> None:
        names = {tool["function"]["name"] for tool in TOOLS}
        self.assertIn("analyze_temporal_conditions", names)
        self.assertNotIn("execute_moisture_prediction", names)
        self.assertNotIn("classify_soil_suitability", names)

    def test_selected_node_temporal_context_is_acquired_before_generation(self) -> None:
        snapshot = {
            "status": "online",
            "nodes": [{"node_id": "NODE_02"}, {"node_id": "NODE_03"}],
        }
        with patch(
            "backend.rag.chat_llm.temporal_service.get_multi_node_temporal_intelligence",
            return_value={"status": "success", "nodes_analyzed": 1, "nodes": {}},
        ) as service:
            result = _get_automatic_temporal_context(
                snapshot,
                "Analyse this selected field",
                requested_node_id="NODE_03",
            )
        service.assert_called_once_with(["NODE_03"])
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
