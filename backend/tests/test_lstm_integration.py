"""The LSTM models must reach the chat prompt, the retrieval query and the API."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.ml import lstm_anomaly_inference, lstm_forecaster, temporal_service
from backend.ml.temporal_data import FEATURE_COLUMNS, latest_complete_window
from backend.ml.train_lstm_forecaster import (
    baseline_predictions,
    fit_training_scaler,
    make_chronological_windows,
    skill_against_baselines,
)
from backend.rag import chat_llm


def rows_for(node_id: str, count: int, *, crop: str = "Maize", moisture: float = 60.0) -> list[dict]:
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    return [
        {
            "Timestamp": (start + timedelta(minutes=index)).isoformat(),
            "Node_ID": node_id,
            "Target_Crop": crop,
            "Nitrogen_mg_k": 30.0,
            "Phosphorus_m": 20.0,
            "Potassium_mg_": 150.0,
            "Moisture_%": moisture,
            "Temperature_C": 27.0,
            "Humidity_%": 65.0,
        }
        for index in range(count)
    ]


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=np.float32)


class _ShiftedReconstruction:
    """Autoencoder stand-in that misses every value by a fixed amount."""

    def __init__(self, shift: float) -> None:
        self.shift = shift

    def predict(self, batch, verbose=0):
        return np.asarray(batch) + self.shift


class AnomalyScreenTests(unittest.TestCase):
    METADATA = {
        "feature_order": list(FEATURE_COLUMNS),
        "sequence_length": 24,
        "threshold": 0.5,
        "feature_thresholds": {
            "nitrogen": 0.5, "phosphorus": 0.5, "potassium": 0.5,
            "moisture": 0.5, "temperature": 0.5, "humidity": 0.5,
        },
    }

    def score(self, shift: float, rows: list[dict]) -> dict:
        artifacts = (_ShiftedReconstruction(shift), _IdentityScaler(), self.METADATA)
        with patch.object(lstm_anomaly_inference, "_load_artifacts", return_value=artifacts):
            return lstm_anomaly_inference.score_rows(rows)

    def test_well_reconstructed_window_is_normal(self) -> None:
        result = self.score(0.1, rows_for("NODE_04", 30))
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["is_anomalous"])
        self.assertEqual(result["severity"], "normal")
        self.assertEqual(result["readings_used"], 24)

    def test_poorly_reconstructed_window_is_flagged_with_severity(self) -> None:
        result = self.score(1.2, rows_for("NODE_04", 30))
        self.assertTrue(result["is_anomalous"])
        self.assertEqual(result["severity"], "high")
        self.assertEqual(len(result["unusual_sensors"]), 6)
        self.assertIn("not as a diagnosis", result["interpretation"])

    def test_short_or_incomplete_history_is_never_scored(self) -> None:
        self.assertEqual(self.score(1.2, rows_for("NODE_04", 10))["status"], "insufficient_history")
        rows = rows_for("NODE_04", 30)
        rows[-1]["Moisture_%"] = None
        self.assertEqual(self.score(1.2, rows)["status"], "incomplete_window")

    def test_uncalibrated_model_reports_its_state(self) -> None:
        with patch.object(lstm_anomaly_inference, "METADATA_PATH", lstm_anomaly_inference.MODEL_PATH.with_name("missing.json")):
            self.assertEqual(lstm_anomaly_inference.artifact_status()["status"], "not_calibrated")


class TemporalServiceLstmTests(unittest.TestCase):
    def analyse(self, rows, forecast=None, anomaly=None):
        forecast = forecast or {
            "status": "success",
            "forecast": {"48m": {"moisture_pct": {"predicted": 91.0}, "temperature_c": {"predicted": 27.0}}},
            "forecast_trends": {"moisture_pct": "rising"},
            "forecast_skill": {"status": "evaluated", "informative_sensors": [], "level_estimate_only_sensors": ["moisture_pct"]},
            "model": {"status": "available", "sequence_length": 48},
        }
        with (
            patch.object(lstm_forecaster, "artifact_status", return_value={"status": "available", "sequence_length": 48}),
            patch.object(lstm_forecaster, "forecast", return_value=forecast) as forecast_call,
            patch.object(temporal_service, "_anomaly_screening", return_value=anomaly or {"status": "success", "is_anomalous": False}),
        ):
            result = temporal_service.get_temporal_farm_intelligence("NODE_09", rows=rows)
        return result, forecast_call

    def test_old_missing_value_does_not_block_the_forecast(self) -> None:
        rows = rows_for("NODE_09", 80)
        for index in (3, 4, 5):  # Too long a run to interpolate, but outside the model window.
            rows[index]["Moisture_%"] = None
        result, forecast_call = self.analyse(rows)
        self.assertEqual(result["forecast_status"], "success")
        self.assertEqual(len(forecast_call.call_args.args[0]), 48)

    def test_missing_value_inside_the_model_window_blocks_it(self) -> None:
        rows = rows_for("NODE_09", 80)
        for index in (70, 71, 72):
            rows[index]["Moisture_%"] = None
        result, forecast_call = self.analyse(rows)
        self.assertEqual(result["forecast_status"], "invalid_input")
        forecast_call.assert_not_called()

    def test_forecast_outside_the_crop_range_becomes_a_risk(self) -> None:
        result, _ = self.analyse(rows_for("NODE_09", 80, crop="Maize"))
        risks = result["forecast_outlook"]["risks"]
        self.assertEqual([risk["sensor"] for risk in risks], ["moisture"])
        self.assertEqual(risks[0]["direction"], "above_reference_range")
        self.assertIn("weak evidence", risks[0]["forecast_reliability"])

    def test_wet_forecast_is_not_a_risk_for_paddy_rice(self) -> None:
        result, _ = self.analyse(rows_for("NODE_09", 80, crop="Rice", moisture=90.0))
        self.assertEqual(result["forecast_outlook"]["crop_profile"], "rice")
        self.assertEqual(result["forecast_outlook"]["risks"], [])

    def test_anomaly_screen_is_part_of_the_result(self) -> None:
        flagged = {"status": "success", "is_anomalous": True, "unusual_sensors": ["humidity"]}
        result, _ = self.analyse(rows_for("NODE_09", 80), anomaly=flagged)
        self.assertEqual(result["anomaly_screening"], flagged)


class ForecasterTrainingTests(unittest.TestCase):
    def test_one_missing_reading_only_removes_the_windows_that_contain_it(self) -> None:
        segment = rows_for("NODE_09", 200)
        for index, row in enumerate(segment):
            row["Moisture_%"] = 50.0 + index % 7
        segment[100]["Moisture_%"] = None
        segments = {"NODE_09": [segment]}
        scaler = fit_training_scaler(segments)
        windows = make_chronological_windows(segments, scaler, sequence_length=12, forecast_steps=6)
        self.assertGreater(len(windows["train"][0]), 80)
        self.assertTrue(np.isfinite(windows["train"][0]).all())
        self.assertGreater(len(windows["validation"][0]), 0)
        self.assertGreater(len(windows["test"][0]), 0)

    def test_skill_requires_a_margin_over_the_best_naive_baseline(self) -> None:
        inputs = np.ones((5, 4, len(FEATURE_COLUMNS)), dtype=np.float32)
        actual = np.full((5, 3, len(FEATURE_COLUMNS)), 2.0, dtype=np.float32)
        baselines = baseline_predictions(inputs, 3)
        self.assertEqual(set(baselines), {"persistence", "window_mean"})
        perfect = skill_against_baselines(actual, actual.copy(), baselines)
        self.assertTrue(all(entry["informative"] for entry in perfect["per_sensor"].values()))
        no_better = skill_against_baselines(actual, baselines["persistence"], baselines)
        self.assertFalse(any(entry["informative"] for entry in no_better["per_sensor"].values()))

    def test_forecast_skill_summary_separates_informative_sensors(self) -> None:
        summary = lstm_forecaster.summarize_skill({"skill_vs_baseline": {"per_sensor": {
            "moisture_pct": {"informative": True, "lstm_mae": 1.0, "best_baseline": "persistence", "best_baseline_mae": 2.0, "improvement_pct": 50.0},
            "humidity_pct": {"informative": False, "lstm_mae": 3.0, "best_baseline": "window_mean", "best_baseline_mae": 3.0, "improvement_pct": 0.0},
        }}})
        self.assertEqual(summary["informative_sensors"], ["moisture_pct"])
        self.assertEqual(summary["level_estimate_only_sensors"], ["humidity_pct"])
        self.assertEqual(lstm_forecaster.summarize_skill({})["status"], "not_evaluated")

    def test_change_from_last_reading_targets_round_trip(self) -> None:
        inputs = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
        futures = inputs[:, -1:, :] + np.arange(1, 4, dtype=np.float32).reshape(1, 3, 1)
        targets = lstm_forecaster.to_targets(inputs, futures, "delta_from_last")
        self.assertTrue(np.allclose(targets[0, :, 0], [1, 2, 3]))
        self.assertTrue(np.allclose(lstm_forecaster.from_outputs(inputs, targets, "delta_from_last"), futures))
        self.assertTrue(np.allclose(lstm_forecaster.to_targets(inputs, futures, "absolute"), futures))

    def test_zero_network_output_means_no_change_in_delta_mode(self) -> None:
        inputs = np.full((1, 4, 6), 0.4, dtype=np.float32)
        model = SimpleNamespace(predict=lambda batch, verbose=0: np.zeros((1, 3, 6), dtype=np.float32))
        delta = lstm_forecaster.predict_scaled(model, inputs, {"target_mode": "delta_from_last"})
        self.assertTrue(np.allclose(delta, 0.4))
        legacy = lstm_forecaster.predict_scaled(model, inputs, {})  # Older artifacts have no target_mode.
        self.assertTrue(np.allclose(legacy, 0.0))

    def test_latest_window_helper_needs_only_the_newest_rows(self) -> None:
        rows = rows_for("NODE_09", 30)
        rows[0]["Humidity_%"] = None
        self.assertEqual(len(latest_complete_window(rows, 24)), 24)
        self.assertIsNone(latest_complete_window(rows, 30))
        self.assertIsNone(latest_complete_window(rows, 31))


class ChatEvidenceTests(unittest.TestCase):
    SNAPSHOT = {
        "status": "online",
        "node_count": 1,
        "nodes": [{
            "node_id": "NODE_06", "timestamp_utc": "2026-09-01T10:00:00+00:00",
            "currently_planted_crop": "Maize", "season": "Peak Rainy",
            "nitrogen_mg_kg": 30.0, "phosphorus_mg_kg": 20.0, "potassium_mg_kg": 150.0,
            "moisture_pct": 88.0, "temperature_c": 25.0, "humidity_pct": 65.0,
            "ai_predicted_ideal_crop": "Rice",
            "crop_recommendation": {"crop": "Rice", "confidence": 0.81},
        }],
    }
    TEMPORAL = {
        "status": "success",
        "nodes": {"NODE_06": {
            "forecast_status": "success",
            "forecast_trends": {"moisture_pct": "rising"},
            "forecast_skill": {"informative_sensors": ["moisture_pct"], "level_estimate_only_sensors": []},
            "forecast_outlook": {"risks": [{"sensor": "moisture", "direction": "above_reference_range"}]},
            "anomaly_screening": {"status": "success", "is_anomalous": True, "severity": "moderate", "unusual_sensors": ["humidity"]},
            "historical_analysis": {"nitrogen": {"trend": "falling"}},
            "events": [{"type": "sustained_wetting"}],
        }},
    }

    def test_prompt_carries_a_separate_lstm_evidence_block(self) -> None:
        prompt = chat_llm._build_user_content(
            "Analyse NODE_06", "Guidance", True, self.SNAPSHOT, temporal_context=self.TEMPORAL,
        )
        self.assertLess(prompt.index("FUTURE SENSOR FORECAST"), prompt.index("LSTM MODEL EVIDENCE"))
        self.assertLess(prompt.index("LSTM MODEL EVIDENCE"), prompt.index("FARM-LEVEL PRIORITY BRIEF"))
        block = prompt[prompt.index("LSTM MODEL EVIDENCE"):prompt.index("FARM-LEVEL PRIORITY BRIEF")]
        evidence = json.loads(block.splitlines()[3])
        self.assertTrue(evidence["NODE_06"]["anomaly_screen"]["is_anomalous"])
        self.assertEqual(evidence["NODE_06"]["crop_recommendation"]["recommended_crop"], "Rice")
        self.assertEqual(evidence["NODE_06"]["forecast"]["informative_sensors"], ["moisture_pct"])

    def test_system_instruction_limits_how_model_output_is_used(self) -> None:
        lowered = chat_llm.SYSTEM_INSTRUCTION.lower()
        self.assertIn("never a diagnosis", lowered)
        self.assertIn("weak evidence", lowered)
        self.assertIn("not evidence that the current crop is failing", lowered.replace("\n", " ").replace("  ", " "))

    def test_condition_query_names_what_the_rules_and_models_found(self) -> None:
        query = chat_llm.build_condition_query(self.SNAPSHOT, self.TEMPORAL, ["NODE_06"]).lower()
        for expected in ("maize", "waterlogging", "high moisture", "nitrogen declining", "sustained wetting", "sensor reading reliability", "peak rainy"):
            self.assertIn(expected, query)

    def test_condition_query_ignores_trends_without_a_management_meaning(self) -> None:
        temporal = {"nodes": {"NODE_06": {"historical_analysis": {
            "temperature": {"trend": "falling"}, "humidity": {"trend": "rising"}, "potassium": {"trend": "rising"},
        }}}}
        query = chat_llm.build_condition_query(self.SNAPSHOT, temporal, ["NODE_06"]).lower()
        for noise in ("temperature", "humidity", "potassium"):
            self.assertNotIn(noise, query.replace("nitrogen phosphorus potassium", ""))

    def test_condition_query_is_empty_without_evidence(self) -> None:
        self.assertEqual(chat_llm.build_condition_query({"nodes": []}, {"nodes": {}}), "")

    def test_generation_runs_a_second_condition_aware_retrieval(self) -> None:
        question_chunk = SimpleNamespace(chunk_id="a", text="General guide.", source="Guide A", page=1, rerank_score=0.2)
        condition_chunk = SimpleNamespace(chunk_id="b", text="Drain waterlogged maize.", source="Guide B", page=2, rerank_score=3.0)
        queries: list[str] = []

        def retriever(text: str):
            queries.append(text)
            return [condition_chunk, question_chunk]

        with (
            patch.object(chat_llm, "_get_farm_snapshot", return_value=self.SNAPSHOT),
            patch.object(chat_llm, "_get_automatic_temporal_context", return_value=self.TEMPORAL),
            patch.object(chat_llm, "_build_llm_providers", return_value=[chat_llm._LLMProvider("AgentRouter", object(), "test-model")]),
            patch.object(chat_llm, "_call_with_provider_fallback", return_value=("Drain the field.", 0)) as call,
        ):
            response = chat_llm.generate_rag_response("Analyse NODE_06", [question_chunk], retriever=retriever)

        self.assertEqual(len(queries), 1)
        self.assertIn("waterlogging", queries[0])
        prompt = call.call_args.args[1][1]["content"]
        self.assertLess(prompt.index("Drain waterlogged maize."), prompt.index("General guide."))
        self.assertEqual(response.sources, ["Guide A", "Guide B"])
        self.assertEqual(response.chunks_used, 2)

    def test_knowledge_question_does_not_trigger_condition_retrieval(self) -> None:
        calls: list[str] = []
        with (
            patch.object(chat_llm, "_get_farm_snapshot", return_value=self.SNAPSHOT),
            patch.object(chat_llm, "_get_automatic_temporal_context", return_value={"status": "not_requested", "nodes": {}}),
            patch.object(chat_llm, "_build_llm_providers", return_value=[chat_llm._LLMProvider("AgentRouter", object(), "test-model")]),
            patch.object(chat_llm, "_call_with_provider_fallback", return_value=("Compost is decomposed organic matter.", 0)),
        ):
            chat_llm.generate_rag_response("What is compost?", [], retriever=lambda text: calls.append(text) or [])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
