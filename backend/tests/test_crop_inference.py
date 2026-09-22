"""Regression coverage for using real per-node history for crop recommendations."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from backend.ml import lstm_crop_inference as inference
from backend.api.routes import ml


def rows(count=24):
    return [{"Node_ID": "NODE_01", "Timestamp": f"2026-09-18T00:{i:02d}:00Z",
             **dict(zip(inference.FEATURES, [40 + i, 50, 200, 45, 25, 60]))}
            for i in range(count)]


class CropInferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.predict.return_value = np.array([[0.2, 0.8]])
        self.imputer = Mock()
        self.imputer.transform.side_effect = lambda frame: frame.fillna(20).to_numpy()
        self.scaler = Mock()
        self.scaler.transform.side_effect = lambda values: values
        for name, value in [("load_artifacts", Mock(return_value=True)), ("_model", self.model),
                            ("_imputer", self.imputer), ("_scaler", self.scaler),
                            ("_labels", {"0": "Rice", "1": "Maize"})]:
            patcher = patch.object(inference, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_latest_24_chronological_readings_reach_model(self):
        result = inference.predict_ideal_crop_from_rows(rows(30))
        sequence = self.model.predict.call_args.args[0]
        self.assertEqual(sequence.shape, (1, 24, 6))
        self.assertEqual(sequence[0, 0, 0], 46)
        self.assertEqual(sequence[0, -1, 0], 69)
        self.assertEqual(result["crop"], "Maize")
        self.assertEqual(result["readings_used"], 24)
        self.assertEqual(result["window_start"], "2026-09-18T00:06:00Z")
        self.assertEqual(result["imputed_values"], 0)

    def test_short_history_never_fabricates_a_sequence(self):
        self.assertIsNone(inference.predict_ideal_crop_from_rows(rows(12)))
        self.model.predict.assert_not_called()

    def test_invalid_and_disconnected_values_use_training_imputer(self):
        window = rows()
        window[0].update(Nitrogen_mg_k=0, Phosphorus_m=0, Potassium_mg_=0)
        window[1]["Humidity_%"] = 999
        result = inference.predict_ideal_crop_from_rows(window)
        frame = self.imputer.transform.call_args.args[0]
        self.assertTrue(frame.iloc[0, :3].isna().all())
        self.assertTrue(np.isnan(frame.iloc[1]["Humidity_%"]))
        self.assertEqual(result["imputed_values"], 4)

    def test_completely_missing_values_do_not_generate_a_crop(self):
        self.assertIsNone(inference.predict_ideal_crop_from_rows([{"Timestamp": str(i)} for i in range(24)]))
        self.model.predict.assert_not_called()

    @patch("backend.ml.node_data.firebase_hardware.fetch_hardware_rows")
    def test_hardware_prediction_uses_firebase_history(self, fetch):
        fetch.return_value = rows()
        self.assertEqual(inference.predict_ideal_crop("NODE_01"), "Maize")
        fetch.assert_called_once_with("NODE_01", limit=24)

    @patch("backend.api.routes.ml.node_data.fetch_node_window")
    def test_api_distinguishes_prediction_from_recorded_planting(self, fetch):
        fetch.return_value = {"status": "ok", "crop": None, "rows": rows(), "latest": rows()[-1], "count": 24}
        result = ml.classify_suitability(ml.ClassifySuitabilityRequest(node_id="NODE_01"))
        self.assertIsNone(result["crop"])
        self.assertEqual(result["predicted_crop"], "Maize")
        self.assertEqual(result["crop_status"], "ready")
        self.assertEqual(result["crop_prediction"]["readings_used"], 24)

    @patch("backend.api.routes.ml.node_data.fetch_node_window")
    def test_api_explains_insufficient_history(self, fetch):
        fetch.return_value = {"status": "ok", "crop": None, "rows": rows(8), "latest": rows(8)[-1], "count": 8}
        result = ml.classify_suitability(ml.ClassifySuitabilityRequest(node_id="NODE_02"))
        self.assertIsNone(result["predicted_crop"])
        self.assertEqual(result["crop_status"], "insufficient_data")
        self.assertEqual(result["readings_used"], 8)


if __name__ == "__main__":
    unittest.main()
