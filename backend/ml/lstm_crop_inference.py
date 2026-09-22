"""
lstm_crop_inference.py
Runs the LSTM Crop Recommendation model on live data.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import logging
import pathlib
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import keras
from backend.ml import node_data

logger = logging.getLogger(__name__)

_ML_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = _ML_DIR / "lstm_crop_model.keras"
SCALER_PATH = _ML_DIR / "scaler_crop.pkl"
IMPUTER_PATH = _ML_DIR / "imputer_crop.pkl"
LABELS_PATH = _ML_DIR / "crop_labels.json"

SEQUENCE_LENGTH = 24
FEATURES = ["Nitrogen_mg_k", "Phosphorus_m", "Potassium_mg_", "Moisture_%", "Temperature_C", "Humidity_%"]
FEATURE_BOUNDS = {
    "Nitrogen_mg_k": (0, 200),
    "Phosphorus_m": (0, 200),
    "Potassium_mg_": (0, 250),
    "Moisture_%": (0, 100),
    "Temperature_C": (-10, 60),
    "Humidity_%": (0, 100),
}

# Global caches for the loaded artifacts
_model = None
_imputer = None
_scaler = None
_labels = None

def load_artifacts():
    global _model, _imputer, _scaler, _labels
    
    if not all(path.exists() for path in (MODEL_PATH, IMPUTER_PATH, SCALER_PATH, LABELS_PATH)):
        return False
        
    if any(artifact is None for artifact in (_model, _imputer, _scaler, _labels)):
        try:
            logger.info("Loading LSTM crop recommendation model...")
            model = keras.models.load_model(MODEL_PATH, compile=False)
            imputer = joblib.load(IMPUTER_PATH)
            scaler = joblib.load(SCALER_PATH)
            with open(LABELS_PATH, "r") as f:
                labels = json.load(f)
            _model, _imputer, _scaler, _labels = model, imputer, scaler, labels
        except Exception as e:
            logger.error(f"Failed to load LSTM artifacts: {e}")
            return False
            
    return True

def predict_ideal_crop_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Predict the best crop from one node's chronological sensor window."""
    if not load_artifacts() or len(rows) < SEQUENCE_LENGTH:
        return None

    try:
        df = pd.DataFrame(rows[-SEQUENCE_LENGTH:])
        feature_frame = df.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
        for feature, (lower, upper) in FEATURE_BOUNDS.items():
            feature_frame.loc[~feature_frame[feature].between(lower, upper), feature] = np.nan
        npk_features = ["Nitrogen_mg_k", "Phosphorus_m", "Potassium_mg_"]
        feature_frame.loc[feature_frame[npk_features].eq(0).all(axis=1), npk_features] = np.nan
        if not feature_frame.notna().any().any():
            return None
        imputed_data = _imputer.transform(feature_frame)
        scaled_data = _scaler.transform(imputed_data)
        preds = _model.predict(np.expand_dims(scaled_data, axis=0), verbose=0)[0]
        if not np.isfinite(preds).all() or len(preds) != len(_labels):
            return None
        class_idx = int(np.argmax(preds))
        return {
            "crop": _labels.get(str(class_idx)),
            "confidence": float(preds[class_idx]),
            "readings_used": len(feature_frame),
            "window_start": rows[-SEQUENCE_LENGTH].get("Timestamp"),
            "window_end": rows[-1].get("Timestamp"),
            "imputed_values": int(feature_frame.isna().sum().sum()),
            "total_values": int(feature_frame.size),
            "class_probabilities": {
                _labels.get(str(index), str(index)): float(probability)
                for index, probability in enumerate(preds)
            },
        }
    except Exception as exc:
        logger.error("Prediction from sensor window failed: %s", exc)
        return None


def predict_ideal_crop(node_id: str) -> Optional[str]:
    """
    Fetches the last 24 readings for `node_id`, runs the LSTM crop model,
    and returns the predicted ideal crop string (e.g., "Maize").
    Returns None if the model is not trained or there's not enough data.
    """
    window = node_data.fetch_node_window(node_id, limit=SEQUENCE_LENGTH)
    if window["status"] != "ok":
        return None
    prediction = predict_ideal_crop_from_rows(window["rows"])
    return prediction.get("crop") if prediction else None
