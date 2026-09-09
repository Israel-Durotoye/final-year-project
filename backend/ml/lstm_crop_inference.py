"""
lstm_crop_inference.py
Runs the LSTM Crop Recommendation model on live data.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import logging
import pathlib
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from supabase import create_client, Client
import keras

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
        
    if _model is None:
        try:
            logger.info("Loading LSTM crop recommendation model...")
            _model = keras.saving.load_model(MODEL_PATH)
            _imputer = joblib.load(IMPUTER_PATH)
            _scaler = joblib.load(SCALER_PATH)
            with open(LABELS_PATH, "r") as f:
                _labels = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load LSTM artifacts: {e}")
            return False
            
    return True

def predict_ideal_crop(node_id: str) -> Optional[str]:
    """
    Fetches the last 24 readings for `node_id`, runs the LSTM crop model,
    and returns the predicted ideal crop string (e.g., "Maize").
    Returns None if the model is not trained or there's not enough data.
    """
    if not load_artifacts():
        logger.warning("LSTM artifacts not found. Please run the Jupyter Notebook first.")
        return None
        
    url = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")
    if not url or not key:
        logger.error("Supabase credentials missing.")
        return None

    client: Client = create_client(url, key)

    try:
        response = client.table("capstone_dataset")\
            .select("*")\
            .eq("Node_ID", node_id)\
            .order("Timestamp", desc=True)\
            .limit(SEQUENCE_LENGTH)\
            .execute()
        
        data = getattr(response, "data", None) or (response.get("data") if isinstance(response, dict) else [])
    except Exception as e:
        logger.error(f"Failed to fetch data for node {node_id}: {e}")
        return None

    if len(data) < SEQUENCE_LENGTH:
        logger.warning(f"Not enough data for {node_id}. Need {SEQUENCE_LENGTH}, got {len(data)}")
        return None

    # Sort chronologically (oldest to newest)
    data.reverse()

    df = pd.DataFrame(data)
    
    try:
        # Extract features and scale
        feature_frame = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        for feature, (lower, upper) in FEATURE_BOUNDS.items():
            feature_frame.loc[~feature_frame[feature].between(lower, upper), feature] = np.nan
        npk_features = ["Nitrogen_mg_k", "Phosphorus_m", "Potassium_mg_"]
        feature_frame.loc[feature_frame[npk_features].eq(0).all(axis=1), npk_features] = np.nan
        imputed_data = _imputer.transform(feature_frame)
        scaled_data = _scaler.transform(imputed_data)
        
        # Reshape to (1, SEQUENCE_LENGTH, num_features)
        input_seq = np.expand_dims(scaled_data, axis=0)
        
        # Predict
        preds = _model.predict(input_seq, verbose=0)
        class_idx = np.argmax(preds, axis=1)[0]
        
        predicted_crop = _labels.get(str(class_idx))
        return predicted_crop
    except Exception as e:
        logger.error(f"Prediction failed for {node_id}: {e}")
        return None
