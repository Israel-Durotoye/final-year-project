"""
lstm_inference.py — LSTM Moisture Prediction Inference Wrapper

Responsibility:
    Loads a pre-trained Keras 3 LSTM model and its paired MinMaxScaler,
    then exposes a single inference function that accepts the last 48 hours
    of sensor readings and returns a predicted soil moisture percentage
    for 24 hours into the future.

    The Keras backend MUST be forced to "torch" before any Keras symbol is
    imported. This is done at module level so it takes effect even when this
    module is imported transitively.

Model artefacts (expected paths relative to project root):
    backend/ml/lstm_moisture_model.keras
    backend/ml/scaler_moisture.pkl
"""

from __future__ import annotations

# ── Backend override — must happen before any keras import ──────────────────
import os
os.environ["KERAS_BACKEND"] = "torch"
# ────────────────────────────────────────────────────────────────────────────

import logging
import pathlib
from typing import Sequence

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve artefact paths relative to this file's location
# ---------------------------------------------------------------------------

_ML_DIR = pathlib.Path(__file__).parent.resolve()
_SCALER_PATH = _ML_DIR / "scaler_moisture.pkl"
_MODEL_PATH  = _ML_DIR / "lstm_moisture_model.keras"

# Expected sequence length (hours of sensor history required)
SEQUENCE_LENGTH: int = 48

# ---------------------------------------------------------------------------
# Lazy-loaded globals — initialised on first call to avoid import-time errors
# when model artefacts are not yet present (e.g. during unit testing).
# ---------------------------------------------------------------------------

_scaler = None   # sklearn-compatible scaler loaded from scaler_moisture.pkl
_model  = None   # keras.Model loaded from lstm_moisture_model.keras


def _load_artefacts() -> None:
    """
    Load the scaler and LSTM model into module-level globals.

    Raises:
        FileNotFoundError: If either artefact file is missing.
        RuntimeError: If loading fails for any other reason.
    """
    global _scaler, _model

    if _scaler is not None and _model is not None:
        return  # already loaded

    # --- Scaler -----------------------------------------------------------
    if not _SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Moisture scaler not found at: {_SCALER_PATH}\n"
            "Ensure backend/ml/scaler_moisture.pkl has been placed in the project."
        )
    try:
        _scaler = joblib.load(_SCALER_PATH)
        logger.info("Loaded moisture scaler from %s", _SCALER_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load scaler from {_SCALER_PATH}: {exc}") from exc

    # --- Keras model (imported here to respect the backend env-var set above)
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM model not found at: {_MODEL_PATH}\n"
            "Ensure backend/ml/lstm_moisture_model.keras has been placed in the project."
        )
    try:
        import keras  # noqa: PLC0415 — intentional late import
        _model = keras.saving.load_model(str(_MODEL_PATH))
        logger.info("Loaded LSTM moisture model from %s", _MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load LSTM model from {_MODEL_PATH}: {exc}") from exc


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def execute_moisture_prediction(recent_sensor_data: Sequence[Sequence[float]]) -> float:
    """
    Predict soil moisture percentage 24 hours into the future.

    Args:
        recent_sensor_data:
            A 2-D array-like of shape ``[48, n_features]`` containing the last
            48 hourly sensor readings in chronological order (oldest first).
            Feature columns must match the order used during model training.

    Returns:
        Predicted soil moisture as a raw float (percentage, e.g. 42.7).

    Raises:
        ValueError: If the input does not have exactly SEQUENCE_LENGTH rows.
        FileNotFoundError: If model artefacts are missing.
        RuntimeError: If the Keras inference call fails.
    """
    # Ensure artefacts are loaded
    _load_artefacts()

    # ------------------------------------------------------------------
    # 1. Validate input shape
    # ------------------------------------------------------------------
    data_array = np.array(recent_sensor_data, dtype=np.float32)

    if data_array.ndim != 2:
        raise ValueError(
            f"recent_sensor_data must be 2-D [rows, features], "
            f"got shape {data_array.shape}."
        )

    n_rows, n_features = data_array.shape

    if n_rows != SEQUENCE_LENGTH:
        raise ValueError(
            f"execute_moisture_prediction requires exactly {SEQUENCE_LENGTH} rows "
            f"(one per hour over the last 48 hours), but received {n_rows}."
        )

    logger.debug(
        "Running moisture prediction on input shape (%d, %d).",
        n_rows, n_features,
    )

    # ------------------------------------------------------------------
    # 2. Scale — fit was done on 2-D [samples, features], so pass as-is
    # ------------------------------------------------------------------
    try:
        scaled = _scaler.transform(data_array)          # shape: [48, features]
    except Exception as exc:
        raise RuntimeError(f"Scaler transform failed: {exc}") from exc

    # ------------------------------------------------------------------
    # 3. Reshape to 3-D tensor [batch=1, timesteps=48, features]
    # ------------------------------------------------------------------
    tensor = scaled.reshape(1, SEQUENCE_LENGTH, n_features)  # [1, 48, features]

    # ------------------------------------------------------------------
    # 4. Run LSTM inference
    # ------------------------------------------------------------------
    try:
        raw_prediction = _model.predict(tensor, verbose=0)  # shape: [1, 1] or [1,]
    except Exception as exc:
        raise RuntimeError(f"LSTM model inference failed: {exc}") from exc

    # Flatten to scalar regardless of output shape
    predicted_value = float(np.squeeze(raw_prediction))

    logger.info("Moisture prediction complete: %.4f%%", predicted_value)
    return predicted_value
