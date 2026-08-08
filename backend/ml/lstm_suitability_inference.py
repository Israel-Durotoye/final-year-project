"""
lstm_suitability_inference.py — Soil Suitability Classifier Inference Wrapper

Responsibility:
    Loads the trained Keras 3 LSTM suitability classifier, its paired
    MinMaxScaler, and the persisted label map, then exposes a single inference
    function that accepts the last 24 sensor readings for a node and returns a
    Good / Fair / Poor verdict with class probabilities.

    The Keras backend MUST be forced to "torch" before any Keras symbol is
    imported. This is done at module level so it takes effect even when this
    module is imported transitively.

Model artefacts (expected paths relative to this file):
    backend/ml/lstm_suitability_model.keras
    backend/ml/scaler_suitability.pkl
    backend/ml/suitability_labels.json
"""

from __future__ import annotations

# ── Backend override — must happen before any keras import ──────────────────
import os
os.environ["KERAS_BACKEND"] = "torch"
# ────────────────────────────────────────────────────────────────────────────

import json
import logging
import pathlib
from typing import Any, Sequence

import joblib
import numpy as np

from backend.ml import soil_health

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve artefact paths relative to this file's location
# ---------------------------------------------------------------------------

_ML_DIR = pathlib.Path(__file__).parent.resolve()
_SCALER_PATH = _ML_DIR / "scaler_suitability.pkl"
_MODEL_PATH = _ML_DIR / "lstm_suitability_model.keras"
_LABELS_PATH = _ML_DIR / "suitability_labels.json"

# Expected sequence length (kept in sync with the trainer).
SEQUENCE_LENGTH: int = 24

# ---------------------------------------------------------------------------
# Lazy-loaded globals — initialised on first call.
# ---------------------------------------------------------------------------

_scaler = None
_model = None
_index_to_label: dict[int, str] | None = None


def _load_artefacts() -> None:
    """
    Load the scaler, LSTM classifier, and label map into module globals.

    Raises:
        FileNotFoundError: If a required artefact file is missing.
        RuntimeError: If loading fails for any other reason.
    """
    global _scaler, _model, _index_to_label

    if _scaler is not None and _model is not None and _index_to_label is not None:
        return

    # --- Scaler -----------------------------------------------------------
    if not _SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Suitability scaler not found at: {_SCALER_PATH}\n"
            "Train the model first via POST /api/v1/ml/train-suitability-model."
        )
    try:
        _scaler = joblib.load(_SCALER_PATH)
        logger.info("Loaded suitability scaler from %s", _SCALER_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load scaler from {_SCALER_PATH}: {exc}") from exc

    # --- Label map --------------------------------------------------------
    if _LABELS_PATH.exists():
        try:
            with _LABELS_PATH.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            raw_map = payload.get("index_to_label") or {}
            _index_to_label = {int(k): v for k, v in raw_map.items()}
        except Exception as exc:
            logger.warning("Failed to read %s (%s); using default labels.", _LABELS_PATH, exc)
            _index_to_label = soil_health.label_map()
    else:
        _index_to_label = soil_health.label_map()

    # --- Keras model (imported here to respect the backend env-var above) --
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Suitability model not found at: {_MODEL_PATH}\n"
            "Train the model first via POST /api/v1/ml/train-suitability-model."
        )
    try:
        import keras  # noqa: PLC0415 — intentional late import
        _model = keras.saving.load_model(str(_MODEL_PATH))
        logger.info("Loaded LSTM suitability model from %s", _MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load LSTM model from {_MODEL_PATH}: {exc}") from exc


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def classify_soil_suitability(
    recent_sensor_data: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """
    Classify a node's soil as Good / Fair / Poor for its crop.

    Args:
        recent_sensor_data:
            A 2-D array-like of shape ``[24, 6]`` containing the last 24 readings
            in chronological order (oldest first). Feature columns must match
            ``soil_health.FEATURES`` order.

    Returns:
        {
            "label": "Good" | "Fair" | "Poor",
            "confidence": float,             # probability of the winning class
            "class_probabilities": {label: prob, ...},
            "sequence_length": 24,
        }

    Raises:
        ValueError: If the input does not have exactly SEQUENCE_LENGTH rows.
        FileNotFoundError: If model artefacts are missing.
        RuntimeError: If the Keras inference call fails.
    """
    _load_artefacts()

    data_array = np.array(recent_sensor_data, dtype=np.float32)

    if data_array.ndim != 2:
        raise ValueError(
            f"recent_sensor_data must be 2-D [rows, features], got shape {data_array.shape}."
        )

    n_rows, n_features = data_array.shape

    if n_rows != SEQUENCE_LENGTH:
        raise ValueError(
            f"classify_soil_suitability requires exactly {SEQUENCE_LENGTH} rows, "
            f"but received {n_rows}."
        )

    try:
        scaled = _scaler.transform(data_array)
    except Exception as exc:
        raise RuntimeError(f"Scaler transform failed: {exc}") from exc

    tensor = scaled.reshape(1, SEQUENCE_LENGTH, n_features)

    try:
        raw_prediction = _model.predict(tensor, verbose=0)
    except Exception as exc:
        raise RuntimeError(f"LSTM model inference failed: {exc}") from exc

    probs = np.asarray(raw_prediction, dtype=np.float64).reshape(-1)
    winning_index = int(np.argmax(probs))
    label = _index_to_label.get(winning_index, soil_health.LABELS[winning_index])

    class_probabilities = {
        _index_to_label.get(i, soil_health.LABELS[i]): round(float(p), 4)
        for i, p in enumerate(probs)
    }

    logger.info(
        "Suitability classification complete: %s (%.2f%%)",
        label,
        float(probs[winning_index]) * 100.0,
    )

    return {
        "label": label,
        "confidence": round(float(probs[winning_index]), 4),
        "class_probabilities": class_probabilities,
        "sequence_length": SEQUENCE_LENGTH,
    }
