"""LSTM autoencoder anomaly screening for one node's recent telemetry.

The autoencoder is trained to reconstruct ordinary 24-reading windows. A window
it reconstructs poorly does not look like the telemetry it learned from, which
is reported as an anomaly screening flag together with the sensors that
contributed most. The flag describes an unusual sensor pattern; it is not an
agronomic diagnosis.

Model artefacts (written by ``lstm_anomaly_trainer``):
    backend/ml/lstm_anomaly_model.keras
    backend/ml/scaler_anomaly.pkl
    backend/ml/anomaly_metadata.json   (calibrated thresholds)
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
from typing import Any

import joblib
import numpy as np

from backend.ml import keras_compat
from backend.ml.temporal_data import (
    FEATURE_COLUMNS,
    FEATURE_SHORT_NAMES,
    latest_complete_window,
)

logger = logging.getLogger(__name__)

_ML_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = _ML_DIR / "lstm_anomaly_model.keras"
SCALER_PATH = _ML_DIR / "scaler_anomaly.pkl"
METADATA_PATH = _ML_DIR / "anomaly_metadata.json"

DEFAULT_SEQUENCE_LENGTH = 24

_lock = threading.Lock()
_model: Any | None = None
_scaler: Any | None = None
_metadata: dict[str, Any] | None = None
_signature: tuple[float, ...] | None = None


def _artifact_signature() -> tuple[float, ...] | None:
    paths = (MODEL_PATH, SCALER_PATH, METADATA_PATH)
    if not all(path.exists() for path in paths):
        return None
    return tuple(path.stat().st_mtime for path in paths)


def artifact_status() -> dict[str, Any]:
    """Describe whether a calibrated anomaly model is deployed."""
    missing = [path.name for path in (MODEL_PATH, SCALER_PATH, METADATA_PATH) if not path.exists()]
    if missing:
        status = "not_calibrated" if missing == [METADATA_PATH.name] else "not_trained"
        return {"status": status, "deployed": False, "missing_artifacts": missing}
    try:
        with METADATA_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        return {"status": "invalid_artifacts", "deployed": False, "reason": f"Unable to read metadata: {exc}"}
    return {
        "status": "available",
        "deployed": True,
        "name": metadata.get("model_name"),
        "trained_at": metadata.get("trained_at"),
        "sequence_length": metadata.get("sequence_length", DEFAULT_SEQUENCE_LENGTH),
        "threshold": metadata.get("threshold"),
        "threshold_quantile": metadata.get("threshold_quantile"),
        "window_counts": metadata.get("window_counts", {}),
        "validation_reconstruction_mae": metadata.get("validation_reconstruction_mae"),
    }


def _load_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    global _model, _scaler, _metadata, _signature
    signature = _artifact_signature()
    if signature is None:
        raise FileNotFoundError(
            "The anomaly model is not trained and calibrated. "
            "Run `python -m backend.ml.lstm_anomaly_trainer`."
        )
    with _lock:
        if _model is not None and _signature == signature:
            return _model, _scaler, _metadata
        with METADATA_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if tuple(metadata.get("feature_order") or ()) != FEATURE_COLUMNS:
            raise RuntimeError("Anomaly model feature order does not match the backend contract.")
        scaler = joblib.load(SCALER_PATH)
        model = keras_compat.load_keras_model(MODEL_PATH)
        _model, _scaler, _metadata, _signature = model, scaler, metadata, signature
        return model, scaler, metadata


def window_errors(model: Any, scaled_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (overall error per window, per-sensor error per window).

    Errors are mean absolute reconstruction errors in scaled units, so sensors
    with different physical ranges are comparable.
    """
    reconstructed = np.asarray(keras_compat.predict(model, scaled_windows))
    absolute = np.abs(reconstructed - scaled_windows)
    per_feature = absolute.mean(axis=1)
    return per_feature.mean(axis=1), per_feature


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Screen the newest complete window of one node's chronological rows."""
    try:
        model, scaler, metadata = _load_artifacts()
    except FileNotFoundError as exc:
        return {"status": artifact_status()["status"], "reason": str(exc)}
    except Exception as exc:
        logger.exception("Anomaly artifacts could not be loaded")
        return {"status": "invalid_artifacts", "reason": str(exc)}

    sequence_length = int(metadata.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    if len(rows) < sequence_length:
        return {
            "status": "insufficient_history",
            "samples_available": len(rows),
            "samples_required": sequence_length,
        }
    matrix = latest_complete_window(rows, sequence_length)
    if matrix is None:
        return {
            "status": "incomplete_window",
            "reason": "The newest readings contain missing sensor values.",
            "samples_required": sequence_length,
        }

    scaled = scaler.transform(np.asarray(matrix, dtype=np.float32))
    overall, per_feature = window_errors(model, scaled.reshape(1, sequence_length, len(FEATURE_COLUMNS)))
    error = float(overall[0])
    threshold = float(metadata["threshold"])
    feature_thresholds = metadata.get("feature_thresholds") or {}

    sensors: dict[str, Any] = {}
    for index, feature in enumerate(FEATURE_COLUMNS):
        name = FEATURE_SHORT_NAMES[feature]
        feature_error = float(per_feature[0, index])
        feature_threshold = feature_thresholds.get(name)
        sensors[name] = {
            "reconstruction_error": round(feature_error, 5),
            "threshold": round(float(feature_threshold), 5) if feature_threshold is not None else None,
            "unusual": bool(feature_threshold is not None and feature_error > float(feature_threshold)),
        }

    unusual = [name for name, entry in sensors.items() if entry["unusual"]]
    ranked = sorted(sensors, key=lambda name: sensors[name]["reconstruction_error"], reverse=True)
    ratio = error / threshold if threshold > 0 else None
    is_anomalous = error > threshold
    if is_anomalous:
        severity = "high" if ratio is not None and ratio >= 2.0 else "moderate"
    else:
        severity = "watch" if unusual else "normal"

    return {
        "status": "success",
        "is_anomalous": is_anomalous,
        "severity": severity,
        "reconstruction_error": round(error, 5),
        "threshold": round(threshold, 5),
        "error_to_threshold_ratio": round(ratio, 3) if ratio is not None else None,
        "unusual_sensors": unusual,
        "largest_contributors": ranked[:2],
        "sensors": sensors,
        "window_start": rows[-sequence_length].get("Timestamp"),
        "window_end": rows[-1].get("Timestamp"),
        "readings_used": sequence_length,
        "interpretation": (
            "The recent sensor pattern differs from the telemetry the model learned from. "
            "Treat it as a prompt to check the sensor and the field, not as a diagnosis."
            if is_anomalous
            else "The recent sensor pattern is consistent with the telemetry the model learned from."
        ),
    }
