"""Lazy-loaded multivariate, multi-step LSTM forecasting inference."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.ml import keras_compat
from backend.ml.temporal_data import FEATURE_COLUMNS, FEATURE_OUTPUT_NAMES


logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(
    os.getenv(
        "TEMPORAL_MODEL_ARTIFACT_DIR",
        str(Path(__file__).resolve().parent / "model_artifacts" / "temporal_forecaster"),
    )
)
MODEL_PATH = ARTIFACT_DIR / "model.keras"
SCALER_PATH = ARTIFACT_DIR / "feature_scaler.pkl"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

_lock = threading.Lock()
_model: Any | None = None
_scaler: Any | None = None
_metadata: dict[str, Any] | None = None
_artifact_signature: tuple[float, float, float] | None = None
_status_cache: tuple[tuple[float, float, float] | None, dict[str, Any]] | None = None


def _signature() -> tuple[float, float, float] | None:
    paths = (MODEL_PATH, SCALER_PATH, METADATA_PATH)
    if not all(path.exists() for path in paths):
        return None
    return tuple(path.stat().st_mtime for path in paths)


def artifact_status() -> dict[str, Any]:
    """Describe the deployed forecaster; cached until the artifact files change."""
    global _status_cache
    signature = _signature()
    if _status_cache is not None and _status_cache[0] == signature:
        return dict(_status_cache[1])
    status = _read_artifact_status()
    _status_cache = (signature, status)
    return dict(status)


def _read_artifact_status() -> dict[str, Any]:
    missing = [str(path) for path in (MODEL_PATH, SCALER_PATH, METADATA_PATH) if not path.exists()]
    if missing:
        return {
            "status": "not_trained",
            "deployed": False,
            "artifact_directory": str(ARTIFACT_DIR),
            "missing_artifacts": missing,
        }
    try:
        with METADATA_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        return {
            "status": "invalid_artifacts",
            "deployed": False,
            "reason": f"Unable to read metadata: {exc}",
        }
    return {
        "status": "available",
        "deployed": True,
        "artifact_directory": str(ARTIFACT_DIR),
        "name": metadata.get("model_name"),
        "trained_at": metadata.get("trained_at"),
        "sequence_length": metadata.get("sequence_length"),
        "forecast_steps": metadata.get("forecast_steps"),
        "feature_order": metadata.get("feature_order"),
        "sampling_interval_minutes": metadata.get("sampling_interval_minutes"),
        "target_mode": metadata.get("target_mode", "absolute"),
        "data_provenance": metadata.get("data_provenance"),
        "window_counts": metadata.get("window_counts", {}),
        "validation_metrics": metadata.get("validation_metrics", {}),
        "test_metrics": metadata.get("test_metrics", {}),
        "skill_vs_baseline": metadata.get("skill_vs_baseline", {}),
    }


def _load_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    global _model, _scaler, _metadata, _artifact_signature
    signature = _signature()
    if signature is None:
        raise FileNotFoundError(
            f"Temporal forecast artifacts are not deployed in {ARTIFACT_DIR}. "
            "Run `python -m backend.ml.train_lstm_forecaster` after sufficient real telemetry exists."
        )
    with _lock:
        if (
            _model is not None
            and _scaler is not None
            and _metadata is not None
            and _artifact_signature == signature
        ):
            return _model, _scaler, _metadata

        with METADATA_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        feature_order = tuple(metadata.get("feature_order") or ())
        if feature_order != FEATURE_COLUMNS:
            raise RuntimeError(
                "Temporal artifact feature order does not match the backend contract: "
                f"expected {list(FEATURE_COLUMNS)}, received {list(feature_order)}."
            )
        scaler = joblib.load(SCALER_PATH)
        model = keras_compat.load_keras_model(MODEL_PATH)
        _model, _scaler, _metadata = model, scaler, metadata
        _artifact_signature = signature
        return model, scaler, metadata


# How the network's output relates to the forecast, recorded in metadata:
#   "absolute"         the output is the scaled future value itself
#   "delta_from_last"  the output is the change from the newest input reading,
#                      so an untrained output of zero equals "no change"
TARGET_MODES = ("absolute", "delta_from_last")


def to_targets(inputs: np.ndarray, futures: np.ndarray, target_mode: str) -> np.ndarray:
    """Convert scaled future values into what the network is trained to output."""
    if target_mode == "delta_from_last":
        return futures - inputs[:, -1:, :]
    return futures


def from_outputs(inputs: np.ndarray, outputs: np.ndarray, target_mode: str) -> np.ndarray:
    """Convert network outputs back into scaled future values."""
    if target_mode == "delta_from_last":
        return outputs + inputs[:, -1:, :]
    return outputs


def predict_scaled(model: Any, scaled_inputs: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Forecast scaled future values for a batch of scaled input windows."""
    outputs = np.asarray(keras_compat.predict(model, scaled_inputs))
    return from_outputs(scaled_inputs, outputs, str(metadata.get("target_mode") or "absolute"))


def _format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{max(1, round(minutes))}m"
    hours = minutes / 60.0
    if hours < 48:
        return f"{round(hours, 1):g}h"
    return f"{round(hours / 24.0, 1):g}d"


def _checkpoint_steps(metadata: dict[str, Any]) -> list[int]:
    forecast_steps = int(metadata["forecast_steps"])
    raw = metadata.get("checkpoint_steps") or [6, 12, 24, forecast_steps]
    return sorted({max(1, min(forecast_steps, int(step))) for step in raw})


def _interval_for(
    metadata: dict[str, Any],
    step: int,
    feature: str,
) -> float | None:
    residuals = metadata.get("validation_residual_intervals") or {}
    step_payload = residuals.get(str(step)) or residuals.get(step) or {}
    value = step_payload.get(FEATURE_OUTPUT_NAMES[feature])
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def forecast(
    matrix: list[list[float]],
    *,
    median_interval_minutes: float | None,
) -> dict[str, Any]:
    """Forecast every measured feature; return real residual intervals when present."""
    try:
        model, scaler, metadata = _load_artifacts()
    except FileNotFoundError as exc:
        return {"status": "not_trained", "forecast": None, "reason": str(exc), "model": artifact_status()}
    except Exception as exc:
        logger.exception("Temporal artifacts could not be loaded")
        return {
            "status": "invalid_artifacts",
            "forecast": None,
            "reason": str(exc),
            "model": artifact_status(),
        }

    sequence_length = int(metadata["sequence_length"])
    if len(matrix) < sequence_length:
        return {
            "status": "insufficient_history",
            "forecast": None,
            "samples_available": len(matrix),
            "samples_required": sequence_length,
            "model": artifact_status(),
        }

    trained_interval = float(metadata.get("sampling_interval_minutes") or 0.0)
    if median_interval_minutes and trained_interval:
        cadence_ratio = median_interval_minutes / trained_interval
        if not 0.8 <= cadence_ratio <= 1.2:
            return {
                "status": "cadence_mismatch",
                "forecast": None,
                "observed_interval_minutes": round(median_interval_minutes, 4),
                "trained_interval_minutes": round(trained_interval, 4),
                "reason": "The deployed model cadence is incompatible with the current contiguous history.",
                "model": artifact_status(),
            }

    values = np.asarray(matrix[-sequence_length:], dtype=np.float32)
    if values.shape != (sequence_length, len(FEATURE_COLUMNS)) or not np.isfinite(values).all():
        return {
            "status": "invalid_input",
            "forecast": None,
            "reason": "A complete finite value is required for every measured feature in the model window.",
            "model": artifact_status(),
        }

    scaled = scaler.transform(values)
    raw = predict_scaled(
        model,
        scaled.reshape(1, sequence_length, len(FEATURE_COLUMNS)),
        metadata,
    )
    expected = (1, int(metadata["forecast_steps"]), len(FEATURE_COLUMNS))
    if raw.shape != expected:
        return {
            "status": "invalid_artifacts",
            "forecast": None,
            "reason": f"Model output shape {raw.shape} does not match metadata {expected}.",
            "model": artifact_status(),
        }
    predicted = scaler.inverse_transform(raw.reshape(-1, len(FEATURE_COLUMNS))).reshape(raw.shape)[0]
    interval_minutes = median_interval_minutes or trained_interval

    checkpoints: dict[str, Any] = {}
    for step in _checkpoint_steps(metadata):
        label = _format_duration(interval_minutes * step)
        point: dict[str, Any] = {"horizon_steps": step, "horizon_minutes": round(interval_minutes * step, 3)}
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            predicted_value = float(predicted[step - 1, feature_index])
            uncertainty = _interval_for(metadata, step, feature)
            payload: dict[str, Any] = {"predicted": round(predicted_value, 4)}
            if uncertainty is not None:
                payload["prediction_interval"] = {
                    "lower": round(predicted_value - uncertainty, 4),
                    "upper": round(predicted_value + uncertainty, 4),
                    "coverage": metadata.get("residual_interval_coverage", 0.9),
                    "method": "held_out_validation_absolute_residual_quantile",
                }
            else:
                payload["prediction_interval"] = None
            point[FEATURE_OUTPUT_NAMES[feature]] = payload
        checkpoints[label] = point

    trends: dict[str, str] = {}
    for feature_index, feature in enumerate(FEATURE_COLUMNS):
        start = float(values[-1, feature_index])
        end = float(predicted[-1, feature_index])
        difference = end - start
        tolerance = max(abs(start) * 0.02, 0.5)
        trends[FEATURE_OUTPUT_NAMES[feature]] = (
            "stable" if abs(difference) <= tolerance else "rising" if difference > 0 else "falling"
        )

    return {
        "status": "success",
        "forecast": checkpoints,
        "forecast_trends": trends,
        "forecast_skill": summarize_skill(metadata),
        "uncertainty_note": (
            "Prediction intervals are included only where held-out validation residuals were saved; "
            "uncertainty generally grows with the horizon."
        ),
        "model": artifact_status(),
    }


def summarize_skill(metadata: dict[str, Any]) -> dict[str, Any]:
    """Condense the held-out comparison against naive baselines per sensor.

    A sensor is ``informative`` only when the LSTM beat the better naive
    baseline on unseen test windows; otherwise its forecast is a level
    estimate and should not be read as a predicted change.
    """
    skill = metadata.get("skill_vs_baseline") or {}
    per_sensor = skill.get("per_sensor") or {}
    if not per_sensor:
        return {"status": "not_evaluated"}
    informative = sorted(name for name, entry in per_sensor.items() if entry.get("informative"))
    return {
        "status": "evaluated",
        "evaluated_on": skill.get("evaluated_on", "held_out_test_windows"),
        "informative_sensors": informative,
        "level_estimate_only_sensors": sorted(set(per_sensor) - set(informative)),
        "per_sensor": {
            name: {
                "lstm_mae": entry.get("lstm_mae"),
                "best_baseline": entry.get("best_baseline"),
                "best_baseline_mae": entry.get("best_baseline_mae"),
                "improvement_pct": entry.get("improvement_pct"),
            }
            for name, entry in per_sensor.items()
        },
    }
