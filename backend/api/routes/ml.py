"""Machine-learning routes: LSTM inference, model status and (re)training."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from backend.ml import (
    lstm_anomaly_inference,
    lstm_crop_inference,
    lstm_forecaster,
    node_data,
    soil_health,
    temporal_service,
)

router = APIRouter()
logger = logging.getLogger(__name__)
LEGACY_SUITABILITY_SEQUENCE_LENGTH = 24

# ---------------------------------------------------------------------------
# Training jobs
#
# Training runs in a background task, so the HTTP request that starts it cannot
# report the outcome. This small registry records each job's state so an
# operator can ask whether training finished or why it failed.
# ---------------------------------------------------------------------------

_jobs_lock = threading.Lock()
_training_jobs: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_training_job(name: str, pipeline: Callable[[], Any]) -> None:
    try:
        result = pipeline()
    except Exception as exc:
        logger.exception("Training job '%s' failed", name)
        with _jobs_lock:
            _training_jobs[name].update(status="failed", finished_at=_now(), error=str(exc)[:500])
        return
    # New artifacts and thresholds apply to the very next request.
    node_data.clear_cache()
    summary = None
    if isinstance(result, dict):
        summary = {
            key: result.get(key)
            for key in ("trained_at", "window_counts", "threshold", "sequence_length", "forecast_steps")
            if key in result
        }
    with _jobs_lock:
        _training_jobs[name].update(status="succeeded", finished_at=_now(), result=summary)


def _start_training_job(name: str, pipeline: Callable[[], Any], background_tasks: BackgroundTasks) -> dict[str, Any]:
    with _jobs_lock:
        if _training_jobs.get(name, {}).get("status") == "running":
            raise HTTPException(status_code=409, detail=f"The {name} training job is already running.")
        _training_jobs[name] = {"status": "running", "started_at": _now()}
    background_tasks.add_task(_run_training_job, name, pipeline)
    return {
        "job": name,
        "status": "running",
        "status_url": "/api/v1/ml/training-status",
    }


@router.post("/ml/train-anomaly-model", tags=["Machine Learning"])
def train_anomaly_model(background_tasks: BackgroundTasks):
    """Train and calibrate the LSTM anomaly autoencoder in the background."""
    logger.info("Received request to train anomaly model. Spawning background task...")
    from backend.ml import lstm_anomaly_trainer

    job = _start_training_job("anomaly", lstm_anomaly_trainer.run_training_pipeline, background_tasks)
    return {"message": "Anomaly model training started in the background.", **job}


@router.post("/ml/train-suitability-model", tags=["Machine Learning"])
def train_suitability_model(background_tasks: BackgroundTasks):
    """
    Legacy compatibility endpoint. The suitability classifier is no longer part
    of Soil Doctor's active reasoning path; temporal forecasting is the active
    LSTM objective.

    Labels (Good/Fair/Poor) are derived by thresholding each node's readings
    against its Target_Crop optimal ranges (see backend/ml/soil_health.py).
    """
    logger.info("Received request to train suitability model. Spawning background task...")
    from backend.ml import lstm_suitability_trainer

    job = _start_training_job("suitability", lstm_suitability_trainer.run_training_pipeline, background_tasks)
    return {"message": "Suitability model training started in the background.", **job}


@router.post("/ml/train-temporal-forecaster", tags=["Machine Learning"])
def train_temporal_forecaster(background_tasks: BackgroundTasks):
    """Train the multivariate forecaster from real telemetry in the background."""
    logger.info("Received request to train temporal forecaster.")
    from backend.ml import train_lstm_forecaster

    job = _start_training_job("forecaster", train_lstm_forecaster.run_training_pipeline, background_tasks)
    return {
        "message": "Temporal forecaster training started in the background.",
        "warning": (
            "Artifacts are saved only if chronological train/validation/test windows "
            "contain enough contiguous real telemetry."
        ),
        **job,
    }


@router.get("/ml/training-status", tags=["Machine Learning"])
def training_status():
    """Report the state of every training job started since the server booted."""
    with _jobs_lock:
        return {"jobs": {name: dict(job) for name, job in _training_jobs.items()}}


def model_status_summary() -> dict[str, str]:
    """Compact deployed/not-deployed view used by the health check and logs."""
    return {
        "forecaster": lstm_forecaster.artifact_status()["status"],
        "anomaly_screen": lstm_anomaly_inference.artifact_status()["status"],
        "crop_recommendation": lstm_crop_inference.artifact_status()["status"],
    }


@router.get("/ml/status", tags=["Machine Learning"])
def model_status():
    """Say which LSTM models are deployed and how they performed when trained."""
    return {
        "forecaster": lstm_forecaster.artifact_status(),
        "anomaly_screen": lstm_anomaly_inference.artifact_status(),
        "crop_recommendation": lstm_crop_inference.artifact_status(),
    }


@router.get("/ml/temporal/{node_id}", tags=["Machine Learning"])
def temporal_intelligence(node_id: str):
    """Return historical intelligence, forecast and anomaly screen for a node."""
    return temporal_service.get_temporal_farm_intelligence(node_id.strip().upper())


class ClassifySuitabilityRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64, description="Sensor node identifier, e.g. NODE_01.")


@router.post("/ml/classify-suitability", tags=["Machine Learning"])
def classify_suitability(request: ClassifySuitabilityRequest):
    """
    Classify a node's soil as Good / Fair / Poor for the crop it is dedicated to.

    Returns the LSTM verdict AND the direct threshold verdict side by side so the
    model's output stays auditable against the underlying agronomic thresholds.
    """
    node_id = request.node_id.strip().upper()

    window = node_data.fetch_node_window(node_id, limit=LEGACY_SUITABILITY_SEQUENCE_LENGTH)

    if window["status"] == "unavailable":
        raise HTTPException(status_code=503, detail=window.get("reason", "Sensor data unavailable."))

    if window["status"] == "insufficient_data" or window["count"] == 0:
        raise HTTPException(status_code=404, detail=window.get("message", f"No data for {node_id}."))

    crop = window["crop"]
    latest = window["latest"]

    predicted_crop = None
    crop_confidence = None
    crop_probabilities = None
    crop_prediction = None
    try:
        crop_prediction = lstm_crop_inference.predict_for_node(node_id, window["rows"])
        if crop_prediction:
            predicted_crop = crop_prediction["crop"]
            crop_confidence = crop_prediction["confidence"]
            crop_probabilities = crop_prediction["class_probabilities"]
    except Exception as exc:
        logger.warning("Crop recommendation failed for %s: %s", node_id, exc)

    # Direct, always-available threshold verdict on the latest reading.
    threshold_label, threshold_score, per_param = soil_health.score_reading(latest, crop)

    response: dict = {
        "node_id": node_id,
        "crop": crop,
        "predicted_crop": predicted_crop,
        "crop_confidence": crop_confidence,
        "crop_probabilities": crop_probabilities,
        "crop_model_available": predicted_crop is not None,
        "crop_prediction": crop_prediction,
        "crop_status": "ready" if predicted_crop else (
            "insufficient_data" if window["count"] < LEGACY_SUITABILITY_SEQUENCE_LENGTH else "unavailable"
        ),
        "readings_required": LEGACY_SUITABILITY_SEQUENCE_LENGTH,
        "crop_profile": soil_health.normalize_crop(crop),
        "readings_used": window["count"],
        "threshold_label": threshold_label,
        "threshold_score": threshold_score,
        "parameter_scores": per_param,
        "model_label": None,
        "model_confidence": None,
        "model_available": False,
    }

    # Add the LSTM verdict when a full window and trained artefacts are present.
    if window["count"] >= LEGACY_SUITABILITY_SEQUENCE_LENGTH:
        try:
            from backend.ml import lstm_suitability_inference

            matrix = node_data.build_feature_matrix(window["rows"])
            prediction = lstm_suitability_inference.classify_soil_suitability(matrix)
            response["model_label"] = prediction["label"]
            response["model_confidence"] = prediction["confidence"]
            response["model_class_probabilities"] = prediction["class_probabilities"]
            response["model_available"] = True
        except FileNotFoundError:
            logger.debug("Suitability model not trained; returning threshold verdict only.")
        except Exception as exc:
            logger.warning("Suitability inference failed for %s: %s", node_id, exc)

    return response
