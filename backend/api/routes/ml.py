from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import logging

from backend.ml import lstm_anomaly_trainer, lstm_suitability_trainer, node_data, soil_health

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ml/train-anomaly-model", tags=["Machine Learning"])
async def train_anomaly_model(background_tasks: BackgroundTasks):
    """
    Triggers the LSTM anomaly autoencoder training as a background task.
    """
    logger.info("Received request to train anomaly model. Spawning background task...")

    # We run it in the background so the HTTP request completes immediately
    background_tasks.add_task(lstm_anomaly_trainer.run_training_pipeline)

    return {"message": "Anomaly model training started in the background."}


@router.post("/ml/train-suitability-model", tags=["Machine Learning"])
async def train_suitability_model(background_tasks: BackgroundTasks):
    """
    Triggers the LSTM soil-suitability classifier training as a background task.

    Labels (Good/Fair/Poor) are derived by thresholding each node's readings
    against its Target_Crop optimal ranges (see backend/ml/soil_health.py).
    """
    logger.info("Received request to train suitability model. Spawning background task...")

    background_tasks.add_task(lstm_suitability_trainer.run_training_pipeline)

    return {"message": "Suitability model training started in the background."}


class ClassifySuitabilityRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64, description="Sensor node identifier, e.g. NODE_01.")


@router.post("/ml/classify-suitability", tags=["Machine Learning"])
async def classify_suitability(request: ClassifySuitabilityRequest):
    """
    Classify a node's soil as Good / Fair / Poor for the crop it is dedicated to.

    Returns the LSTM verdict AND the direct threshold verdict side by side so the
    model's output stays auditable against the underlying agronomic thresholds.
    """
    node_id = request.node_id.strip()

    window = node_data.fetch_node_window(node_id, limit=lstm_suitability_trainer.SEQUENCE_LENGTH)

    if window["status"] == "unavailable":
        raise HTTPException(status_code=503, detail=window.get("reason", "Sensor data unavailable."))

    if window["status"] == "insufficient_data" or window["count"] == 0:
        raise HTTPException(status_code=404, detail=window.get("message", f"No data for {node_id}."))

    crop = window["crop"]
    latest = window["latest"]

    # Direct, always-available threshold verdict on the latest reading.
    threshold_label, threshold_score, per_param = soil_health.score_reading(latest, crop)

    response: dict = {
        "node_id": node_id,
        "crop": crop,
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
    if window["count"] >= lstm_suitability_trainer.SEQUENCE_LENGTH:
        try:
            from backend.ml import lstm_suitability_inference

            matrix = node_data.build_feature_matrix(window["rows"])
            prediction = lstm_suitability_inference.classify_soil_suitability(matrix)
            response["model_label"] = prediction["label"]
            response["model_confidence"] = prediction["confidence"]
            response["model_class_probabilities"] = prediction["class_probabilities"]
            response["model_available"] = True
        except FileNotFoundError:
            logger.info("Suitability model not trained yet; returning threshold verdict only.")
        except Exception as exc:
            logger.warning("Suitability inference failed for %s: %s", node_id, exc)
    else:
        logger.info(
            "Only %d readings for %s (<%d); returning threshold verdict only.",
            window["count"],
            node_id,
            lstm_suitability_trainer.SEQUENCE_LENGTH,
        )

    return response
