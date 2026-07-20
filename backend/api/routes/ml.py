from fastapi import APIRouter, BackgroundTasks, HTTPException
import logging

from backend.ml import lstm_anomaly_trainer

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
