"""
lstm_anomaly_trainer.py
Trains an LSTM Autoencoder for Anomaly Detection.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import logging
import pathlib
from typing import List, Dict

import joblib
import numpy as np
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers, callbacks

logger = logging.getLogger(__name__)

# --- Configuration ---
_ML_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = _ML_DIR / "lstm_anomaly_model.keras"
SCALER_PATH = _ML_DIR / "scaler_anomaly.pkl"

SEQUENCE_LENGTH = 24  # 24-hour sequences
FEATURES = [
    "Nitrogen_mg_k",
    "Phosphorus_m",
    "Potassium_mg_",
    "Moisture_%",
    "Temperature_C",
    "Humidity_%"
]
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

def fetch_data() -> Dict[str, List[Dict]]:
    url = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials missing.")
        
    client: Client = create_client(url, key)
    # Fetch all data, ordered by timestamp
    logger.info("Fetching data from Supabase for anomaly training...")
    try:
        # In a real scenario we might paginate, but we assume < 10k rows for now
        response = client.table("capstone_dataset").select("*").order("Timestamp").limit(10000).execute()
        data = getattr(response, "data", None) or (response.get("data") if isinstance(response, dict) else [])
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise
        
    logger.info(f"Fetched {len(data)} rows.")
    
    # Group by node
    nodes_data = {}
    for row in data:
        node_id = row.get("Node_ID")
        if not node_id: continue
        if node_id not in nodes_data:
            nodes_data[node_id] = []
        nodes_data[node_id].append(row)
        
    return nodes_data

def prepare_sequences(nodes_data: Dict[str, List[Dict]]) -> tuple[np.ndarray, MinMaxScaler]:
    logger.info("Preparing sequences...")
    # First, collect all valid rows to fit the scaler
    all_rows = []
    node_arrays = {}
    
    for node_id, rows in nodes_data.items():
        node_matrix = []
        for r in rows:
            feat_vec = []
            for f in FEATURES:
                val = r.get(f)
                if val is None: val = 0.0
                feat_vec.append(float(val))
            node_matrix.append(feat_vec)
        
        arr = np.array(node_matrix, dtype=np.float32)
        node_arrays[node_id] = arr
        all_rows.append(arr)
        
    if not all_rows:
        raise ValueError("No data available to train the model.")
        
    full_data = np.vstack(all_rows)
    scaler = MinMaxScaler()
    scaler.fit(full_data)
    
    # Now build sequences
    sequences = []
    for node_id, arr in node_arrays.items():
        if len(arr) < SEQUENCE_LENGTH:
            continue
            
        scaled_arr = scaler.transform(arr)
        
        # Create overlapping sequences
        for i in range(len(scaled_arr) - SEQUENCE_LENGTH + 1):
            seq = scaled_arr[i:i + SEQUENCE_LENGTH]
            sequences.append(seq)
            
    X = np.array(sequences)
    logger.info(f"Created {len(X)} sequences of shape {X.shape}")
    return X, scaler

def build_autoencoder(n_features: int) -> keras.Model:
    model = keras.Sequential([
        # Encoder
        layers.Input(shape=(SEQUENCE_LENGTH, n_features)),
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(16, activation='relu', return_sequences=False),
        layers.RepeatVector(SEQUENCE_LENGTH),
        # Decoder
        layers.LSTM(16, activation='relu', return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mae'])
    return model

def run_training_pipeline():
    logger.info("Starting LSTM Anomaly Model training pipeline...")
    try:
        nodes_data = fetch_data()
        X, scaler = prepare_sequences(nodes_data)
        
        if len(X) == 0:
            logger.error("Not enough data to form sequences.")
            return
            
        n_features = len(FEATURES)
        model = build_autoencoder(n_features)
        
        logger.info("Training LSTM Autoencoder...")
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
        
        history = model.fit(
            X, X,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model on the dataset to print out metrics
        loss, accuracy, mae = model.evaluate(X, X, verbose=0)
        logger.info("--- MODEL TRAINING COMPLETE ---")
        logger.info(f"Final Reconstruction Loss (MSE): {loss:.4f}")
        logger.info(f"Final Mean Absolute Error (MAE): {mae:.4f}")
        logger.info(f"Final Model Accuracy: {accuracy * 100:.2f}%")
        
        # Save artifacts
        model.save(str(MODEL_PATH))
        joblib.dump(scaler, SCALER_PATH)
        
        logger.info(f"Successfully trained and saved model to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
