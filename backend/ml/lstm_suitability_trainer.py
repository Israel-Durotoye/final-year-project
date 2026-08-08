"""
lstm_suitability_trainer.py
Trains an LSTM classifier that decides whether a node's soil is
Good / Fair / Poor for the crop that node is dedicated to.

Pipeline
--------
    fetch  -> rows from capstone_dataset, grouped by node (keeps Target_Crop)
    label  -> soil_health.score_reading() labels each window's latest reading
    train  -> LSTM over 24-step windows of the 6 measured sensors -> softmax(3)
    save   -> lstm_suitability_model.keras + scaler_suitability.pkl
              + suitability_labels.json (class-index -> label name)

The training TARGET is produced by the exact same thresholding code used at
serving time (backend/ml/soil_health.py), so the model learns the threshold
policy and the two can never drift apart.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import logging
import pathlib
from collections import Counter
from typing import Dict, List

import joblib
import numpy as np
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers, callbacks

from backend.ml import soil_health

logger = logging.getLogger(__name__)

# --- Configuration ---
_ML_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = _ML_DIR / "lstm_suitability_model.keras"
SCALER_PATH = _ML_DIR / "scaler_suitability.pkl"
LABELS_PATH = _ML_DIR / "suitability_labels.json"

SEQUENCE_LENGTH = 24  # 24-step windows, matching the anomaly model
FEATURES = soil_health.FEATURES  # 6 measured sensor columns
NUM_CLASSES = len(soil_health.LABELS)

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

FARM_DATA_TABLE = os.environ.get("FARM_DATA_TABLE", "capstone_dataset")


def fetch_data() -> Dict[str, List[Dict]]:
    """Fetch all rows and group them by node, preserving chronological order."""

    url = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials missing.")

    client: Client = create_client(url, key)
    logger.info("Fetching data from Supabase for suitability training...")
    try:
        response = (
            client.table(FARM_DATA_TABLE)
            .select("*")
            .order("Timestamp")
            .limit(10000)
            .execute()
        )
        data = getattr(response, "data", None) or (
            response.get("data") if isinstance(response, dict) else []
        )
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

    logger.info(f"Fetched {len(data)} rows.")

    nodes_data: Dict[str, List[Dict]] = {}
    for row in data:
        node_id = row.get("Node_ID")
        if not node_id:
            continue
        nodes_data.setdefault(node_id, []).append(row)

    return nodes_data


def prepare_sequences(
    nodes_data: Dict[str, List[Dict]],
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Build scaled 24-step windows (X) and their Good/Fair/Poor labels (y).

    The label for a window is the suitability of its LAST reading against the
    node's crop — i.e. "is the soil good right now, given recent history".
    """

    logger.info("Preparing labelled sequences...")

    # 1) Collect every node's raw feature matrix (unscaled) to fit the scaler.
    node_arrays: Dict[str, np.ndarray] = {}
    all_rows: List[np.ndarray] = []

    for node_id, rows in nodes_data.items():
        node_matrix = []
        for r in rows:
            feat_vec = []
            for f in FEATURES:
                val = r.get(f)
                if val is None:
                    val = 0.0
                feat_vec.append(float(val))
            node_matrix.append(feat_vec)

        arr = np.array(node_matrix, dtype=np.float32)
        node_arrays[node_id] = arr
        if arr.size:
            all_rows.append(arr)

    if not all_rows:
        raise ValueError("No data available to train the model.")

    full_data = np.vstack(all_rows)
    scaler = MinMaxScaler()
    scaler.fit(full_data)

    # 2) Build windows + labels per node.
    sequences: List[np.ndarray] = []
    labels: List[int] = []

    for node_id, rows in nodes_data.items():
        arr = node_arrays[node_id]
        if len(arr) < SEQUENCE_LENGTH:
            continue

        scaled_arr = scaler.transform(arr)

        for i in range(len(scaled_arr) - SEQUENCE_LENGTH + 1):
            window = scaled_arr[i : i + SEQUENCE_LENGTH]

            # Label from the UNSCALED latest reading of this window.
            last_row = rows[i + SEQUENCE_LENGTH - 1]
            crop = last_row.get("Target_Crop")
            label, _score, _per_param = soil_health.score_reading(last_row, crop)

            sequences.append(window)
            labels.append(soil_health.label_index(label))

    if not sequences:
        raise ValueError(
            f"No node has at least {SEQUENCE_LENGTH} readings; cannot form windows."
        )

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    distribution = {soil_health.LABELS[i]: c for i, c in sorted(Counter(labels).items())}
    logger.info(
        "Created %d sequences of shape %s | class distribution: %s",
        len(X),
        X.shape,
        distribution,
    )

    return X, y, scaler


def build_classifier(n_features: int) -> keras.Model:
    """LSTM sequence classifier -> softmax over Good/Fair/Poor."""

    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, n_features)),
        layers.LSTM(32, activation="relu", return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(16, activation="relu", return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Inverse-frequency class weights to counter label imbalance."""

    counts = Counter(int(v) for v in y)
    total = len(y)
    weights: Dict[int, float] = {}
    for cls in range(NUM_CLASSES):
        c = counts.get(cls, 0)
        # Absent class -> weight 1.0; present class -> total / (num_classes * count).
        weights[cls] = (total / (NUM_CLASSES * c)) if c else 1.0
    return weights


def run_training_pipeline():
    """Full train-and-save pipeline. Safe to call from a FastAPI background task."""

    logger.info("Starting LSTM Suitability Model training pipeline...")
    try:
        nodes_data = fetch_data()
        X, y, scaler = prepare_sequences(nodes_data)

        if len(X) == 0:
            logger.error("Not enough data to form sequences.")
            return

        n_features = len(FEATURES)
        model = build_classifier(n_features)
        class_weights = _compute_class_weights(y)
        logger.info("Class weights: %s", class_weights)

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        )

        logger.info("Training LSTM classifier...")
        model.fit(
            X,
            y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        loss, accuracy = model.evaluate(X, y, verbose=0)
        logger.info("--- SUITABILITY MODEL TRAINING COMPLETE ---")
        logger.info(f"Final Loss (sparse CCE): {loss:.4f}")
        logger.info(f"Final Accuracy: {accuracy * 100:.2f}%")

        # Save artifacts.
        model.save(str(MODEL_PATH))
        joblib.dump(scaler, SCALER_PATH)
        with LABELS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "labels": soil_health.LABELS,
                    "index_to_label": soil_health.label_map(),
                    "sequence_length": SEQUENCE_LENGTH,
                    "features": FEATURES,
                },
                fh,
                indent=2,
            )

        logger.info(f"Successfully trained and saved model to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training_pipeline()
