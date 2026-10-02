"""
lstm_anomaly_trainer.py
Trains and calibrates the LSTM autoencoder used for anomaly screening.

Pipeline
--------
    fetch     -> simulator Supabase telemetry plus the physical-node log
    clean     -> per node: sort, deduplicate, drop physically impossible values,
                 split at timestamp gaps (no window crosses a gap)
    split     -> chronological train / validation partitions per segment
    scale     -> MinMaxScaler fitted on training rows only
    train     -> LSTM autoencoder reconstructs 24-reading windows
    calibrate -> the anomaly threshold is a high quantile of the reconstruction
                 error on held-out validation windows
    save      -> lstm_anomaly_model.keras + scaler_anomaly.pkl
                 + anomaly_metadata.json (thresholds used at inference)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from backend.ml import firebase_hardware, lstm_anomaly_inference, supabase_hardware
from backend.ml.temporal_data import (
    FEATURE_COLUMNS,
    FEATURE_SHORT_NAMES,
    GAP_MULTIPLIER,
    PHYSICAL_BOUNDS,
    feature_matrix_with_gaps,
    parse_timestamp,
    prepare_temporal_rows,
)

logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_PATH = lstm_anomaly_inference.MODEL_PATH
SCALER_PATH = lstm_anomaly_inference.SCALER_PATH
METADATA_PATH = lstm_anomaly_inference.METADATA_PATH

SEQUENCE_LENGTH = lstm_anomaly_inference.DEFAULT_SEQUENCE_LENGTH
FEATURES = list(FEATURE_COLUMNS)
EPOCHS = int(os.getenv("ANOMALY_EPOCHS", "100"))
BATCH_SIZE = 32
TRAIN_FRACTION = float(os.getenv("ANOMALY_TRAIN_FRACTION", "0.80"))
THRESHOLD_QUANTILE = float(os.getenv("ANOMALY_THRESHOLD_QUANTILE", "0.99"))
MIN_TRAINING_WINDOWS = int(os.getenv("ANOMALY_MIN_TRAINING_WINDOWS", "100"))
RANDOM_SEED = int(os.getenv("ANOMALY_RANDOM_SEED", "42"))


def fetch_data() -> dict[str, list[dict[str, Any]]]:
    """Fetch every available reading and group it by node."""
    from backend.ml.train_lstm_forecaster import fetch_all_telemetry, group_rows

    rows = fetch_all_telemetry()
    logger.info("Fetched %d simulator-project rows for anomaly training.", len(rows))

    try:
        hardware_rows = (
            supabase_hardware.fetch_all_hardware_rows()
            if supabase_hardware.is_configured()
            else firebase_hardware.fetch_all_hardware_rows()
        )
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("Physical-node telemetry unavailable for anomaly training: %s", exc)
        hardware_rows = []
    logger.info("Fetched %d physical-node rows for anomaly training.", len(hardware_rows))

    grouped = group_rows(rows)
    # Physical readings replace any old simulated rows stored under the same ID.
    grouped.update(group_rows(hardware_rows))
    return grouped


def _contiguous_segments(rows: list[dict[str, Any]], node_id: str) -> list[np.ndarray]:
    """Return gap-free feature matrices; unusable values become NaN."""
    prepared = prepare_temporal_rows(rows, node_id=node_id)
    ordered = prepared.rows
    if not ordered:
        return []
    matrix = np.asarray(feature_matrix_with_gaps(ordered), dtype=np.float32)
    for index, feature in enumerate(FEATURE_COLUMNS):
        lower, upper = PHYSICAL_BOUNDS[feature]
        column = matrix[:, index]
        column[(column < lower) | (column > upper)] = np.nan

    threshold = (prepared.median_interval_minutes or 0.0) * GAP_MULTIPLIER
    boundaries = [0]
    if threshold:
        timestamps = [parse_timestamp(row["Timestamp"]) for row in ordered]
        for index in range(1, len(timestamps)):
            gap = (timestamps[index] - timestamps[index - 1]).total_seconds() / 60.0
            if gap > threshold:
                boundaries.append(index)
    boundaries.append(len(ordered))
    return [matrix[start:end] for start, end in zip(boundaries, boundaries[1:]) if end > start]


def _windows(matrix: np.ndarray, start: int, end: int) -> list[np.ndarray]:
    """Complete windows whose last reading lies in ``[start, end)``."""
    windows = []
    for last in range(max(start, SEQUENCE_LENGTH - 1), end):
        window = matrix[last - SEQUENCE_LENGTH + 1 : last + 1]
        if np.isfinite(window).all():
            windows.append(window)
    return windows


def prepare_sequences(
    nodes_data: dict[str, list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Build scaled chronological train and validation windows."""
    segments = [
        segment
        for node_id, rows in nodes_data.items()
        for segment in _contiguous_segments(rows, node_id)
        if len(segment) >= SEQUENCE_LENGTH
    ]
    if not segments:
        raise ValueError(f"No node has {SEQUENCE_LENGTH} consecutive readings; cannot form windows.")

    training_rows = [segment[: int(len(segment) * TRAIN_FRACTION)] for segment in segments]
    stacked = np.vstack(training_rows)
    stacked = stacked[np.isfinite(stacked).all(axis=1)]
    if not len(stacked):
        raise ValueError("No complete training readings are available to fit the scaler.")
    scaler = MinMaxScaler()
    scaler.fit(stacked)

    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    for segment in segments:
        scaled = scaler.transform(segment)
        split = int(len(segment) * TRAIN_FRACTION)
        train.extend(_windows(scaled, 0, split))
        validation.extend(_windows(scaled, split, len(segment)))

    logger.info("Created %d training and %d validation windows.", len(train), len(validation))
    return (
        np.asarray(train, dtype=np.float32),
        np.asarray(validation, dtype=np.float32),
        scaler,
    )


def build_autoencoder(n_features: int) -> Any:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    from keras import layers

    model = keras.Sequential([
        # Encoder
        layers.Input(shape=(SEQUENCE_LENGTH, n_features)),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(16, return_sequences=False),
        layers.RepeatVector(SEQUENCE_LENGTH),
        # Decoder
        layers.LSTM(16, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_features)),
    ])
    # Reconstruction is a regression task; classification accuracy is meaningless here.
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def run_training_pipeline() -> dict[str, Any]:
    """Train, calibrate and save the anomaly model. Raises on failure."""
    logger.info("Starting LSTM anomaly model training pipeline...")
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    from keras import callbacks

    np.random.seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)

    nodes_data = fetch_data()
    train, validation, scaler = prepare_sequences(nodes_data)
    if len(train) < MIN_TRAINING_WINDOWS:
        raise ValueError(
            f"Only {len(train)} training windows are available; at least "
            f"{MIN_TRAINING_WINDOWS} are required. No model was saved."
        )
    if len(validation) == 0:
        raise ValueError("No validation windows are available to calibrate the anomaly threshold.")

    model = build_autoencoder(len(FEATURES))
    model.fit(
        train,
        train,
        validation_data=(validation, validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            # min_delta stops the run once improvements become negligible.
            callbacks.EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ],
        verbose=2,
    )

    overall, per_feature = lstm_anomaly_inference.window_errors(model, validation)
    train_overall, _ = lstm_anomaly_inference.window_errors(model, train)
    threshold = float(np.quantile(overall, THRESHOLD_QUANTILE))
    metadata = {
        "model_name": "lstm_autoencoder_anomaly_screen",
        "task": "reconstruction_error_anomaly_screening",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "error_metric": "mean_absolute_error_in_minmax_scaled_units",
        "scaler_fit_scope": "training_rows_only",
        "chronological_split": {"train": TRAIN_FRACTION, "validation": round(1 - TRAIN_FRACTION, 4)},
        "window_counts": {"train": int(len(train)), "validation": int(len(validation))},
        "nodes": sorted(nodes_data),
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold": round(threshold, 6),
        "feature_thresholds": {
            FEATURE_SHORT_NAMES[feature]: round(float(np.quantile(per_feature[:, index], THRESHOLD_QUANTILE)), 6)
            for index, feature in enumerate(FEATURE_COLUMNS)
        },
        "train_reconstruction_mae": round(float(train_overall.mean()), 6),
        "validation_reconstruction_mae": round(float(overall.mean()), 6),
        "validation_error_quantiles": {
            str(quantile): round(float(np.quantile(overall, quantile)), 6)
            for quantile in (0.5, 0.9, 0.95, 0.99)
        },
    }

    model.save(str(MODEL_PATH))
    joblib.dump(scaler, SCALER_PATH)
    temporary = METADATA_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    temporary.replace(METADATA_PATH)

    logger.info(
        "Anomaly model saved | validation MAE=%.4f | threshold(q%.2f)=%.4f",
        metadata["validation_reconstruction_mae"],
        THRESHOLD_QUANTILE,
        threshold,
    )
    return metadata


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_training_pipeline(), indent=2))
