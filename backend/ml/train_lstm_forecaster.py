"""Reproducible training pipeline for the temporal multivariate LSTM.

The default source is real Supabase telemetry. CSV input must explicitly state
its provenance; development/synthetic runs are saved outside the live artifact
directory and are never loaded by Soil Doctor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from backend.ml.lstm_forecaster import ARTIFACT_DIR
from backend.ml.temporal_data import (
    DEFAULT_SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
    FEATURE_OUTPUT_NAMES,
    FARM_DATA_TABLE,
    complete_feature_matrix,
    parse_timestamp,
    prepare_temporal_rows,
)


FORECAST_STEPS = int(os.getenv("TEMPORAL_FORECAST_STEPS", "48"))
CHECKPOINT_STEPS = tuple(
    int(value.strip())
    for value in os.getenv("TEMPORAL_FORECAST_CHECKPOINTS", "6,12,24,48").split(",")
    if value.strip()
)
TRAIN_FRACTION = float(os.getenv("TEMPORAL_TRAIN_FRACTION", "0.70"))
VALIDATION_FRACTION = float(os.getenv("TEMPORAL_VALIDATION_FRACTION", "0.15"))
MIN_TRAINING_WINDOWS = int(os.getenv("TEMPORAL_MIN_TRAINING_WINDOWS", "100"))
RANDOM_SEED = int(os.getenv("TEMPORAL_RANDOM_SEED", "42"))


def fetch_all_telemetry(page_size: int = 1000, max_rows: int = 100000) -> list[dict[str, Any]]:
    from supabase import create_client  # lazy: training-only dependency path

    url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials are not configured.")
    client = create_client(url, key)
    rows: list[dict[str, Any]] = []
    start = 0
    while start < max_rows:
        end = min(start + page_size - 1, max_rows - 1)
        response = (
            client.table(FARM_DATA_TABLE)
            .select("*")
            .order("Timestamp")
            .range(start, end)
            .execute()
        )
        page = getattr(response, "data", None) or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"Timestamp", "Node_ID", *FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "CSV does not match the production telemetry schema. Missing columns: "
            + ", ".join(missing)
        )
    return frame.to_dict(orient="records")


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        node_id = str(row.get("Node_ID") or "").strip()
        if node_id:
            grouped[node_id].append(row)
    return dict(grouped)


def _split_contiguous(prepared: Any) -> list[list[dict[str, Any]]]:
    rows = prepared.analysis_rows
    if not rows:
        return []
    median_minutes = prepared.median_interval_minutes
    if not median_minutes:
        return [rows]
    threshold = median_minutes * 3.0
    segments: list[list[dict[str, Any]]] = [[rows[0]]]
    for row in rows[1:]:
        previous_time = parse_timestamp(segments[-1][-1]["Timestamp"])
        current_time = parse_timestamp(row["Timestamp"])
        gap = (current_time - previous_time).total_seconds() / 60.0
        if gap > threshold:
            segments.append([])
        segments[-1].append(row)
    return [segment for segment in segments if segment]


def prepare_node_series(rows: list[dict[str, Any]]) -> tuple[dict[str, list[list[dict[str, Any]]]], float]:
    grouped = group_rows(rows)
    segments_by_node: dict[str, list[list[dict[str, Any]]]] = {}
    intervals: list[float] = []
    for node_id, node_rows in grouped.items():
        prepared = prepare_temporal_rows(node_rows, node_id=node_id)
        if prepared.median_interval_minutes:
            intervals.append(prepared.median_interval_minutes)
        segments_by_node[node_id] = _split_contiguous(prepared)
    if not intervals:
        raise ValueError("No valid timestamp intervals were found in the training telemetry.")
    cadence = float(median(intervals))
    if any(not 0.8 <= value / cadence <= 1.2 for value in intervals):
        raise ValueError(
            "Node sampling intervals differ by more than 20%. Train separate cadence-compatible models or resample explicitly."
        )
    return segments_by_node, cadence


def _partition_bounds(length: int) -> tuple[int, int]:
    train_end = int(length * TRAIN_FRACTION)
    validation_end = int(length * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return train_end, validation_end


def fit_training_scaler(segments_by_node: dict[str, list[list[dict[str, Any]]]]) -> MinMaxScaler:
    training_rows: list[list[float]] = []
    for segments in segments_by_node.values():
        for segment in segments:
            train_end, _ = _partition_bounds(len(segment))
            matrix = complete_feature_matrix(segment[:train_end])
            if matrix:
                training_rows.extend(matrix)
    if not training_rows:
        raise ValueError("No complete training-only sensor rows are available to fit the scaler.")
    scaler = MinMaxScaler()
    scaler.fit(np.asarray(training_rows, dtype=np.float32))
    return scaler


def _windows_for_partition(
    matrix: np.ndarray,
    *,
    target_start: int,
    target_end: int,
    sequence_length: int,
    forecast_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    first_target = max(target_start, sequence_length)
    last_target = target_end - forecast_steps
    for target_index in range(first_target, last_target + 1):
        input_start = target_index - sequence_length
        x_window = matrix[input_start:target_index]
        y_window = matrix[target_index : target_index + forecast_steps]
        if np.isfinite(x_window).all() and np.isfinite(y_window).all():
            x_values.append(x_window)
            y_values.append(y_window)
    return x_values, y_values


def make_chronological_windows(
    segments_by_node: dict[str, list[list[dict[str, Any]]]],
    scaler: MinMaxScaler,
    *,
    sequence_length: int,
    forecast_steps: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    buckets: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        "train": ([], []),
        "validation": ([], []),
        "test": ([], []),
    }
    for segments in segments_by_node.values():
        for segment in segments:
            matrix = complete_feature_matrix(segment)
            if not matrix:
                continue
            scaled = scaler.transform(np.asarray(matrix, dtype=np.float32))
            train_end, validation_end = _partition_bounds(len(scaled))
            partitions = {
                "train": (0, train_end),
                "validation": (train_end, validation_end),
                "test": (validation_end, len(scaled)),
            }
            for name, (start, end) in partitions.items():
                x_values, y_values = _windows_for_partition(
                    scaled,
                    target_start=start,
                    target_end=end,
                    sequence_length=sequence_length,
                    forecast_steps=forecast_steps,
                )
                buckets[name][0].extend(x_values)
                buckets[name][1].extend(y_values)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (x_values, y_values) in buckets.items():
        result[name] = (
            np.asarray(x_values, dtype=np.float32),
            np.asarray(y_values, dtype=np.float32),
        )
    return result


def build_model(sequence_length: int, forecast_steps: int, feature_count: int) -> Any:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    from keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length, feature_count)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.RepeatVector(forecast_steps),
            layers.LSTM(32, return_sequences=True),
            layers.Dropout(0.2),
            layers.TimeDistributed(layers.Dense(feature_count)),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _inverse(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, len(FEATURE_COLUMNS))).reshape(values.shape)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = None if math.isclose(denominator, 0.0) else 1.0 - float(np.sum(np.square(error))) / denominator
    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(
        2.0 * np.abs(error),
        smape_denominator,
        out=np.zeros_like(error),
        where=smape_denominator > 1e-8,
    )
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6) if r2 is not None else None,
        "smape_pct": round(float(np.mean(smape_terms) * 100.0), 6),
    }


def evaluate_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    checkpoint_steps: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    residual_intervals: dict[str, Any] = {}
    for step in checkpoint_steps:
        index = step - 1
        metrics[str(step)] = {}
        residual_intervals[str(step)] = {}
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            output_name = FEATURE_OUTPUT_NAMES[feature]
            truth = actual[:, index, feature_index]
            estimate = predicted[:, index, feature_index]
            metrics[str(step)][output_name] = _metrics(truth, estimate)
            residual_intervals[str(step)][output_name] = round(
                float(np.quantile(np.abs(estimate - truth), 0.90)),
                6,
            )
    return metrics, residual_intervals


def save_evaluation_plots(actual: np.ndarray, predicted: np.ndarray, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    sample_count = min(200, len(actual))
    final_step = actual.shape[1] - 1
    for feature_index, feature in enumerate(FEATURE_COLUMNS):
        name = FEATURE_OUTPUT_NAMES[feature]
        truth = actual[:sample_count, final_step, feature_index]
        estimate = predicted[:sample_count, final_step, feature_index]
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(truth, label="actual")
        axis.plot(estimate, label="predicted")
        axis.set_title(f"{name}: actual vs predicted at final horizon")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / f"{name}_actual_vs_predicted.png", dpi=150)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 4))
        axis.scatter(estimate, estimate - truth, s=10, alpha=0.6)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(f"{name}: residuals at final horizon")
        axis.set_xlabel("predicted")
        axis.set_ylabel("residual")
        figure.tight_layout()
        figure.savefig(directory / f"{name}_residuals.png", dpi=150)
        plt.close(figure)


def train(
    rows: list[dict[str, Any]],
    *,
    provenance: str,
    output_dir: Path,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    forecast_steps: int = FORECAST_STEPS,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict[str, Any]:
    np.random.seed(RANDOM_SEED)
    segments, cadence = prepare_node_series(rows)
    scaler = fit_training_scaler(segments)
    windows = make_chronological_windows(
        segments,
        scaler,
        sequence_length=sequence_length,
        forecast_steps=forecast_steps,
    )
    for partition in ("train", "validation", "test"):
        if len(windows[partition][0]) == 0:
            raise ValueError(
                f"No {partition} windows could be formed. More contiguous real telemetry is required."
            )
    if len(windows["train"][0]) < MIN_TRAINING_WINDOWS:
        raise ValueError(
            f"Only {len(windows['train'][0])} training windows are available; "
            f"at least {MIN_TRAINING_WINDOWS} are required. No model was saved."
        )

    os.environ.setdefault("KERAS_BACKEND", "torch")
    from keras import callbacks

    model = build_model(sequence_length, forecast_steps, len(FEATURE_COLUMNS))
    model.fit(
        windows["train"][0],
        windows["train"][1],
        validation_data=windows["validation"],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
        ],
        verbose=1,
    )

    validation_pred = _inverse(scaler, np.asarray(model.predict(windows["validation"][0], verbose=0)))
    validation_actual = _inverse(scaler, windows["validation"][1])
    test_pred = _inverse(scaler, np.asarray(model.predict(windows["test"][0], verbose=0)))
    test_actual = _inverse(scaler, windows["test"][1])
    checkpoints = sorted({min(forecast_steps, max(1, step)) for step in CHECKPOINT_STEPS})
    validation_metrics, residual_intervals = evaluate_forecast(
        validation_actual, validation_pred, checkpoints
    )
    test_metrics, _ = evaluate_forecast(test_actual, test_pred, checkpoints)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / "model.keras"))
    joblib.dump(scaler, output_dir / "feature_scaler.pkl")
    save_evaluation_plots(test_actual, test_pred, output_dir / "evaluation_plots")
    metadata = {
        "model_name": "multivariate_temporal_lstm",
        "task": "multi_step_multivariate_regression",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_provenance": provenance,
        "deployment_eligible": provenance == "real_telemetry",
        "feature_order": list(FEATURE_COLUMNS),
        "target_order": list(FEATURE_COLUMNS),
        "sequence_length": sequence_length,
        "forecast_steps": forecast_steps,
        "checkpoint_steps": checkpoints,
        "sampling_interval_minutes": cadence,
        "input_history_duration_minutes": round(sequence_length * cadence, 4),
        "forecast_duration_minutes": round(forecast_steps * cadence, 4),
        "chronological_split": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": round(1.0 - TRAIN_FRACTION - VALIDATION_FRACTION, 4),
            "shuffle": False,
        },
        "scaler_fit_scope": "training_rows_only",
        "window_counts": {name: len(values[0]) for name, values in windows.items()},
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_residual_intervals": residual_intervals,
        "residual_interval_coverage": 0.90,
    }
    temporary_metadata = output_dir / "metadata.json.tmp"
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    temporary_metadata.replace(output_dir / "metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Optional telemetry CSV with the exact production schema.")
    parser.add_argument(
        "--data-provenance",
        choices=("real_telemetry", "development_synthetic"),
        default="real_telemetry",
    )
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--forecast-steps", type=int, default=FORECAST_STEPS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.csv:
        rows = load_csv(args.csv)
    else:
        if args.data_provenance != "real_telemetry":
            raise ValueError("Supabase input must use data_provenance=real_telemetry.")
        rows = fetch_all_telemetry()

    output_dir = ARTIFACT_DIR
    if args.data_provenance != "real_telemetry":
        output_dir = ARTIFACT_DIR.parent / "development_temporal_forecaster"
    metadata = train(
        rows,
        provenance=args.data_provenance,
        output_dir=output_dir,
        sequence_length=args.sequence_length,
        forecast_steps=args.forecast_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
