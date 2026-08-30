"""Walk-forward backtesting for the deployed temporal LSTM.

At every cutoff T the input contains only readings at or before T. Targets are
strictly later readings and are used only after inference for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml import lstm_forecaster
from backend.ml.temporal_data import FEATURE_COLUMNS, complete_feature_matrix
from backend.ml.train_lstm_forecaster import (
    _inverse,
    evaluate_forecast,
    fetch_all_telemetry,
    load_csv,
    prepare_node_series,
)


def backtest(rows: list[dict[str, Any]], stride: int = 1) -> dict[str, Any]:
    model, scaler, metadata = lstm_forecaster._load_artifacts()  # noqa: SLF001
    sequence_length = int(metadata["sequence_length"])
    forecast_steps = int(metadata["forecast_steps"])
    segments_by_node, cadence = prepare_node_series(rows)
    actual_windows: list[np.ndarray] = []
    predicted_windows: list[np.ndarray] = []
    cutoffs: list[dict[str, Any]] = []

    for node_id, segments in segments_by_node.items():
        for segment in segments:
            matrix = complete_feature_matrix(segment)
            if not matrix:
                continue
            values = np.asarray(matrix, dtype=np.float32)
            for target_start in range(sequence_length, len(values) - forecast_steps + 1, max(1, stride)):
                # No row after target_start - 1 is present in the model input.
                inputs = values[target_start - sequence_length : target_start]
                actual = values[target_start : target_start + forecast_steps]
                scaled_input = scaler.transform(inputs).reshape(1, sequence_length, len(FEATURE_COLUMNS))
                scaled_prediction = np.asarray(model.predict(scaled_input, verbose=0))
                prediction = _inverse(scaler, scaled_prediction)[0]
                actual_windows.append(actual)
                predicted_windows.append(prediction)
                cutoffs.append(
                    {
                        "node_id": node_id,
                        "cutoff": segment[target_start - 1]["Timestamp"],
                        "first_target": segment[target_start]["Timestamp"],
                        "last_target": segment[target_start + forecast_steps - 1]["Timestamp"],
                    }
                )

    if not actual_windows:
        raise ValueError("No eligible walk-forward backtest windows are available.")
    actual_array = np.asarray(actual_windows, dtype=np.float32)
    predicted_array = np.asarray(predicted_windows, dtype=np.float32)
    checkpoint_steps = sorted(
        {min(forecast_steps, max(1, int(step))) for step in metadata.get("checkpoint_steps", [forecast_steps])}
    )
    metrics, _intervals = evaluate_forecast(actual_array, predicted_array, checkpoint_steps)
    return {
        "status": "success",
        "method": "walk_forward_no_future_inputs",
        "windows": len(actual_windows),
        "sampling_interval_minutes": cadence,
        "sequence_length": sequence_length,
        "forecast_steps": forecast_steps,
        "metrics": metrics,
        "cutoffs": cutoffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_csv(args.csv) if args.csv else fetch_all_telemetry()
    result = backtest(rows, stride=args.stride)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
