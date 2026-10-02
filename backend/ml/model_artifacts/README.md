# Temporal forecasting artifacts

The Soil Doctor forecaster loads its artifacts from:

```text
backend/ml/model_artifacts/temporal_forecaster/
├── model.keras
├── feature_scaler.pkl
├── metadata.json
└── evaluation_plots/
```

`metadata.json` records how the model was trained and how well it did on data
it never saw:

- `window_counts`: training, validation and test windows used.
- `validation_metrics` / `test_metrics`: MAE, RMSE, R² and sMAPE per sensor at
  each reported horizon.
- `validation_residual_intervals`: the 90th-percentile absolute error on
  validation windows, used at inference as a prediction interval.
- `skill_vs_baseline`: test-set error of the LSTM against two naive forecasts
  (repeat the last reading; repeat the input-window mean). A sensor is marked
  `informative` only when the LSTM is at least `TEMPORAL_SKILL_MARGIN_PCT`
  (default 5%) better than the better baseline. Soil Doctor tells the user when
  a forecast is only a level estimate.

Train from the telemetry table once enough contiguous history exists:

```bash
python -m backend.ml.train_lstm_forecaster
```

Backtest the deployed model without future leakage:

```bash
python -m backend.ml.backtest_lstm --stride 6 --output backend/ml/model_artifacts/temporal_forecaster/backtest.json
```

CSV development runs require the exact production schema and explicit
provenance. Synthetic/development artifacts are isolated from the live loader:

```bash
python -m backend.ml.train_lstm_forecaster \
  --csv path/to/development_data.csv \
  --data-provenance development_synthetic
```

A reading with a missing sensor value removes only the training windows that
contain it; the rest of that stretch of history is still used.

Configuration is controlled by environment variables, including
`TEMPORAL_HISTORY_ROWS` (default `100`), `TEMPORAL_SEQUENCE_LENGTH` (default
`48`), `TEMPORAL_FORECAST_STEPS` (default `48`), and
`TEMPORAL_FORECAST_CHECKPOINTS` (default `6,12,24,48` sample steps). Metadata
records the measured cadence and converts those sample steps into real elapsed
time at inference. A node whose reporting interval differs from the training
interval by more than 20% receives no forecast (`cadence_mismatch`).

Retrain after the simulator or the hardware has collected new history; the more
continuous readings there are, the more the model can learn about how the
sensors change over time.
