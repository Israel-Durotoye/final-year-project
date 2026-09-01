# Temporal forecasting artifacts

The active Soil Doctor forecaster loads only validated real-telemetry artifacts
from:

```text
backend/ml/model_artifacts/temporal_forecaster/
├── model.keras
├── feature_scaler.pkl
├── metadata.json
└── evaluation_plots/
```

No temporal model is currently deployed. This repository does not yet contain a
sufficient, gap-free real telemetry history for a credible chronological
train/validation/test evaluation. The service therefore returns deterministic
historical analysis with `forecast: null` and an explicit forecast status.

Train from real Supabase telemetry after enough contiguous history is available:

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

Configuration is controlled by environment variables, including
`TEMPORAL_HISTORY_ROWS` (default `100`), `TEMPORAL_SEQUENCE_LENGTH` (default
`48`), `TEMPORAL_FORECAST_STEPS` (default `48`), and
`TEMPORAL_FORECAST_CHECKPOINTS` (default `6,12,24,48` sample steps). Metadata
records the measured cadence and converts those sample steps into real elapsed
time at inference.
