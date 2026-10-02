# Soil Doctor - Backend

FastAPI service that turns sensor telemetry into farm advice. It combines three
LSTM models with a retrieval-augmented (RAG) chat assistant.

## Quick start

1. Create and activate an environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```

2. Copy `.env.example` to `.env` in the project root and fill in the Supabase
   values and at least one language-model key.

3. Build the knowledge store once (reads `rag_import/agronomic_knowledge.jsonl`,
   writes `backend/data/agronomic_knowledge/`):

   ```bash
   python -m backend.scripts.ingest_agronomic_knowledge
   ```

4. Run the server from the project root:

   ```bash
   uvicorn backend.main:app --port 8000
   ```

5. Check that everything is ready:

   ```bash
   curl http://localhost:8000/health
   ```

`status` is `ready` once the retrieval models are loaded, and `models` shows
which LSTM models are deployed.

## How a chat request flows

```text
question ──► hybrid search (dense + BM25 → RRF → cross-encoder rerank)
        │
        ├──► telemetry gateway (one cached read per node)
        │       ├─ live farm snapshot (latest reading per node)
        │       └─ temporal analysis (trends, events, data quality)
        │
        ├──► LSTM models
        │       ├─ forecaster      → future estimates + reliability per sensor
        │       ├─ anomaly screen  → is the newest window unusual, which sensors
        │       └─ crop model      → recommended crop for the node
        │
        ├──► condition-aware search: the knowledge base is searched again for
        │    what the rules and models found (e.g. "rice low moisture ...")
        │
        └──► language model (AgentRouter → Conduit → local model fallback)
```

The LSTM output reaches the language model as its own `LSTM MODEL EVIDENCE`
prompt block and also decides what agronomic knowledge is retrieved.

## API

| Route | Purpose |
| --- | --- |
| `GET /health` | Readiness of retrieval and each LSTM model |
| `POST /api/v1/chat/` | Ask Soil Doctor (`response_mode`: `chat`, `widget`, `field_report`, `field_summary`) |
| `GET /api/v1/chat/history/{conversation_id}` | Saved conversation |
| `GET /api/v1/ml/status` | Deployed models with their training metrics |
| `GET /api/v1/ml/temporal/{node_id}` | History analysis, forecast, forecast outlook and anomaly screen |
| `POST /api/v1/ml/classify-suitability` | Crop recommendation and threshold verdict |
| `POST /api/v1/ml/train-temporal-forecaster` | Retrain the forecaster (background) |
| `POST /api/v1/ml/train-anomaly-model` | Retrain and recalibrate the anomaly screen (background) |
| `GET /api/v1/ml/training-status` | Whether a training job is running, succeeded or failed |
| `POST /evaluate` | Score readings against a crop's reference ranges |

Interactive documentation is at `http://localhost:8000/docs`.

## LSTM models

| Model | Input | Output | Files |
| --- | --- | --- | --- |
| Forecaster | 48 consecutive readings, 6 sensors | next 48 readings with 90% intervals | `ml/model_artifacts/temporal_forecaster/` |
| Anomaly screen | newest 24 readings | reconstruction error vs calibrated threshold | `ml/lstm_anomaly_model.keras`, `ml/scaler_anomaly.pkl`, `ml/anomaly_metadata.json` |
| Crop recommendation | newest 24 readings | one of 7 crops with a model score | `ml/lstm_crop_model.keras`, `ml/scaler_crop.pkl`, `ml/imputer_crop.pkl` |

Train from the command line (project root):

```bash
python -m backend.ml.train_lstm_forecaster
python -m backend.ml.lstm_anomaly_trainer
```

Both split the data in time order, fit the scaler on training rows only, and
record held-out metrics in a metadata file. The forecaster also records how it
compares with two naive baselines; a sensor is treated as *informative* only when
the LSTM beats the better baseline by a margin. See
[`ml/model_artifacts/README.md`](ml/model_artifacts/README.md).

## Telemetry sources

`backend/ml/node_data.py` is the only module that reads telemetry.

- `NODE_01`–`NODE_03` are physical nodes. They are read from the hardware
  Supabase project when it is configured; otherwise `NODE_01`/`NODE_02` come
  from the gateway's Firebase log and `NODE_03` is reported as unavailable.
- Every other node is read from the simulator Supabase table.

Readings are cached for `TELEMETRY_CACHE_SECONDS` (default 20), so one chat
request makes about three network reads instead of one per node per feature.

## LLM provider fallback

AgentRouter is the primary provider. Each remote provider gets three attempts
for rate limits, timeouts, connection failures and upstream 5xx errors. Conduit
is the second provider; an in-process Transformers model is the final fallback
and does not call an external API:

```dotenv
AGENTROUTER_API_KEY=your-agentrouter-key
AGENTROUTER_MODEL=deepseek-v4-flash

CONDUIT_API_KEY=sk-cdt-your-key
CONDUIT_API_BASE_URL=https://conduit.ozdoev.net/v1
CONDUIT_MODEL=claude-opus-4.8
CONDUIT_FALLBACK_ENABLED=true

LOCAL_LLM_FALLBACK_ENABLED=true
LOCAL_LLM_MODEL=google/flan-t5-small
LOCAL_LLM_DEVICE=cpu
LOCAL_LLM_LOCAL_FILES_ONLY=true
```

A reply that comes back empty, or cut short because the model spent its output
budget on hidden reasoning, is retried once with a larger budget and then passed
to the next provider. A provider that exhausts its retries is skipped for
`LLM_PROVIDER_COOLDOWN_SECONDS` (default 120) so an outage slows one request,
not every request. Authentication and invalid-request errors do not trigger fallback. The chat
response's `model` field names the model that produced the final answer.
Download `LOCAL_LLM_MODEL` once (set `LOCAL_LLM_LOCAL_FILES_ONLY=false` for that
first run) before relying on the offline fallback.

## Screening ranges

Crop reference ranges live in `optimal_thresholds.json` (project root, with an
optional local override in `backend/data/`). Profiles exist for maize, rice and
cassava. The rice and cassava profiles are provisional: they follow the
project's reference crop profiles and should be confirmed against local
agronomic guidance. A node whose crop has no profile is screened with generic
ranges on the sensor's own 0–100 % moisture scale.

## Tests

```bash
python -m pytest backend/tests -q
```

## Notes

- CORS allows `localhost` and `127.0.0.1` on ports 8080 and 5173. Set
  `CORS_ALLOW_ORIGINS` for any other address.
- Restart the backend after changing `.env`.
