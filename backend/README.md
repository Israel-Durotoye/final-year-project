# Soil Doctor - Backend

This folder contains a minimal FastAPI wrapper around the project's `ThresholdEvaluator`.

Quick start (local development):

1. Create a virtual environment and activate it (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2.Install dependencies:

```bash
pip install -r requirements.txt
```

3.Rebuild the live agronomic knowledge store from the canonical JSONL dataset:

```bash
python -m backend.scripts.ingest_agronomic_knowledge
```

The importer reads `rag_import/agronomic_knowledge.jsonl`, embeds each
`knowledge_text` value with the backend's configured embedding model, uses
`fact_id` as the Chroma document ID, and writes only runtime artifacts to
`backend/data/agronomic_knowledge/`. The prebuilt `rag_import/chroma_db` is not
used.

4.Run the dev server (uvicorn):

```bash
uvicorn backend.main:app --reload --port 8000
```

### LLM provider fallback

AgentRouter remains the primary provider. Configure Conduit to keep Soil Doctor
available when AgentRouter hits a rate/usage limit, or exhausts retries because
of timeouts, connection errors, or upstream 5xx failures:

```dotenv
AGENTROUTER_API_KEY=your-agentrouter-key
AGENTROUTER_MODEL=gpt-5.6-sol

CONDUIT_API_KEY=sk-cdt-your-key
CONDUIT_API_BASE_URL=https://conduit.ozdoev.net/v1
CONDUIT_MODEL=claude-opus-4.8
CONDUIT_FALLBACK_ENABLED=true
```

Authentication and invalid-request errors do not trigger fallback. The existing
chat response contract is unchanged; its `model` field contains the model that
actually produced the final answer.

5.Example request (curl):

```bash
curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"telemetry": {"nitrogen_ppm": 18, "phosphorus_ppm": 42, "potassium_ppm": 180, "soil_moisture": 32, "ambient_humidity": 64, "ambient_temperature": 24.5}}'
```

## Temporal farm intelligence

Soil Doctor automatically fetches the latest 100 chronological readings for a
selected node (or each active node for a farm-wide analysis), audits timestamp
gaps/duplicates/missing values, detects historical trends and events, and adds
that evidence to the RAG prompt separately from the live snapshot and future
forecast.

The active model objective is multivariate forecasting of the six production
measurements in this exact order:

```text
Nitrogen_mg_k, Phosphorus_m, Potassium_mg_, Moisture_%, Temperature_C, Humidity_%
```

Legacy crop-suitability and moisture-only endpoints/files remain for backwards
compatibility, but they are no longer used by Soil Doctor's automatic reasoning
or AgentRouter tool path. See
[`backend/ml/model_artifacts/README.md`](ml/model_artifacts/README.md) for model
status, training, backtesting, artifact locations, and configuration.

Notes:
-The service reads `optimal_thresholds.json` from the project root by default.
-CORS is permissive for local development; tighten this for production.
