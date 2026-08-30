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

5.Example request (curl):

```bash
curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"telemetry": {"nitrogen_ppm": 18, "phosphorus_ppm": 42, "potassium_ppm": 180, "soil_moisture": 32, "ambient_humidity": 64, "ambient_temperature": 24.5}}'
```

Notes:
-The service reads `optimal_thresholds.json` from the project root by default.
-CORS is permissive for local development; tighten this for production.
