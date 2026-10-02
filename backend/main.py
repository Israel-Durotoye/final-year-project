"""
main.py — Soil Doctor: FastAPI Application Entry Point

This file is the single wiring point for the entire backend.
Its only jobs are:
  1. Manage the RAGEngine lifecycle (startup / shutdown)
  2. Register all API routers with their URL prefixes
  3. Configure CORS for the React frontend
  4. Expose the health check and the threshold evaluator

To run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os

# The ML routes use TensorFlow/Keras, while SentenceTransformers only needs
# Transformers' PyTorch backend. Prevent Transformers from importing tf_keras
# after TensorFlow has already been initialised by the ML routes; that double
# import registers TensorFlow monitoring metrics twice and aborts startup.
os.environ.setdefault("USE_TF", "0")

from dotenv import load_dotenv
load_dotenv()

import dataclasses
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.api.routes import chat as chat_router
from backend.api.routes import ml as ml_router
from backend.prescriptive.evaluator import ThresholdEvaluator
from backend.rag.rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# One line per outgoing HTTP request drowns the application's own messages.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton RAGEngine — one instance shared across all requests
# ---------------------------------------------------------------------------
_rag_engine = RAGEngine()
_threshold_evaluator: ThresholdEvaluator | None = None


def _warm_up_lstm_models() -> None:
    """Load the LSTM models once so the first user request is not slow."""
    try:
        from backend.ml import lstm_anomaly_inference, lstm_crop_inference, lstm_forecaster

        lstm_crop_inference.load_artifacts()
        for loader in (lstm_forecaster._load_artifacts, lstm_anomaly_inference._load_artifacts):  # noqa: SLF001
            try:
                loader()
            except FileNotFoundError:
                pass  # Not trained yet; the service reports this to callers.
        logger.info("LSTM models warmed up: %s", ml_router.model_status_summary())
    except Exception:
        logger.warning("LSTM warm-up did not complete; models will load on first use.", exc_info=True)


# ---------------------------------------------------------------------------
# Application Lifespan — startup and shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup; everything after runs at shutdown.

    The retrieval models (SentenceTransformer + CrossEncoder) load here, once,
    before the server accepts requests. The LSTM models load in a background
    thread so they do not delay startup.
    """
    logger.info("Soil Doctor backend starting up...")
    _rag_engine.initialize()
    chat_router.set_engine(_rag_engine)
    threading.Thread(target=_warm_up_lstm_models, name="lstm-warm-up", daemon=True).start()
    logger.info("Startup complete. Backend is ready.")
    yield
    logger.info("Soil Doctor backend shutting down...")
    _rag_engine.shutdown()


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Soil Doctor API",
    description=(
        "Prescriptive agronomic Decision Support System. "
        "Combines a RAG chatbot with LSTM forecasting, anomaly screening and "
        "crop recommendation over live sensor telemetry."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS — allow the React frontend ────────────────────────────────────────
# Vite serves the app on port 8080; the browser may reach it as localhost or
# 127.0.0.1. Set CORS_ALLOW_ORIGINS (comma-separated) for any other address.
_DEFAULT_ORIGINS = (
    "http://localhost:8080,http://127.0.0.1:8080,"
    "http://localhost:5173,http://127.0.0.1:5173"
)
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(chat_router.router, prefix="/api/v1")
app.include_router(ml_router.router, prefix="/api/v1")


# ── Root health check ────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root() -> dict:
    return {
        "service": "Soil Doctor API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health() -> dict:
    """One call that says whether every part of the backend is ready."""
    return {
        "status": "ready" if _rag_engine.is_ready() else "starting",
        "knowledge_base_chunks": _rag_engine.document_count() if _rag_engine.is_ready() else 0,
        "models": ml_router.model_status_summary(),
    }


# ── Threshold evaluator ──────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    telemetry: dict[str, Any] = Field(..., description="Readings keyed by threshold parameter name.")
    crop: str = Field(default="maize_corn", description="Crop profile key or common crop name.")


@app.post("/evaluate", tags=["Prescriptive"])
def evaluate(request: EvaluateRequest) -> dict:
    """Score a telemetry snapshot against a crop's reference ranges."""
    global _threshold_evaluator
    if _threshold_evaluator is None:
        _threshold_evaluator = ThresholdEvaluator()
    from backend.ml import soil_health

    crop_key = soil_health.resolve_crop_profile(request.crop) or request.crop
    try:
        report = _threshold_evaluator.evaluate(request.telemetry, crop=crop_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    # Shape expected by the frontend client in src/lib/api.ts.
    return {
        "summary": report.as_summary_dict(),
        "flags": [dataclasses.asdict(flag) for flag in report.flagged_flags()],
        "llm_alert_block": report.llm_alert_block,
    }
