"""
main.py — Soil Doctor: FastAPI Application Entry Point

This file is the single wiring point for the entire backend.
Its only jobs are:
  1. Manage the RAGEngine lifecycle (startup / shutdown)
  2. Register all API routers with their URL prefixes
  3. Configure CORS for the React frontend
  4. Expose the health check

To run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import chat as chat_router
from backend.rag.rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton RAGEngine — one instance shared across all requests
# ---------------------------------------------------------------------------
_rag_engine = RAGEngine()


# ---------------------------------------------------------------------------
# Application Lifespan — startup and shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup; everything after runs at shutdown.

    Model loading (SentenceTransformer + FlagReranker) happens here —
    once, synchronously, before the server accepts any requests.
    """
    logger.info("Soil Doctor backend starting up...")
    _rag_engine.initialize()
    chat_router.set_engine(_rag_engine)
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
        "Combines a RAG chatbot (Tab 1) with a real-time sensor-driven "
        "prescriptive engine (Tab 2)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS — allow the React frontend (localhost:8080) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(chat_router.router, prefix="/api/v1")
# Future routers (add here when built):
# app.include_router(prescriptive_router.router, prefix="/api/v1")


# ── Root health check ────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root() -> dict:
    return {
        "service": "Soil Doctor API",
        "status": "running",
        "docs": "/docs",
    }
