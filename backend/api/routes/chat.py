"""
chat.py — Soil Doctor API: RAG Chat Endpoint

Responsibility:
    Exposes a single POST /chat endpoint that ties together the full
    RAG pipeline:

        Request (ChatRequest)
            ↓
        rag_engine.hybrid_search(query)     ← retrieves context chunks
            ↓
        chat_llm.generate_rag_response()    ← calls Gemini with grounded prompt
            ↓
        Response (ChatResponse)

FastAPI Integration Pattern:
    This router is NOT self-contained — it depends on a RAGEngine instance
    that must be initialised once at application startup (model loading takes
    ~10–30s). The engine is provided via FastAPI's dependency injection system.

    In main.py:
        from contextlib import asynccontextmanager
        from backend.rag.rag_engine import RAGEngine
        from backend.api.routes import chat as chat_router

        engine = RAGEngine()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            engine.initialize()          # Runs once on startup
            chat_router.set_engine(engine)
            yield
            engine.shutdown()            # Runs once on shutdown

        app = FastAPI(lifespan=lifespan)
        app.include_router(chat_router.router, prefix="/api/v1")

CORS:
    This router does not configure CORS — that belongs on the FastAPI app
    in main.py, since the React frontend will need CORS headers on all routes.

Error Handling Strategy:
    HTTP 400  — invalid/empty query from the client
    HTTP 503  — RAGEngine not yet initialised (app is still starting up)
    HTTP 500  — internal error (Gemini API failure, unexpected exception)
    All errors return a structured JSON body: {"detail": "<message>"}
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.rag.chat_llm import RAGResponse, generate_rag_response
from backend.rag.rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(
    prefix="/chat",
    tags=["RAG Chatbot"],
)

# ---------------------------------------------------------------------------
# Module-level engine reference
# Populated by set_engine() during application lifespan startup.
# Using a module-level variable (rather than app.state) keeps this router
# independently testable — tests can call set_engine(mock_engine) directly.
# ---------------------------------------------------------------------------
_engine: RAGEngine | None = None


def set_engine(engine: RAGEngine) -> None:
    """
    Inject the initialised RAGEngine instance into this router.
    Must be called from the FastAPI lifespan context AFTER engine.initialize().

    Args:
        engine : A fully initialised RAGEngine instance.
    """
    global _engine
    _engine = engine
    logger.info("RAGEngine injected into chat router.")


def _require_engine() -> RAGEngine:
    """
    Internal guard that raises HTTP 503 if the engine is not yet ready.
    Called at the START of each endpoint body (after Pydantic body validation)
    so that 422 errors from invalid requests are returned before 503 errors.
    """
    if _engine is None or not _engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The RAG engine is still initialising. "
                "Model loading typically takes 10–30 seconds. "
                "Please retry in a moment."
            ),
        )
    return _engine


# ---------------------------------------------------------------------------
# Pydantic I/O models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Incoming request payload for POST /chat.

    Fields:
        query   : The user's question. Must be non-empty after stripping
                  whitespace. Max 2000 characters to prevent abuse.
        top_k   : Number of chunks to retrieve from the knowledge base.
                  The retrieval engine will internally rerank and return
                  the top FINAL_TOP_N (3) of these for generation.
    """
    query: str = Field(
        ...,
        min_length = 1,
        max_length = 2000,
        description = "The agronomic question from the user.",
        examples = ["What is the correct lime application rate for acidic maize soil?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidate chunks to retrieve before reranking.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be blank or whitespace-only.")
        return stripped


class SourceReference(BaseModel):
    """A single source document reference included in the response."""
    filename: str = Field(..., description="Source document filename.")


class ChatResponse(BaseModel):
    """
    Response payload from POST /chat.

    Fields:
        answer          : The model-generated answer (markdown formatted).
        sources         : Documents the answer was grounded in.
        chunks_used     : Number of retrieved chunks provided to the model.
        grounded        : False if the knowledge base had no relevant context.
        generation_time : Seconds the Gemini API call took.
        model           : Gemini model identifier used.
    """
    answer: str = Field(..., description="The AI-generated agronomic answer.")
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Source documents cited in the answer.",
    )
    chunks_used: int = Field(
        ..., description="Number of knowledge base chunks provided as context."
    )
    grounded: bool = Field(
        ...,
        description=(
            "True if the answer is grounded in retrieved knowledge base content. "
            "False indicates the knowledge base had no relevant context for this query."
        ),
    )
    generation_time: float = Field(
        ..., description="Gemini API wall-clock time in seconds."
    )
    model: str = Field(..., description="Gemini model used for generation.")


# ---------------------------------------------------------------------------
# Health sub-endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="RAG engine readiness check",
    response_description="Engine status and document count.",
)
async def chat_health() -> dict:
    """
    Lightweight readiness probe for the RAG pipeline.
    Returns the number of chunks in the knowledge base so the frontend
    can warn users if no documents have been ingested yet.
    """
    engine = _require_engine()
    doc_count = engine.document_count()
    return {
        "status": "ready",
        "knowledge_base_chunks": doc_count,
        "knowledge_base_populated": doc_count > 0,
    }


# ---------------------------------------------------------------------------
# Primary chat endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the Soil Doctor a question",
    response_description="AI-generated agronomic answer grounded in the knowledge base.",
)
async def post_chat(request: ChatRequest) -> ChatResponse:
    """
    Execute the full RAG pipeline for a single conversational turn.

    **Pipeline:**
    1. Validate and sanitise the incoming query (Pydantic, runs first).
    2. Check the RAG engine is ready (HTTP 503 if not).
    3. Run hybrid search (dense + sparse + rerank) over the knowledge base.
    4. Pass retrieved chunks to Gemini with a grounding-enforced system prompt.
    5. Return the structured ChatResponse.

    **Anti-hallucination guarantee:**
    The generation layer's system instruction explicitly forbids the model
    from answering questions not covered by the retrieved context. If no
    relevant chunks are found, the response will acknowledge the gap
    rather than fabricate an answer.
    """
    # Engine check AFTER Pydantic body validation — ensures 422 takes priority
    # over 503 when the request body is malformed.
    engine = _require_engine()
    query = request.query   # already stripped by the Pydantic validator

    logger.info("POST /chat | Query: '%s...' | top_k: %d", query[:60], request.top_k)

    # ── Step 1: Retrieve relevant chunks ──────────────────────────────
    try:
        chunks = engine.hybrid_search(query, top_k=request.top_k)
    except Exception as exc:
        logger.exception("hybrid_search failed for query: '%s'", query[:60])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    logger.info("Retrieval complete | Chunks returned: %d", len(chunks))

    # ── Step 2: Generate the grounded response ─────────────────────────
    try:
        rag_result: RAGResponse = generate_rag_response(
            user_query=query,
            retrieved_chunks=chunks,
        )
    except EnvironmentError as exc:
        # Missing API key — configuration problem, not a client error
        logger.error("API key error in generate_rag_response: %s", exc)
        # Provide a clearer, actionable message for developers running locally.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Server configuration error: missing or invalid Google API key. "
                "Set the GOOGLE_API_KEY environment variable with a valid key."
            ),
        ) from exc
    except RuntimeError as exc:
        logger.exception("generate_rag_response failed for query: '%s'", query[:60])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        ) from exc

    # ── Step 3: Shape and return the response ──────────────────────────
    sources = [SourceReference(filename=s) for s in rag_result.sources]

    response = ChatResponse(
        answer=rag_result.answer,
        sources=sources,
        chunks_used=rag_result.chunks_used,
        grounded=rag_result.grounded,
        generation_time=rag_result.generation_time_seconds,
        model=rag_result.model_name,
    )

    logger.info(
        "POST /chat complete | Grounded: %s | Sources: %s | %.3fs",
        response.grounded,
        [s.filename for s in sources],
        response.generation_time,
    )
    return response
