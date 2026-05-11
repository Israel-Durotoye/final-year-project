"""
chat_llm.py — Soil Doctor RAG Pipeline: Generation Layer

Responsibility:
    Accepts a user query and a list of RetrievedChunk objects from
    rag_engine.py, constructs a grounded prompt, and calls Gemini to
    generate a contextual answer strictly anchored to the retrieved text.

    This module deliberately contains NO retrieval logic. It receives
    pre-retrieved, pre-reranked chunks from rag_engine.py and converts
    them into a natural-language answer.

Anti-Hallucination Contract (enforced in the system instruction):
    The model is explicitly told to:
      ① Answer ONLY from the provided context chunks — never from prior knowledge.
      ② Cite which source document each claim comes from.
      ③ If the context does not contain the answer, say so directly
         rather than interpolating from general agriculture knowledge.
      ④ Never recommend specific product brands not mentioned in the context.

SDK Note:
    Uses `google-genai` (the current Google SDK, v1.x) for consistency with
    prescriptive_llm.py. The older `google-generativeai` package reached
    end-of-life and should not be introduced alongside the current SDK
    in the same codebase.

Dependencies:
    pip install google-genai
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from google import genai
from google.genai import types as genai_types

if TYPE_CHECKING:
    # Avoids a circular import in type hints — rag_engine is in the same package
    from backend.rag.rag_engine import RetrievedChunk

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.3     # Slightly higher than prescriptive engine —
                               # conversational fluency matters more here,
                               # but still low enough to suppress fabrication.
DEFAULT_MAX_TOKENS  = 1024
DEFAULT_TOP_P       = 0.90

# Relevance gate: chunks with a rerank_score below this threshold are
# considered noise and excluded from the context window.
# bge-reranker-large produces raw logit scores; values below ~-3.0 are
# typically irrelevant to the query.
RERANK_SCORE_THRESHOLD = -3.0


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """
    Structured result of a generate_rag_response() call.

    Attributes:
        answer                  : Model-generated answer string (markdown).
        sources                 : Deduplicated list of source filenames cited.
        chunks_used             : Number of retrieved chunks included in context.
        chunks_above_threshold  : Number of chunks that cleared the relevance gate.
        generation_time_seconds : Wall-clock time of the Gemini API call.
        model_name              : Model identifier used for generation.
        grounded                : True if at least one context chunk was provided.
        timestamp_utc           : ISO 8601 timestamp of generation.
    """
    answer: str
    sources: list[str]
    chunks_used: int
    chunks_above_threshold: int
    generation_time_seconds: float
    model_name: str
    grounded: bool
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Core function: generate_rag_response
# ---------------------------------------------------------------------------

def generate_rag_response(
    user_query: str,
    retrieved_chunks: list["RetrievedChunk"],
    *,
    model_name: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_TOKENS,
    rerank_threshold: float = RERANK_SCORE_THRESHOLD,
) -> RAGResponse:
    """
    Generate a RAG-grounded answer for a user query using retrieved context.

    Args:
        user_query          : The farmer or agronomist's natural-language question.
        retrieved_chunks    : Ordered list of RetrievedChunk objects from
                              rag_engine.hybrid_search(). Must be sorted by
                              rerank_score descending (guaranteed by rag_engine).
        model_name          : Gemini model to use. Default: 'gemini-1.5-flash'.
        api_key             : Google API key. Falls back to GOOGLE_API_KEY env var.
        temperature         : Sampling temperature. Default: 0.3.
        max_output_tokens   : Hard cap on response length. Default: 1024.
        rerank_threshold    : Cross-encoder score below which chunks are excluded.
                              Set to -float('inf') to include all chunks regardless
                              of score (not recommended for production).

    Returns:
        RAGResponse — structured result with the answer and generation metadata.

    Raises:
        EnvironmentError : If no API key is available.
        RuntimeError     : If the Gemini API call fails or returns a blocked response.
    """

    # ------------------------------------------------------------------ #
    # 1.  Filter chunks by relevance threshold                            #
    # ------------------------------------------------------------------ #
    # Exclude chunks where the cross-encoder judged the document as
    # irrelevant to the query. This prevents the LLM from picking up
    # noise that somehow appeared in the candidate pool.
    qualifying_chunks = [
        c for c in retrieved_chunks
        if c.rerank_score >= rerank_threshold
    ]

    logger.info(
        "RAG generation | Query: '%s...' | Total chunks: %d | "
        "Above threshold (%.1f): %d",
        user_query[:60],
        len(retrieved_chunks),
        rerank_threshold,
        len(qualifying_chunks),
    )

    # ------------------------------------------------------------------ #
    # 2.  Build the context block and collect source citations            #
    # ------------------------------------------------------------------ #
    context_block, sources = _build_context_block(qualifying_chunks)

    # ------------------------------------------------------------------ #
    # 3.  Construct the system instruction and user content               #
    # ------------------------------------------------------------------ #
    system_instruction = _build_system_instruction()
    user_content = _build_user_content(
        query=user_query,
        context_block=context_block,
        has_context=bool(qualifying_chunks),
    )

    # ------------------------------------------------------------------ #
    # 4.  Initialise the Gemini client                                    #
    # ------------------------------------------------------------------ #
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "No Google API key found. Provide it via the `api_key` argument "
            "or set the GOOGLE_API_KEY environment variable."
        )
    client = genai.Client(api_key = resolved_key)

    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=DEFAULT_TOP_P,
        candidate_count=1,
        system_instruction=system_instruction,
        safety_settings=[
            genai_types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE",
            ),
        ],
    )

    # ------------------------------------------------------------------ #
    # 5.  Execute the API call                                            #
    # ------------------------------------------------------------------ #
    t_start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_content,
            config=config,
        )
    except Exception as exc:
        # Detect common API-key / argument errors and surface a clearer
        # environment-style error so FastAPI can map it to a helpful message.
        logger.error("Gemini API call failed in chat_llm: %s", exc)
        msg = str(exc)
        if (
            "API key not valid" in msg
            or "API_KEY_INVALID" in msg
            or "INVALID_ARGUMENT" in msg
            or "No Google API key" in msg
        ):
            raise EnvironmentError(
                "Invalid or missing Google API key. "
                "Set the GOOGLE_API_KEY environment variable to a valid key."
            ) from exc
        raise RuntimeError(
            f"Gemini API call failed for model '{model_name}'. "
            f"Original error: {exc}"
        ) from exc
    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------ #
    # 6.  Extract the response text                                       #
    # ------------------------------------------------------------------ #
    try:
        answer_text = response.text
    except (ValueError, AttributeError) as exc:
        block_reason = "UNKNOWN"
        try:
            block_reason = response.prompt_feedback.block_reason.name
        except AttributeError:
            pass
        raise RuntimeError(
            f"Gemini returned no usable text. Block reason: {block_reason}."
        ) from exc

    logger.info(
        "RAG response generated in %.3fs | Model: %s | "
        "Chunks used: %d | Sources: %s",
        elapsed,
        model_name,
        len(qualifying_chunks),
        sources,
    )

    return RAGResponse(
        answer=answer_text,
        sources=sources,
        chunks_used=len(qualifying_chunks),
        chunks_above_threshold=len(qualifying_chunks),
        generation_time_seconds=round(elapsed, 3),
        model_name=model_name,
        grounded=bool(qualifying_chunks),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_context_block(chunks: list["RetrievedChunk"]) -> tuple[str, list[str]]:
    """
    Format the retrieved chunks into a numbered context block for injection
    into the prompt, and collect deduplicated source citations.

    Returns:
        (context_block_string, sorted_unique_source_list)
    """
    if not chunks:
        return "(No relevant context was found in the knowledge base.)", []

    lines: list[str] = []
    seen_sources: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        source_label = f"{chunk.source}, p.{chunk.page}" if chunk.page else chunk.source
        lines.append(f"[CHUNK {i}] Source: {source_label}")
        lines.append(f"Relevance score: {chunk.rerank_score:.3f}")
        lines.append(chunk.text.strip())
        lines.append("")   # blank line separator between chunks

        if chunk.source not in seen_sources:
            seen_sources.append(chunk.source)

    return "\n".join(lines).strip(), sorted(seen_sources)


def _build_system_instruction() -> str:
    """
    Build the privileged system instruction that establishes the Soil Doctor
    RAG persona and enforces the grounding constraint.

    Kept in a dedicated function so it can be unit-tested independently
    and easily updated without touching generate_rag_response().
    """
    return """You are the Soil Doctor — an expert agronomist and soil scientist \
with deep knowledge of tropical and sub-Saharan African cereal crop production, \
soil chemistry, and precision fertilisation. You are embedded in a digital \
Decision Support System used by farmers and agricultural extension officers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUNDING RULES — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — CONTEXT FIDELITY:
You MUST base your answer EXCLUSIVELY on the KNOWLEDGE BASE CONTEXT provided \
in the user message. Do not use any information from your pre-training that \
is not present in or directly implied by the provided context chunks.

RULE 2 — HONEST GAPS:
If the provided context does not contain enough information to answer the \
question accurately, you MUST respond with a clear statement such as: \
"The documents in my current knowledge base do not contain specific information \
about [topic]. I recommend consulting [relevant resource type]." \
NEVER fabricate facts, invent statistics, or speculate beyond what the \
context supports.

RULE 3 — CITATION:
For every factual claim, cite the source chunk it came from using the format \
(Source: [filename], p.[page]). If a claim is supported by multiple chunks, \
cite all of them.

RULE 4 — PRODUCT DISCIPLINE:
Only mention specific fertiliser products, chemical compounds, or brand names \
that appear explicitly in the context chunks. Never suggest products from \
general agricultural knowledge.

RULE 5 — TONE:
Write for a technically literate but non-academic audience. Use plain language \
where possible, define agronomic jargon on first use, and structure longer \
answers with clear headings where helpful."""


def _build_user_content(
    query: str,
    context_block: str,
    has_context: bool,
) -> str:
    """
    Assemble the user-turn content block that delivers the context chunks
    and the user's question to the model.
    """
    if not has_context:
        # Explicit signal to the model that no context was retrieved —
        # this triggers RULE 2 (honest gaps) in the system instruction.
        context_note = (
            "⚠️  NOTE: The retrieval system returned no context chunks that "
            "meet the relevance threshold for this query. You must apply "
            "RULE 2 and acknowledge the knowledge gap."
        )
    else:
        context_note = (
            "The following context chunks have been retrieved from the agronomic "
            "knowledge base and reranked by a cross-encoder for relevance. "
            "Use ONLY this information to answer the question below."
        )

    return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE BASE CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_note}

{context_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{query.strip()}

Answer below, following all rules in your operating charter:"""
