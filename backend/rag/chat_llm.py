"""
chat_llm.py — Soil Doctor RAG Pipeline: Generation Layer

Responsibility:
    Accepts a user query and a list of RetrievedChunk objects from
    rag_engine.py, constructs a grounded prompt, and calls the Groq API
    to generate a contextual answer strictly anchored to the retrieved text.

    This module deliberately contains NO retrieval logic. It receives
    pre-retrieved, pre-reranked chunks from rag_engine.py and converts
    them into a natural-language answer.
"""

from __future__ import annotations

import logging
import os
import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from openai import OpenAI
from supabase import create_client, Client

if TYPE_CHECKING:
    from backend.rag.rag_engine import RetrievedChunk

# ---------------------------------------------------------------------------
# Logging & Constants
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.3     
DEFAULT_MAX_TOKENS  = 1024
DEFAULT_TOP_P       = 0.90
RERANK_SCORE_THRESHOLD = -3.0

# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    chunks_used: int
    chunks_above_threshold: int
    generation_time_seconds: float
    model_name: str
    grounded: bool
    timestamp_utc: str

# ---------------------------------------------------------------------------
# Core function
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
    
    qualifying_chunks = [c for c in retrieved_chunks if c.rerank_score >= rerank_threshold]

    logger.info(
        "RAG generation | Query: '%s...' | Total chunks: %d | Above threshold: %d",
        user_query[:60], len(retrieved_chunks), len(qualifying_chunks)
    )

    context_block, sources = _build_context_block(qualifying_chunks)
    system_instruction = _build_system_instruction()
    user_content = _build_user_content(user_query, context_block, bool(qualifying_chunks))

    resolved_key = api_key or os.environ.get("GROQ_API_KEY")
    if not resolved_key:
        raise EnvironmentError("No Groq API key found. Set the GROQ_API_KEY environment variable.")

    # Initialise standard OpenAI client pointing to Groq's blazing fast LPUs
    client = OpenAI(api_key=resolved_key, base_url="https://api.groq.com/openai/v1")

    # ------------------------------------------------------------------ #
    # Tool Registration
    # ------------------------------------------------------------------ #
    def get_live_sensor_data(node_id: str) -> dict:
        # Read Supabase credentials from environment
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            return {"error": "Supabase credentials not configured. Set SUPABASE_URL and SUPABASE_KEY."}

        client: Client = create_client(supabase_url, supabase_key)

        # Query the most recent telemetry row for the given node_id
        try:
            result = (
                client
                .table("sensor_telemetry")
                .select("*")
                .eq("node_id", node_id)
                .order("timestamp_utc", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise RuntimeError(f"Supabase query failed: {exc}") from exc

        # Support both result.data attribute and dict-like response
        data = getattr(result, "data", None) or (result.get("data") if isinstance(result, dict) else None)

        if data and len(data) > 0:
            return data[0]
        return {"error": f"No live telemetry found for {node_id}. Ensure the node is online."}

    tools = [{
        "type": "function",
        "function": {
            "name": "get_live_sensor_data",
            "description": "Return real-time sensor telemetry for a single hardware node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node identifier, e.g. NODE_01"},
                },
                "required": ["node_id"],
            },
        }
    }]

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]

    # ------------------------------------------------------------------ #
    # Execution & Tool Loop
    # ------------------------------------------------------------------ #
    t_start = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_output_tokens,
            top_p=DEFAULT_TOP_P,
        )
    except Exception as exc:
        if "400" in str(exc) or "tool_use_failed" in str(exc):
            logger.warning("Groq tool parsing failed (likely hallucinated tool arguments). Retrying without tools. Error: %s", exc)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                top_p=DEFAULT_TOP_P,
            )
        else:
            raise RuntimeError(f"Groq API call failed: {exc}") from exc

    elapsed = time.perf_counter() - t_start
    message = response.choices[0].message

    # Check if Llama 3 requested to use the sensor tool
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        logger.info("Model requested function call: %s", tool_call.function.name)

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}
            
        node_id = args.get("node_id", "")

        try:
            tool_result = get_live_sensor_data(node_id)
        except Exception as exc:
            raise RuntimeError(f"Tool execution failed: {exc}") from exc

        # Append exact conversation history for OpenAI tool strictness
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(tool_result),
        })

        # Second trip to get the final answer with telemetry included
        t_start2 = time.perf_counter()
        response2 = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="none",
            temperature=temperature,
            max_tokens=max_output_tokens,
            top_p=DEFAULT_TOP_P,
        )
        elapsed += time.perf_counter() - t_start2
        answer_text = response2.choices[0].message.content
    else:
        answer_text = message.content

    if not answer_text:
        raise RuntimeError("Groq returned no usable text.")

    logger.info("RAG response generated in %.3fs | Model: %s", elapsed, model_name)

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
    if not chunks:
        return "(No relevant context was found in the knowledge base.)", []

    lines, seen_sources = [], []
    for i, chunk in enumerate(chunks, start=1):
        source_label = f"{chunk.source}, p.{chunk.page}" if hasattr(chunk, 'page') and chunk.page else chunk.source
        lines.append(f"[CHUNK {i}] Source: {source_label}")
        lines.append(f"Relevance score: {chunk.rerank_score:.3f}")
        lines.append(chunk.text.strip())
        lines.append("") 

        if chunk.source not in seen_sources:
            seen_sources.append(chunk.source)

    return "\n".join(lines).strip(), sorted(seen_sources)

def _build_system_instruction() -> str:
    return """You are the Soil Doctor — an expert agronomist and soil scientist with deep knowledge of tropical and sub-Saharan African cereal crop production, soil chemistry, and precision fertilisation. You are embedded in a digital Decision Support System used by farmers and agricultural extension officers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUNDING RULES — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — CONTEXT AND TELEMETRY FIDELITY:
You must base your answer on the KNOWLEDGE BASE CONTEXT provided below, AND on any live telemetry data provided directly in the user's question. Use the knowledge base to guide your agronomic recommendations based on the telemetry data.

RULE 2 — HONEST GAPS AND EXPERT DIAGNOSIS:
If the user asks a general question and no relevant context is provided, you must acknowledge the gap (e.g., "The documents in my knowledge base do not contain specific information about [topic]."). However, if the user provides specific sensor telemetry data (e.g., Nitrogen, Phosphorus, Moisture values), you MUST act as an expert agronomist, analyze the data, and provide specific recommendations based on general agronomic principles, even if that specific node or exact scenario is not covered in the knowledge base.

RULE 3 — NO CITATIONS:
Do NOT include source names, filenames, or citations anywhere in your response.

RULE 4 — PRODUCT DISCIPLINE:
Only mention specific fertiliser products, chemical compounds, or brand names that appear explicitly in the context chunks, unless you are making a general agronomic recommendation based on telemetry data.

RULE 5 — TONE AND PRESCRIPTIVE FOCUS:
Provide direct, highly concise, and action-oriented answers. You are a PRESCRIPTIVE MAINTENANCE SYSTEM. Do not output conversational filler (e.g., "Since we don't have any live sensor data...", "Here is your diagnostic report..."). Do not just regurgitate the telemetry values back to the user. Instead, tell the user exactly WHAT needs to be done and WHY, focusing immediately on the most critical deviations (e.g., "Apply nitrogen fertilizer immediately because N levels are critically low at 22ppm"). Use concise bullet points for actionable tasks."""

def _build_user_content(query: str, context_block: str, has_context: bool) -> str:
    if not has_context:
        context_note = "⚠️ NOTE: The retrieval system returned no knowledge base chunks. However, if the user provided telemetry data in their question, you MUST analyze it using your expert agronomic knowledge and provide a diagnosis."
    else:
        context_note = "The following context chunks have been retrieved from the agronomic knowledge base. Use this information to answer the question, especially if it relates to diagnosing telemetry."

    return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE BASE CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_note}

{context_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{query.strip()}

Answer below, following all rules in your operating charter:
"""