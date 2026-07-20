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

from backend.rag import diagnostics, prescriptions, intent_classifier, prompt_router
from backend.ml import lstm_inference

try:
    import tiktoken
except ImportError:
    tiktoken = None

if TYPE_CHECKING:
    from backend.rag.rag_engine import RetrievedChunk

# ---------------------------------------------------------------------------
# Logging & Constants
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.3     
DEFAULT_MAX_TOKENS  = 1024
DEFAULT_TOP_P       = 0.90
RERANK_SCORE_THRESHOLD = -3.0

# Token limits for Groq Llama 3.1 8B (conservative estimates)
LLAMA_3_1_CONTEXT_LIMIT = 4096  # tokens
LLAMA_3_1_RESERVE = 512         # reserved for response
MAX_CONTEXT_TOKENS = LLAMA_3_1_CONTEXT_LIMIT - LLAMA_3_1_RESERVE - DEFAULT_MAX_TOKENS

# Groq API resilience
GROQ_MAX_RETRIES = 3
GROQ_INITIAL_BACKOFF = 1.0  # seconds
GROQ_MAX_BACKOFF = 32.0     # seconds
GROQ_BACKOFF_MULTIPLIER = 2.0

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
# Utilities — Token counting, truncation, and API resilience
# ---------------------------------------------------------------------------

def _estimate_token_count(text: str) -> int:
    """
    Estimate token count for a string.
    Uses tiktoken if available; falls back to rough 4-char-per-token heuristic.
    """
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except Exception:
            pass
    # Fallback: rough estimate (average ~4 chars per token for English)
    return max(1, len(text) // 4)


def _truncate_context_if_needed(context_block: str, query: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> tuple[str, bool]:
    """
    Truncate context block if it would exceed token limits.
    
    Returns:
        (truncated_context, was_truncated)
    """
    query_tokens = _estimate_token_count(query)
    context_tokens = _estimate_token_count(context_block)
    
    if context_tokens + query_tokens <= max_tokens:
        return context_block, False
    
    # Need to truncate: work backwards from the end, removing chunks
    lines = context_block.split("\n")
    truncated_lines = []
    current_tokens = 0
    target = max_tokens - query_tokens - 100  # 100-token safety margin
    
    for line in lines:
        line_tokens = _estimate_token_count(line)
        if current_tokens + line_tokens > target:
            break
        truncated_lines.append(line)
        current_tokens += line_tokens
    
    if not truncated_lines:
        # Even first chunk is too large; keep it anyway and warn
        logger.warning(
            "Context truncation: first chunk alone exceeds token limit. "
            "Keeping full first chunk; total context may exceed Groq limit."
        )
        return context_block, True
    
    truncated = "\n".join(truncated_lines)
    truncated += "\n\n[...context truncated due to length...]"
    logger.warning(
        "Context truncated from %d to %d tokens to fit token budget.",
        context_tokens, _estimate_token_count(truncated)
    )
    return truncated, True


def _call_groq_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs
) -> dict:
    """
    Wrap client.chat.completions.create with exponential backoff retry.
    Handles transient 429/503 errors and network timeouts.
    
    Args:
        client: OpenAI-compatible client (pointing to Groq).
        model: Model identifier.
        messages: Chat messages.
        tools: Optional function tools.
        **kwargs: Additional arguments (temperature, max_tokens, etc.).
    
    Returns:
        Raw response object from client.chat.completions.create.
    
    Raises:
        RuntimeError: After MAX_RETRIES attempts, or for non-transient errors.
    """
    backoff = GROQ_INITIAL_BACKOFF
    last_exc = None
    
    for attempt in range(GROQ_MAX_RETRIES):
        try:
            if tools is not None:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    **kwargs
                )
            else:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
        except Exception as exc:
            exc_str = str(exc).lower()
            is_transient = any(
                err in exc_str
                for err in ["429", "503", "timeout", "connection", "temporarily unavailable"]
            )
            
            if not is_transient or attempt == GROQ_MAX_RETRIES - 1:
                # Not transient, or last attempt — raise immediately
                if not is_transient:
                    raise RuntimeError(f"Groq API call failed (non-transient): {exc}") from exc
                else:
                    last_exc = exc
                    break
            
            # Transient error — retry with backoff
            logger.warning(
                "Groq API transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1, GROQ_MAX_RETRIES, exc, backoff
            )
            time.sleep(backoff)
            backoff = min(GROQ_MAX_BACKOFF, backoff * GROQ_BACKOFF_MULTIPLIER)
    
    if last_exc:
        raise RuntimeError(
            f"Groq API call failed after {GROQ_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc
    
    raise RuntimeError("Groq API retry loop exited unexpectedly.")


def _refine_with_llm(client: OpenAI, raw_answer: str) -> str:
    """
    Post-processing refinement pass.

    Tries Google Gemini models first (free tier), then falls back to
    Groq Llama 3.1 8B Instant if Gemini quota is exhausted.
    """
    REFINE_INSTRUCTIONS = (
        "You are an output formatter for Soil Doctor, a precision agriculture assistant. "
        "Your ONLY job is to clean up the text you receive and return an improved version.\n\n"
        "Rules:\n"
        "1. Keep ALL factual content, numbers, and recommendations intact — do NOT remove data.\n"
        "2. Remove any robotic phrasing like 'Based on the context provided' or 'According to the retrieved chunks'.\n"
        "3. Use a warm, conversational but professional tone — like a knowledgeable agronomist talking to a farmer.\n"
        "4. Break long walls of text into short, readable paragraphs.\n"
        "5. Use bullet points or numbered lists where appropriate for action items.\n"
        "6. Use bold (**text**) for key values, parameters, or critical warnings.\n"
        "7. Do NOT add any new information that was not in the original text.\n"
        "8. Do NOT add greetings, sign-offs, or disclaimers.\n"
        "9. Return ONLY the refined text — no meta-commentary about what you changed."
    )

    # ── Strategy 1: Try Gemini models (free tier) ─────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        _GEMINI_MODELS = ["gemini-2.0-flash-lite", "gemini-1.5-flash"]
        try:
            from google import genai

            gemini_client = genai.Client(api_key=gemini_key)
            for model_name in _GEMINI_MODELS:
                try:
                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=f"{REFINE_INSTRUCTIONS}\n\n--- TEXT TO REFINE ---\n{raw_answer}",
                        config=genai.types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=DEFAULT_MAX_TOKENS,
                        ),
                    )
                    refined = response.text
                    if refined and len(refined.strip()) > 20:
                        logger.info("Gemini refinement pass completed (%s).", model_name)
                        return refined.strip()
                    else:
                        logger.warning("Gemini %s returned empty/short result; trying next.", model_name)
                except Exception as exc:
                    logger.warning("Gemini %s failed (%s); trying next model.", model_name, exc)
                    continue
        except ImportError:
            logger.warning("google-genai not installed; skipping Gemini refinement.")

    # ── Strategy 2: Fall back to Groq 8B Instant (free tier) ──────────
    try:
        response = _call_groq_with_retry(
            client,
            "llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": REFINE_INSTRUCTIONS},
                {"role": "user", "content": raw_answer},
            ],
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        refined = response.choices[0].message.content
        if refined and len(refined.strip()) > 20:
            logger.info("Groq 8B refinement fallback completed successfully.")
            return refined.strip()
    except Exception as exc:
        logger.warning("Groq 8B refinement fallback also failed (%s).", exc)

    logger.warning("All refinement strategies exhausted; returning raw answer.")
    return raw_answer


def _extract_telemetry_from_context(query: str, context: str) -> dict | None:
    """
    Attempt to extract telemetry parameters from user query or context.
    
    Looks for patterns like "pH: 5.2" or "nitrogen: 45 ppm" and extracts
    numeric values for diagnostic engine.
    
    Returns dict with keys: ph, nitrogen, phosphorus, potassium, moisture,
    temperature, salinity, organic_matter, context, timestamp
    Or None if insufficient telemetry data found.
    """
    import re
    from datetime import datetime
    
    combined_text = f"{query} {context}".lower()
    telemetry = {}
    
    # pH extraction
    ph_match = re.search(r'ph\s*[:=]?\s*([\d.]+)', combined_text)
    if ph_match:
        telemetry['ph'] = float(ph_match.group(1))
    
    # Nitrogen (ppm)
    n_match = re.search(r'(?:nitrogen|n)\s*[:=]?\s*([\d.]+)\s*(?:ppm)?', combined_text)
    if n_match:
        telemetry['nitrogen'] = float(n_match.group(1))
    
    # Phosphorus (ppm)
    p_match = re.search(r'(?:phosphorus|phosphate|p)\s*[:=]?\s*([\d.]+)\s*(?:ppm)?', combined_text)
    if p_match:
        telemetry['phosphorus'] = float(p_match.group(1))
    
    # Potassium (ppm)
    k_match = re.search(r'(?:potassium|potash|k)\s*[:=]?\s*([\d.]+)\s*(?:ppm)?', combined_text)
    if k_match:
        telemetry['potassium'] = float(k_match.group(1))
    
    # Moisture (%)
    moisture_match = re.search(r'(?:moisture|water|humidity)\s*[:=]?\s*([\d.]+)\s*(?:%)?', combined_text)
    if moisture_match:
        telemetry['moisture'] = float(moisture_match.group(1))
    
    # Temperature (°C or C)
    temp_match = re.search(r'(?:temperature|temp)\s*[:=]?\s*([\d.]+)\s*(?:°?c|celsius)?', combined_text)
    if temp_match:
        telemetry['temperature'] = float(temp_match.group(1))
    
    # Salinity/EC (dS/m)
    salinity_match = re.search(r'(?:salinity|ec|conductivity)\s*[:=]?\s*([\d.]+)\s*(?:ds/m)?', combined_text)
    if salinity_match:
        telemetry['salinity'] = float(salinity_match.group(1))
    
    # Organic Matter (%)
    om_match = re.search(r'(?:organic\s+matter|om)\s*[:=]?\s*([\d.]+)\s*(?:%)?', combined_text)
    if om_match:
        telemetry['organic_matter'] = float(om_match.group(1))
    
    # Only proceed if at least 3 parameters found
    if len(telemetry) < 3:
        return None
    
    # Add timestamp and context
    telemetry['timestamp'] = datetime.now().isoformat()
    telemetry['context'] = query
    
    logger.debug("Extracted telemetry for diagnostic: %s", telemetry)
    return telemetry


def generate_rag_response(
    user_query: str,
    retrieved_chunks: list["RetrievedChunk"],
    *,
    conversation_history: list[dict[str, str]] | None = None,
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
    
    # ── Token overflow protection ──────────────────────────────────────
    context_block, was_truncated = _truncate_context_if_needed(context_block, user_query)
    
    # Extract telemetry first to pass to routing decision
    telemetry_params = None
    try:
        telemetry_params = _extract_telemetry_from_context(user_query, context_block)
    except Exception as exc:
        logger.debug("Failed to extract telemetry parameters: %s", exc)

    # Route query using prompt_router to dynamically select prompt instructions
    router = prompt_router.get_router()
    routing_decision = router.route(user_query, context_block, telemetry_params)
    
    system_instruction = routing_decision.template.system_instruction
    memory_preference = routing_decision.template.memory_preference
    requires_diagnostics = routing_decision.template.requires_diagnostics

    # ── Active Field Context Injection ────────────────────────────────────────
    # TODO: Connect these to actual frontend session variables later.
    # These values will eventually be passed in from the API request payload
    # (e.g., selected crop and growth stage from the dashboard session).
    active_crop   = "Maize"
    growth_stage  = "V6 Stage"
    augmented_user_prompt = (
        f"[Active Field Context: {active_crop} at {growth_stage}]\n\n"
        f"User Query: {user_query}"
    )
    # ─────────────────────────────────────────────────────────────────────────

    user_content = _build_user_content(
        augmented_user_prompt,
        context_block,
        bool(qualifying_chunks),
        conversation_history,
        memory_preference=memory_preference,
    )

    resolved_key = api_key or os.environ.get("GROQ_API_KEY")
    if not resolved_key:
        raise EnvironmentError("No Groq API key found. Set the GROQ_API_KEY environment variable.")

    # Initialise standard OpenAI client pointing to Groq's blazing fast LPUs
    client = OpenAI(
        api_key=resolved_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=60.0,
    )

    # ────────────────────────────────────────────────────────────────────
    # Tool Registration with graceful degradation
    # ────────────────────────────────────────────────────────────────────
    def get_live_sensor_data(node_id: str) -> dict:
        """
        Fetch live sensor telemetry from Supabase.
        Returns graceful error dict if Supabase unavailable.
        """
        # Read Supabase credentials from environment
        supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not configured; sensor data unavailable.")
            return {"status": "offline", "reason": "Supabase not configured"}

        try:
            supabase_client: Client = create_client(supabase_url, supabase_key)
        except Exception as exc:
            logger.warning("Failed to initialize Supabase client: %s", exc)
            return {"status": "offline", "reason": "Client initialization failed"}

        # Query the most recent telemetry row for the given node_id
        try:
            result = (
                supabase_client
                .table("capstone_dataset")
                .select("*")
                .eq("Node_ID", node_id)
                .order("Timestamp", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            logger.warning("Supabase query failed for node %s: %s", node_id, exc)
            return {"status": "offline", "reason": f"Query failed: {str(exc)[:50]}"}

        # Support both result.data attribute and dict-like response
        data = getattr(result, "data", None) or (result.get("data") if isinstance(result, dict) else None)

        if data and len(data) > 0:
            row = data[0]
            mapped = {
                "node_id": row.get("Node_ID"),
                "timestamp_utc": row.get("Timestamp"),
                "nitrogen": row.get("Nitrogen_mg_k"),
                "phosphorus": row.get("Phosphorus_m"),
                "potassium": row.get("Potassium_mg_"),
                "moisture": row.get("Moisture_%"),
                "temperature": row.get("Temperature_C"),
                "humidity": row.get("Humidity_%"),
                # Impute missing parameters
                "ph": 6.5,
                "salinity": 0.5,
                "organic_matter": 2.0,
            }
            return {"status": "online", **mapped}
        
        logger.warning("No telemetry found for node %s", node_id)
        return {"status": "offline", "reason": f"No data for node {node_id}"}

    def _run_moisture_prediction(node_id: str) -> str:
        """
        Fetch the last 48 hourly rows for *node_id* from Supabase, pass them
        to the LSTM inference wrapper, and return a JSON string suitable for
        use as a tool-call response.

        DB column names differ from the LSTM feature names used during training.
        The mapping below translates actual Supabase columns → model features:
            nitrogen_ppm    → nitrogen
            phosphorus_ppm  → phosphorus
            potassium_ppm   → potassium
            soil_moisture   → moisture
            soil_temperature_c → temperature
            ph, salinity, organic_matter → unchanged

        Falls back gracefully if Supabase is unavailable or the model artefacts
        are missing.
        """
        # Actual Supabase column names to fetch
        _DB_COLS = [
            "Nitrogen_mg_k", "Phosphorus_m", "Potassium_mg_",
            "Moisture_%", "Temperature_C",
        ]
        # Corresponding LSTM feature order (must match scaler/model training order)
        _FEATURE_ORDER = [
            "ph", "nitrogen", "phosphorus", "potassium",
            "moisture", "temperature", "salinity", "organic_matter",
        ]
        _REQUIRED_ROWS = 48

        supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("Supabase not configured; cannot run moisture prediction.")
            return json.dumps({
                "status": "offline",
                "message": "Sensor database unavailable. Cannot run moisture forecast.",
            })

        try:
            supabase_client: Client = create_client(supabase_url, supabase_key)
        except Exception as exc:
            logger.warning("Failed to init Supabase client for prediction: %s", exc)
            return json.dumps({"status": "offline", "message": "Database connection failed."})

        # Fetch the most recent 48 rows (ordered DESC, then reversed to chronological)
        try:
            result = (
                supabase_client
                .table("capstone_dataset")
                .select(", ".join(["Timestamp"] + _DB_COLS))
                .eq("Node_ID", node_id)
                .order("Timestamp", desc=True)
                .limit(_REQUIRED_ROWS)
                .execute()
            )
        except Exception as exc:
            logger.warning("Supabase query failed for moisture prediction (node %s): %s", node_id, exc)
            return json.dumps({"status": "offline", "message": f"Query failed: {str(exc)[:80]}"})

        rows = getattr(result, "data", None) or (result.get("data") if isinstance(result, dict) else None) or []

        if len(rows) < _REQUIRED_ROWS:
            msg = (
                f"Insufficient history for node {node_id}: "
                f"need {_REQUIRED_ROWS} rows, got {len(rows)}."
            )
            logger.warning(msg)
            return json.dumps({"status": "insufficient_data", "message": msg})

        # Reverse so data is chronological (oldest → newest)
        rows = list(reversed(rows))

        # Build 2-D array [48, n_features] in LSTM feature order, imputing missing features
        sensor_matrix = []
        for row in rows:
            sensor_matrix.append([
                6.5, # ph (imputed)
                float(row.get("Nitrogen_mg_k") or 0.0),
                float(row.get("Phosphorus_m") or 0.0),
                float(row.get("Potassium_mg_") or 0.0),
                float(row.get("Moisture_%") or 0.0),
                float(row.get("Temperature_C") or 0.0),
                0.5, # salinity (imputed)
                2.0, # organic_matter (imputed)
            ])

        try:
            predicted_moisture = lstm_inference.execute_moisture_prediction(sensor_matrix)
        except FileNotFoundError as exc:
            logger.error("LSTM model artefacts missing: %s", exc)
            return json.dumps({
                "status": "model_unavailable",
                "message": "Moisture prediction model is not yet deployed on this server.",
            })
        except Exception as exc:
            logger.error("LSTM inference failed: %s", exc)
            return json.dumps({"status": "error", "message": f"Prediction error: {str(exc)[:120]}"})

        logger.info(
            "Moisture prediction for node %s: %.4f%% (24-hour forecast)",
            node_id, predicted_moisture,
        )
        return json.dumps({
            "status": "success",
            "node_id": node_id,
            "predicted_moisture_pct": round(predicted_moisture, 4),
            "forecast_horizon": "24 hours",
            "model": "LSTM (Keras 3 / PyTorch backend)",
        })

    tools = [
        {
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
        },
        {
            "type": "function",
            "function": {
                "name": "execute_moisture_prediction",
                "description": (
                    "Predicts the soil moisture percentage 24 hours into the future. "
                    "Call this when the user asks about future field conditions, "
                    "survival, or upcoming irrigation needs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": (
                                "Node identifier to fetch the last 48 hours of sensor "
                                "readings from, e.g. NODE_01."
                            ),
                        },
                    },
                    "required": ["node_id"],
                },
            }
        },
    ]

    # ────────────────────────────────────────────────────────────────────
    # Structured Diagnostics (when telemetry is available)
    # ────────────────────────────────────────────────────────────────────
    diagnostic_context = ""
    if requires_diagnostics:
        try:
            if telemetry_params:
                # Run diagnostic engine
                diagnostic_engine = diagnostics.SoilDiagnosticEngine()
                diagnosis = diagnostic_engine.diagnose(**telemetry_params)
                
                # Generate action plan
                prescription_engine = prescriptions.PrescriptionEngine()
                action_plan = prescription_engine.generate_action_plan(diagnosis)
                
                # Format diagnostic results as JSON for LLM context
                diagnostic_json = {
                    "diagnosis": {
                        "issues": [
                            {
                                "parameter": issue.parameter,
                                "measured_value": issue.measured_value,
                                "optimal_range": issue.optimal_range,
                                "severity": issue.severity.name,
                                "description": issue.description,
                                "root_cause": issue.root_cause,
                            }
                            for issue in diagnosis.issues
                        ],
                        "severity_summary": diagnosis.severity_summary.name,
                        "interactions": diagnosis.interactions,
                        "timestamp": diagnosis.timestamp,
                    },
                    "action_plan": {
                        "critical_first_steps": action_plan.critical_first_steps,
                        "expected_timeline": action_plan.expected_timeline,
                        "corrective_actions": [
                            {
                                "priority": action.priority,
                                "action": action.action,
                                "target_parameter": action.target_parameter,
                                "severity": action.severity.name,
                                "impact": action.impact,
                                "dosage": action.dosage,
                                "timeline": action.timeline,
                                "reasoning": action.reasoning,
                            }
                            for action in action_plan.corrective_actions
                        ],
                        "monitoring_parameters": action_plan.monitoring_parameters,
                    }
                }
                
                diagnostic_context = f"\n\n[STRUCTURED DIAGNOSTIC RESULTS]\n{json.dumps(diagnostic_json, indent=2)}\n[END DIAGNOSTIC RESULTS]\n"
                logger.info("Diagnostics generated for telemetry with %d issues identified", len(diagnosis.issues))
        except Exception as exc:
            logger.debug("Could not generate structured diagnostics: %s (this is OK if telemetry wasn't in context)", exc)
            diagnostic_context = ""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content + diagnostic_context}
    ]

    # ────────────────────────────────────────────────────────────────────
    # Execution & Tool Loop with retries and graceful degradation
    # ────────────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    
    try:
        response = _call_groq_with_retry(
            client,
            model_name,
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_output_tokens,
            top_p=DEFAULT_TOP_P,
        )
    except Exception as exc:
        # If tool call fails, retry without tools
        if "400" in str(exc) or "tool" in str(exc).lower():
            logger.warning(
                "Groq API failed with tool context (likely hallucinated tool args). "
                "Retrying without tools. Error: %s", exc
            )
            try:
                response = _call_groq_with_retry(
                    client,
                    model_name,
                    messages,
                    tools=None,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    top_p=DEFAULT_TOP_P,
                )
            except Exception as retry_exc:
                raise RuntimeError(f"Groq API failed (both with and without tools): {retry_exc}") from retry_exc
        else:
            raise RuntimeError(f"Groq API call failed: {exc}") from exc

    elapsed = time.perf_counter() - t_start
    message = response.choices[0].message

    # Check if Llama 3 requested to use one of the registered tools
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        called_fn = tool_call.function.name
        logger.info("Model requested function call: %s", called_fn)

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse tool arguments: %s. Using empty args.", exc)
            args = {}

        node_id = args.get("node_id", "")

        # ── Branch: get_live_sensor_data ──────────────────────────────────
        if called_fn == "get_live_sensor_data":
            # Execute tool with graceful fallback
            tool_result = get_live_sensor_data(node_id)

            # If sensor data is offline, create a helpful fallback message
            if tool_result.get("status") == "offline":
                tool_result_content = json.dumps({
                    "status": "offline",
                    "message": "Sensor data currently offline. Using historical context and agronomic expertise."
                })
                logger.warning("Sensor tool returned offline status: %s", tool_result.get("reason"))
            else:
                tool_result_content = json.dumps(tool_result)

                # Since live telemetry was successfully retrieved, we now have sensor values!
                # Re-run diagnostics and dynamically update system instructions.
                telemetry_keys = ["ph", "nitrogen", "phosphorus", "potassium", "moisture", "temperature", "salinity", "organic_matter"]
                live_telemetry = {k: tool_result[k] for k in telemetry_keys if k in tool_result and tool_result[k] is not None}

                if live_telemetry:
                    try:
                        diagnostic_engine = diagnostics.SoilDiagnosticEngine()
                        diagnosis = diagnostic_engine.diagnose(**live_telemetry)

                        prescription_engine = prescriptions.PrescriptionEngine()
                        action_plan = prescription_engine.generate_action_plan(diagnosis)

                        diagnostic_json = {
                            "diagnosis": {
                                "issues": [
                                    {
                                        "parameter": issue.parameter,
                                        "measured_value": issue.measured_value,
                                        "optimal_range": issue.optimal_range,
                                        "severity": issue.severity.name,
                                        "description": issue.description,
                                        "root_cause": issue.root_cause,
                                    }
                                    for issue in diagnosis.issues
                                ],
                                "severity_summary": diagnosis.severity_summary.name,
                                "interactions": diagnosis.interactions,
                                "timestamp": diagnosis.timestamp,
                            },
                            "action_plan": {
                                "critical_first_steps": action_plan.critical_first_steps,
                                "expected_timeline": action_plan.expected_timeline,
                                "corrective_actions": [
                                    {
                                        "priority": action.priority,
                                        "action": action.action,
                                        "target_parameter": action.target_parameter,
                                        "severity": action.severity.name,
                                        "impact": action.impact,
                                        "dosage": action.dosage,
                                        "timeline": action.timeline,
                                        "reasoning": action.reasoning,
                                    }
                                    for action in action_plan.corrective_actions
                                ],
                                "monitoring_parameters": action_plan.monitoring_parameters,
                            }
                        }
                        diagnostic_context = f"\n\n[STRUCTURED DIAGNOSTIC RESULTS]\n{json.dumps(diagnostic_json, indent=2)}\n[END DIAGNOSTIC RESULTS]\n"
                        logger.info("Tool execution: diagnostics successfully run on live sensor telemetry")
                    except Exception as exc:
                        logger.warning("Failed to run diagnostics on live sensor telemetry: %s", exc)
                        diagnostic_context = ""

                    # Re-route based on the newly available live telemetry to toggle DIAGNOSTIC_MODE
                    live_decision = router.route(user_query, context_block, live_telemetry)
                    system_instruction = live_decision.template.system_instruction

        # ── Branch: execute_moisture_prediction ───────────────────────────
        elif called_fn == "execute_moisture_prediction":
            tool_result_content = _run_moisture_prediction(node_id)

        # ── Unknown tool — log and return a safe error message ────────────
        else:
            logger.warning("Unknown tool called by model: %s", called_fn)
            tool_result_content = json.dumps({
                "status": "error",
                "message": f"Unknown tool '{called_fn}'. No handler registered."
            })

        # Append exact conversation history for OpenAI tool strictness
        messages[0]["content"] = system_instruction
        messages[1]["content"] = user_content + diagnostic_context

        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": tool_result_content,
        })

        # Second trip to get the final answer with telemetry included (or offline gracefully)
        t_start2 = time.perf_counter()
        try:
            response2 = _call_groq_with_retry(
                client,
                model_name,
                messages,
                tools=tools,
                tool_choice="none",
                temperature=temperature,
                max_tokens=max_output_tokens,
                top_p=DEFAULT_TOP_P,
            )
        except Exception as exc:
            logger.error("Second Groq API call (after tool execution) failed: %s. Returning base response.", exc)
            answer_text = message.content or "(Unable to generate response due to API error.)"
        else:
            elapsed += time.perf_counter() - t_start2
            answer_text = response2.choices[0].message.content
    else:
        answer_text = message.content

    if not answer_text:
        logger.warning("Groq returned no usable text; using fallback.")
        answer_text = "(No response generated. Please try rephrasing your question.)"

    # ── LLM Refinement Pass ────────────────────────────────────────────
    # Send the raw answer through a lightweight LLM to clean up formatting
    # and produce a polished, conversational output.
    t_refine = time.perf_counter()
    answer_text = _refine_with_llm(client, answer_text)
    elapsed += time.perf_counter() - t_refine
    # ──────────────────────────────────────────────────────────────────

    logger.info("RAG response generated in %.3fs | Model: %s | Context truncated: %s", elapsed, model_name, was_truncated)

    return RAGResponse(
        answer=answer_text,
        sources=sources,
        chunks_used=len(qualifying_chunks),
        chunks_above_threshold=len(qualifying_chunks),
        generation_time_seconds=round(elapsed, 3),
        model_name=model_name,
        grounded=bool(qualifying_chunks) and not was_truncated,  # Grounded only if had context and wasn't heavily truncated
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
    """
    DEPRECATED: Use prompt_router to select system instructions based on intent.
    
    This function is kept for backward compatibility only.
    New code should use prompt_router.get_router().route(query).template.system_instruction
    """
    return prompt_router.get_router().templates[prompt_router.ResponseMode.GENERAL].system_instruction


def _filter_conversation_history(
    history: list[dict[str, str]] | None,
    memory_preference: str,
    query: str,
) -> list[dict[str, str]] | None:
    """
    Filter conversation history to only include items directly relevant to the query.
    
    Args:
        history: Full conversation history.
        memory_preference: "minimal", "selective", or "full".
        query: Current user query.
    """
    if not history:
        return None
        
    if memory_preference == "minimal":
        return None
        
    if memory_preference == "full":
        return history

    # selective: keep history only if relevant
    # 1. Standardize query keywords (remove common agronomic/English stopwords)
    stop_words = {
        "what", "how", "why", "where", "when", "who", "which", "whose", "whom",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "the", "a", "an", "and", "or", "but", "if", "then",
        "else", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "can", "will", "just", "should", "now", "soil",
        "doctor", "plant", "crop", "grow", "farming", "advisor", "agronomist"
    }
    
    query_words = [w.strip("?,.:;!") for w in query.lower().split()]
    keywords = {w for w in query_words if len(w) > 3 and w not in stop_words}
    
    # 2. Check for contextual pronouns or references in query
    contextual_cues = {
        "this", "that", "these", "those", "previously", "before", "as mentioned",
        "above", "earlier", "result", "results", "diagnosis", "recommendation"
    }
    has_contextual_cues = any(w in query_words for w in contextual_cues)
    
    # Check for personal/it pronouns, but don't count if this is a weather query (impersonal "it")
    has_it_pronoun = any(w in query_words for w in ["it", "they", "them", "he", "she"])
    is_weather_query = any(w in query_words for w in ["rain", "weather", "forecast", "monsoon", "climate", "temperature"])
    if has_it_pronoun and not is_weather_query:
        has_contextual_cues = True
    
    # If no contextual pronouns and no keywords, history is likely not relevant
    if not has_contextual_cues and not keywords:
        logger.debug("No contextual cues or keywords in query; excluding conversation memory.")
        return None
        
    # We filter history to keep exchanges that match keyword overlaps
    filtered = []
    # Loop over exchanges (user + assistant pairs) from most recent back
    # Keep up to last 3 exchanges (6 messages max)
    recent = history[-6:]
    
    for msg in recent:
        content = msg.get("content", "").lower()
        if has_contextual_cues or any(kw in content for kw in keywords):
            filtered.append(msg)
            
    if not filtered:
        logger.debug("No semantic overlap found in history; excluding conversation memory.")
        return None
        
    logger.debug("Selective memory active: keeping %d of %d history messages.", len(filtered), len(history))
    return filtered

def _build_user_content(
    query: str,
    context_block: str,
    has_context: bool,
    conversation_history: list[dict[str, str]] | None = None,
    memory_preference: str = "selective",
    response_mode: str = "structured",
) -> str:
    """
    Build user message content with adaptive memory and context inclusion.
    
    Args:
        query: User question.
        context_block: RAG context from knowledge base.
        has_context: Whether context was retrieved.
        conversation_history: Full conversation history.
        memory_preference: "minimal", "selective", or "full".
        response_mode: "natural", "structured", etc.
    
    Returns:
        Formatted user message content.
    """
    # Filter history based on preference (fix history NameError to conversation_history)
    filtered_history = _filter_conversation_history(conversation_history, memory_preference, query)
    
    if not has_context:
        context_note = "⚠️ NOTE: The retrieval system returned no knowledge base chunks. However, use your expert agronomic knowledge to answer."
    else:
        context_note = "The following context chunks have been retrieved from the agronomic knowledge base. Use this information to answer the question."

    # Build history section only if we have filtered history
    history_section = ""
    if filtered_history:
        history_lines = []
        for msg in filtered_history:
            role_label = "Farmer" if msg.get("role") == "user" else "Soil Doctor"
            history_lines.append(f"{role_label}: {msg.get('content', '').strip()}")
        history_section = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION MEMORY (Relevant Context)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(history_lines)}

"""

    return f"""{history_section}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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