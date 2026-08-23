"""
prompt_router.py — Dynamic Prompt Selection Based on Intent

Routes user queries to appropriate system prompts and response modes.
Eliminates the one-size-fits-all diagnostic template by selecting
context-appropriate instructions based on detected intent.

Response Modes:
    - GENERAL_MODE: Natural, conversational responses
    - INTERCROPPING_MODE: Companion planting advice with pros/cons
    - DIAGNOSTIC_MODE: Structured soil diagnosis (CURRENT CONDITION, DIAGNOSIS, etc.)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

from backend.rag import intent_classifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response Mode Enumeration
# ---------------------------------------------------------------------------

class ResponseMode(str, Enum):
    """Response generation modes mapped to intents."""
    GENERAL_MODE = "general_mode"
    INTERCROPPING_MODE = "intercropping_mode"
    DIAGNOSTIC_MODE = "diagnostic_mode"

    # Aliases for backward compatibility
    GENERAL = "general_mode"
    INTERCROPPING = "intercropping_mode"
    DIAGNOSTIC = "diagnostic_mode"


# ---------------------------------------------------------------------------
# Prompt Template Models
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """System prompt template for a response mode."""
    mode: ResponseMode
    system_instruction: str
    response_format: str  # E.g., "natural", "structured", "comparative"
    requires_context: bool  # Whether RAG context should be included
    requires_diagnostics: bool  # Whether soil diagnostics should be generated
    memory_preference: str  # "minimal", "selective", "full"


@dataclass
class RoutingDecision:
    """Result of prompt routing decision."""
    mode: ResponseMode
    template: PromptTemplate
    intent: intent_classifier.Intent
    confidence: float
    secondary_intent: Optional[intent_classifier.Intent]
    explanation: str


# ---------------------------------------------------------------------------
# Prompt Router
# ---------------------------------------------------------------------------

class PromptRouter:
    """
    Routes queries to appropriate response modes and system prompts.
    
    Usage:
        router = PromptRouter()
        decision = router.route("What crops go well with maize?")
        # Returns: RoutingDecision with INTERCROPPING_MODE and appropriate prompt
    """

    def __init__(self):
        """Initialize router with intent classifier and prompt templates."""
        self.classifier = intent_classifier.get_classifier()
        self.templates = self._build_prompt_templates()
        self.intent_to_mode = self._build_intent_mode_mapping()
        self.intent_instructions = self._build_intent_instructions()

    def evaluate_diagnostic_activation(
        self,
        query: str,
        context: str = "",
        telemetry: Optional[dict[str, Any]] = None
    ) -> dict[str, bool]:
        """
        Evaluate if diagnostic criteria are met:
        - telemetry exists: live sensor telemetry structure is present or mentioned.
        - sensor values exist: specific numeric values are in query or telemetry.
        - symptoms are reported: keywords indicating yellowing, spots, pests.
        - soil measurements are provided: explicit readings (pH, moisture, NPK).
        """
        q_lower = query.lower()
        c_lower = context.lower()
        combined = f"{q_lower} {c_lower}"

        # 1. Telemetry exists
        telemetry_exists = (
            (telemetry is not None and len(telemetry) > 0) or
            any(k in combined for k in ["node_", "telemetry", "sensor node", "live data"])
        )

        # 2. Sensor values exist
        sensor_values_exist = (
            (telemetry is not None and any(k in telemetry for k in ["ph", "nitrogen", "moisture", "temperature"])) or
            any(k in combined for k in ["ph:", "nitrogen:", "phosphorus:", "potassium:", "moisture:", "temp:"])
        )

        # 3. Symptoms are reported
        symptom_keywords = [
            "yellow", "wilt", "spot", "blight", "pest", "rot", "insect", "bug",
            "mold", "fungus", "bacterial", "infestation", "sick", "damage", "rust",
            "dying", "pale", "stunt", "chewed", "curl", "lesion", "spots"
        ]
        symptoms_reported = any(k in combined for k in symptom_keywords)

        # 4. Soil measurements are provided
        measurement_patterns = [
            r"ph\s*(?:of|is|:)?\s*[\d.]+",
            r"nitrogen\s*(?:of|is|:)?\s*[\d.]+",
            r"phosphorus\s*(?:of|is|:)?\s*[\d.]+",
            r"potassium\s*(?:of|is|:)?\s*[\d.]+",
            r"moisture\s*(?:of|is|:)?\s*[\d.]+\s*%",
            r"[\d.]+\s*(?:ppm|%|celsius|°c|ds/m)"
        ]
        soil_measurements_provided = (
            (telemetry is not None and len(telemetry) >= 3) or
            any(re.search(pat, combined) is not None for pat in measurement_patterns)
        )

        return {
            "telemetry_exists": telemetry_exists,
            "sensor_values_exist": sensor_values_exist,
            "symptoms_reported": symptoms_reported,
            "soil_measurements_provided": soil_measurements_provided
        }

    def route(
        self,
        query: str,
        context: str = "",
        telemetry: Optional[dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route a user query to appropriate response mode.
        
        Args:
            query: User question or statement.
            context: Context block from RAG.
            telemetry: Extracted telemetry parameters.
        
        Returns:
            RoutingDecision with mode, template, and explanation.
        """
        # Classify the query
        classification = self.classifier.classify(query)
        intent = classification.intent
        
        # Map intent to response mode
        mode = self.intent_to_mode.get(intent, ResponseMode.GENERAL_MODE)
        
        # Check criteria for DIAGNOSTIC_MODE activation
        criteria = self.evaluate_diagnostic_activation(query, context, telemetry)
        activation_met = any(criteria.values())
        
        # If routed to DIAGNOSTIC_MODE but activation criteria not met, downgrade to GENERAL_MODE
        original_mode = mode
        downgraded = False
        if mode == ResponseMode.DIAGNOSTIC_MODE and not activation_met:
            mode = ResponseMode.GENERAL_MODE
            downgraded = True
            
        template = self.templates[mode]
        
        # Refactor the prompt builder: get topic instructions dynamically based on intent
        intent_instruction = self.intent_instructions.get(intent, "")
        
        # Combine base instruction and specific intent instruction
        combined_instruction = f"{template.system_instruction}\n\n[SPECIFIC INTENT INSTRUCTION]\n{intent_instruction}"
        
        explanation = f"Intent: {intent.value} (conf={classification.confidence:.2f}) → Mode: {mode.value}"
        if downgraded:
            explanation += " (Downgraded to GENERAL_MODE due to missing diagnostic criteria)"
        
        decision = RoutingDecision(
            mode=mode,
            template=PromptTemplate(
                mode=template.mode,
                system_instruction=combined_instruction,
                response_format=template.response_format,
                requires_context=template.requires_context,
                requires_diagnostics=template.requires_diagnostics and not downgraded,
                memory_preference=template.memory_preference,
            ),
            intent=intent,
            confidence=classification.confidence,
            secondary_intent=classification.secondary_intent,
            explanation=explanation,
        )
        
        logger.debug("Routing decision: %s", decision.explanation)
        return decision

    # ──────────────────────────────────────────────────────────────────
    # Private: Intent-to-Mode Mapping
    # ──────────────────────────────────────────────────────────────────

    def _build_intent_mode_mapping(self) -> dict[intent_classifier.Intent, ResponseMode]:
        """Map intent categories to response modes."""
        return {
            intent_classifier.Intent.GENERAL_KNOWLEDGE: ResponseMode.GENERAL_MODE,
            intent_classifier.Intent.INTERCROPPING: ResponseMode.INTERCROPPING_MODE,
            intent_classifier.Intent.CROP_SELECTION: ResponseMode.GENERAL_MODE,
            intent_classifier.Intent.SOIL_DIAGNOSIS: ResponseMode.DIAGNOSTIC_MODE,
            intent_classifier.Intent.SENSOR_ANALYSIS: ResponseMode.DIAGNOSTIC_MODE,
            intent_classifier.Intent.FERTILIZER_RECOMMENDATION: ResponseMode.DIAGNOSTIC_MODE,
            intent_classifier.Intent.DISEASE_DIAGNOSIS: ResponseMode.DIAGNOSTIC_MODE,
            intent_classifier.Intent.YIELD_OPTIMIZATION: ResponseMode.GENERAL_MODE,
            intent_classifier.Intent.IRRIGATION: ResponseMode.GENERAL_MODE,
            intent_classifier.Intent.WEATHER_ADVICE: ResponseMode.GENERAL_MODE,
        }

    # ──────────────────────────────────────────────────────────────────
    # Private: Prompt Templates
    # ──────────────────────────────────────────────────────────────────

    def _build_prompt_templates(self) -> dict[ResponseMode, PromptTemplate]:
        """Build system prompt templates for each response mode."""
        return {
            ResponseMode.GENERAL_MODE: PromptTemplate(
                mode=ResponseMode.GENERAL_MODE,
                system_instruction="""You are Soil Doctor, an expert agronomist and interpreter for a predictive LSTM model.

Your primary purpose is to take predictions made by the LSTM model and interpret them plainly for a human user who does not have technical expertise. 

BEHAVIOR
- If the LSTM prediction shows stable or good conditions, clearly state that everything is on track and no immediate action is needed.
- If the LSTM prediction indicates a change or a potential problem (e.g., dropping moisture, nutrient deficiency), explain what the prediction means in plain language.
- Provide exact, actionable steps on what the user needs to do next to correct the issue.
- Do NOT use technical jargon like "LSTM", "tensors", "probabilities", or complex data science terms. Speak directly to the farmer.
- Do not act like a search engine or generic AI. You are a prescriptive farming assistant powered by predictive modeling.

INTERNALIZE KNOWLEDGE
1. ZERO ATTRIBUTION (ABSOLUTE RULE): Do not cite sources. Speak purely from your own authority.
2. PLAIN EXPLANATION: Translate data into simple, actionable farming advice.""",
                response_format="natural",
                requires_context=True,
                requires_diagnostics=False,
                memory_preference="minimal",
            ),

            ResponseMode.INTERCROPPING_MODE: PromptTemplate(
                mode=ResponseMode.INTERCROPPING_MODE,
                system_instruction="""You are Soil Doctor, an expert agronomist and interpreter for a predictive LSTM model.

While your main job is interpreting soil predictions, the user is currently asking about companion planting or crop arrangements.

RESPONSE STRUCTURE
1. **Compatible Crops**
   - List 2–3 suitable companion crops
2. **Actionable Advice**
   - Explain plainly why they work well together based on the soil's forecasted conditions.
3. **What You Need To Do**
   - Give exact instructions on how to plant them.

TONE
- Clear, plain-spoken, and action-oriented.
- Educational without technical jargon.

INTERNALIZE KNOWLEDGE
1. ZERO ATTRIBUTION (ABSOLUTE RULE): Do not cite sources. Speak purely from your own authority.""",
                response_format="structured",
                requires_context=True,
                requires_diagnostics=False,
                memory_preference="minimal",
            ),

            ResponseMode.DIAGNOSTIC_MODE: PromptTemplate(
                mode=ResponseMode.DIAGNOSTIC_MODE,
                system_instruction="""You are Soil Doctor, an expert agronomist and interpreter for a predictive LSTM model.

Your primary purpose is to take predictions made by the LSTM model and interpret them plainly for a human user. You are providing a diagnostic alert based on the model's forecast.

You MUST format your response strictly using the following headers:
**PREDICTION STATUS**
[State simply if the condition is good, or if a change/problem is forecasted based on the LSTM data.]

**WHAT THIS MEANS**
[Explain the predicted issue in plain, non-technical language. Do not use terms like "LSTM" or "probabilities".]

**SEVERITY**
[State overall severity: NORMAL, LOW, MODERATE, HIGH, or CRITICAL]

**WHAT YOU NEED TO DO**
[Provide exact, actionable steps the user must take right now to address the forecasted issue.]

TONE
- Clear, plain-spoken, and action-oriented.
- Educational without technical jargon.

INTERNALIZE KNOWLEDGE
1. ZERO ATTRIBUTION (ABSOLUTE RULE): Do not cite sources. Speak purely from your own authority.
2. FORMAT ADHERENCE: You MUST use the structured headers defined above. Translate data into simple, actionable advice.""",
                response_format="structured",
                requires_context=True,
                requires_diagnostics=True,
                memory_preference="selective",
            ),
        }

    def _build_intent_instructions(self) -> dict[intent_classifier.Intent, str]:
        """Build specific prompt instructions for each of the 10 intents."""
        return {
            intent_classifier.Intent.GENERAL_KNOWLEDGE: (
                "Provide general, factual agronomic information. Explain concepts clearly and simply, "
                "focusing on sustainable farming practices."
            ),
            intent_classifier.Intent.INTERCROPPING: (
                "Provide companion planting recommendations. Detail how these crops interact, nutrient sharing, "
                "pest deterrence, and spatial configurations."
            ),
            intent_classifier.Intent.CROP_SELECTION: (
                "Advise on crop choice. Consider climate suitability, soil constraints, planting windows, "
                "and crop rotation history from context."
            ),
            intent_classifier.Intent.SOIL_DIAGNOSIS: (
                "Assess soil health. Look for texture, pH levels, compaction signs, or aeration issues mentioned in context."
            ),
            intent_classifier.Intent.SENSOR_ANALYSIS: (
                "Analyze soil sensor telemetry readings. Explain the meaning of pH, moisture %, NPK ppm, "
                "or EC/salinity numbers."
            ),
            intent_classifier.Intent.FERTILIZER_RECOMMENDATION: (
                "Provide fertilizer recommendations. Focus on nutrient application, organic compost/manure options, "
                "chemical NPK application rates, safety warnings, and application timing."
            ),
            intent_classifier.Intent.DISEASE_DIAGNOSIS: (
                "Diagnose crop diseases or pests. Focus on leaf symptoms, pathogen types (fungal, bacterial, viral), "
                "insect damage patterns, and organic/chemical remedies."
            ),
            intent_classifier.Intent.YIELD_OPTIMIZATION: (
                "Optimize yield and productivity. Provide guidelines on planting density, weed control, pruning, "
                "and efficiency strategies."
            ),
            intent_classifier.Intent.IRRIGATION: (
                "Advise on water management. Recommend watering intervals, drip/sprinkler suitability, moisture monitoring, "
                "and fields drainage improvements."
            ),
            intent_classifier.Intent.WEATHER_ADVICE: (
                "Deliver seasonal and weather advice. Connect crop requirements to local rainfall, temperature risks, "
                "frost protection, or drought preparation."
            ),
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_router() -> PromptRouter:
    """Get or create a singleton prompt router."""
    if not hasattr(get_router, "_instance"):
        get_router._instance = PromptRouter()
    return get_router._instance

