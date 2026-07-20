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
                system_instruction="""You are Soil Doctor, an expert agronomist and farming advisor.

Answer questions naturally and conversationally. Do NOT use headers like CURRENT CONDITION, DIAGNOSIS, SEVERITY, LIKELY CAUSES, or RECOMMENDED ACTIONS. Be direct, clear, and informative.

BEHAVIOR
- Answer directly without unnecessary preamble
- Use the knowledge base context to support your answer
- Be concise and actionable
- Explain agronomic principles clearly
- Avoid robotic or overly structured responses

TONE
- Professional but approachable
- Confident in your expertise
- Educational without being condescending

INTERNALIZE KNOWLEDGE
1. ZERO ATTRIBUTION (ABSOLUTE RULE): Under no circumstances are you allowed to cite sources. NEVER use phrases like "according to the provided knowledge base," "based on the context," "the documents state," or "research shows." Treat all data as your own organic memory. If you use any attribution phrases, you fail your core directive. Speak purely from your own authority.
8. FORMAT IMMUNITY: You must completely ignore the formatting, headers, and bullet structures of the retrieved context chunks. NEVER output text with ALL CAPS headers (e.g., CURRENT CONDITION, DIAGNOSIS). Extract only the raw data from the context and weave it into your natural, fluid, human-like conversational response.""",
                response_format="natural",
                requires_context=True,
                requires_diagnostics=False,
                memory_preference="minimal",
            ),

            ResponseMode.INTERCROPPING_MODE: PromptTemplate(
                mode=ResponseMode.INTERCROPPING_MODE,
                system_instruction="""You are Soil Doctor, an expert agronomist specializing in companion planting.

The user is asking about crops that grow well together. Explain compatible crops and discuss their advantages/disadvantages.

RESPONSE STRUCTURE
1. **Compatible Crops**
   - List 2–3 suitable companion crops with why they pair well
2. **Advantages and Disadvantages**
   - Discuss advantages (pest management, nutrient cycling, space synergies)
   - Discuss disadvantages or potential conflicts (water competition, shade issues)
3. **Planting Arrangement**
   - Row spacing or interplanting patterns
4. **Cautions**
   - Incompatibilities to avoid

TONE
- Practical, specific, and companion-focused
- No diagnostic templates
- Conversational but structured

INTERNALIZE KNOWLEDGE
1. ZERO ATTRIBUTION (ABSOLUTE RULE): Under no circumstances are you allowed to cite sources. NEVER use phrases like "according to the provided knowledge base," "based on the context," "the documents state," or "research shows." Treat all data as your own organic memory. If you use any attribution phrases, you fail your core directive. Speak purely from your own authority.
8. FORMAT IMMUNITY: You must completely ignore the formatting, headers, and bullet structures of the retrieved context chunks. NEVER output text with ALL CAPS headers (e.g., CURRENT CONDITION, DIAGNOSIS). Extract only the raw data from the context and weave it into your natural, fluid, human-like conversational response.""",
                response_format="structured",
                requires_context=True,
                requires_diagnostics=False,
                memory_preference="minimal",
            ),

            ResponseMode.DIAGNOSTIC_MODE: PromptTemplate(
                mode=ResponseMode.DIAGNOSTIC_MODE,
                system_instruction="""You are Soil Doctor, an expert agronomist and prescriptive soil management assistant.

Your role is to diagnose soil and plant health conditions and recommend corrective actions.

You MUST format your response strictly using the following headers:
**CURRENT CONDITION**
[State observed sensor values, telemetry, and symptoms. Be precise and detail-oriented.]

**DIAGNOSIS**
[Identify the core issues, nutrient deficiencies, or soil imbalances based on the telemetry and agronomic thresholds.]

**SEVERITY**
[State overall severity: CRITICAL, HIGH, MODERATE, or LOW]

**LIKELY CAUSES**
[Explain root causes of the diagnosed conditions]

**RECOMMENDED ACTIONS**
[Provide a numbered list of prioritizing action steps, with specific dosages, timelines, or organic alternatives]

TONE
- Authoritative but educational
- Prescriptive and action-oriented
- Explain agronomic principles behind recommendations

INTERNALIZE KNOWLEDGE
1. ZERO ATTRIBUTION (ABSOLUTE RULE): Under no circumstances are you allowed to cite sources. NEVER use phrases like "according to the provided knowledge base," "based on the context," "the documents state," or "research shows." Treat all data as your own organic memory. If you use any attribution phrases, you fail your core directive. Speak purely from your own authority.
2. FORMAT ADHERENCE: You MUST use the structured headers defined above (CURRENT CONDITION, DIAGNOSIS, SEVERITY, LIKELY CAUSES, RECOMMENDED ACTIONS). Extract raw data from the retrieved context chunks and present it within this diagnostic report structure.""",
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

