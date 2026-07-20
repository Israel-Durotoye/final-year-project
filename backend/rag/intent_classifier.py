"""
intent_classifier.py — Multi-Intent Classification for Smart Response Routing

Classifies user queries into agronomic intent categories to enable
context-aware response generation. Uses rule-based keyword matching
with confidence scoring and extensible keyword dictionaries.

Intent Categories:
    - general_knowledge: Factual questions about farming, crops, soil
    - intercropping: Companion planting, crop combinations
    - crop_selection: Which crops to grow, suitability
    - soil_diagnosis: Analysis of soil condition from description
    - sensor_analysis: Interpretation of sensor telemetry data
    - fertilizer_recommendation: Nutrient recommendations
    - disease_diagnosis: Plant disease or pest identification
    - yield_optimization: Improving yields, productivity
    - irrigation: Water management, scheduling
    - weather_advice: Climate, forecasts, seasonal planning
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent Enumeration
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    """Supported agronomic intent categories."""
    GENERAL_KNOWLEDGE = "general_knowledge"
    INTERCROPPING = "intercropping"
    CROP_SELECTION = "crop_selection"
    SOIL_DIAGNOSIS = "soil_diagnosis"
    SENSOR_ANALYSIS = "sensor_analysis"
    FERTILIZER_RECOMMENDATION = "fertilizer_recommendation"
    DISEASE_DIAGNOSIS = "disease_diagnosis"
    YIELD_OPTIMIZATION = "yield_optimization"
    IRRIGATION = "irrigation"
    WEATHER_ADVICE = "weather_advice"


# ---------------------------------------------------------------------------
# Classification Output Models
# ---------------------------------------------------------------------------

@dataclass
class IntentClassification:
    """Result of intent classification."""
    intent: Intent
    confidence: float  # 0.0 to 1.0
    keywords_matched: list[str] = field(default_factory=list)
    explanation: str = ""
    secondary_intent: Optional[Intent] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert classification result to requested dictionary format."""
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 2)
        }


# ---------------------------------------------------------------------------
# Intent Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Classifies user queries into agronomic intent categories.
    
    Uses rule-based keyword matching with confidence scoring.
    Easily extensible through keyword dictionary updates.
    
    Usage:
        classifier = IntentClassifier()
        result = classifier.classify("What should I plant with maize?")
        # IntentClassification(intent=INTERCROPPING, confidence=1.0, ...)
    """

    def __init__(self):
        """Initialize classifier with keyword dictionaries."""
        self.keywords = self._build_keyword_dictionary()

    def classify(self, query: str) -> IntentClassification:
        """
        Classify user query intent.
        
        Args:
            query: User question or statement.
        
        Returns:
            IntentClassification with intent, confidence, and matched keywords.
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to classifier: %s", query)
            return IntentClassification(
                intent=Intent.GENERAL_KNOWLEDGE,
                confidence=0.5,
                explanation="Query is empty or invalid; defaulting to general knowledge."
            )

        query_lower = query.lower()
        scores = {}
        matched_per_intent = {}

        # Score each intent using weighted keyword list
        for intent, kw_list in self.keywords.items():
            matched = []
            matched_weight_sum = 0.0

            for keyword, weight in kw_list:
                # Case-insensitive substring match
                if keyword.lower() in query_lower:
                    matched.append(keyword)
                    matched_weight_sum += weight

            if matched:
                # Map matched weight sum to confidence (sum of 6.0+ gives close to 1.0 confidence)
                confidence = min(0.3 + 0.7 * (matched_weight_sum / 6.0), 1.0)
                scores[intent] = confidence
                matched_per_intent[intent] = matched

        # Determine primary intent (highest score)
        if not scores:
            logger.debug("No keywords matched for query: '%s'", query[:60])
            return IntentClassification(
                intent=Intent.GENERAL_KNOWLEDGE,
                confidence=0.3,
                explanation="No specific keywords detected; defaulting to general knowledge."
            )

        primary_intent = max(scores, key=scores.get)
        primary_confidence = scores[primary_intent]

        # Identify secondary intent if confidence gap is small
        secondary_intent = None
        scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(scores_sorted) > 1:
            second_score = scores_sorted[1][1]
            # If second-place is close (within 15%), flag as secondary intent
            if second_score > primary_confidence - 0.15:
                secondary_intent = scores_sorted[1][0]

        explanation = self._build_explanation(
            primary_intent,
            primary_confidence,
            matched_per_intent.get(primary_intent, [])
        )

        result = IntentClassification(
            intent=primary_intent,
            confidence=primary_confidence,
            keywords_matched=matched_per_intent.get(primary_intent, []),
            explanation=explanation,
            secondary_intent=secondary_intent,
        )

        logger.debug(
            "Classified query as %s (confidence=%.2f) | Keywords: %s",
            primary_intent.value, primary_confidence, result.keywords_matched
        )

        return result

    # ──────────────────────────────────────────────────────────────────
    # Private: Keyword Dictionary
    # ──────────────────────────────────────────────────────────────────

    def _build_keyword_dictionary(self) -> dict[Intent, list[tuple[str, float]]]:
        """
        Build extensible keyword dictionary for each intent with weights.
        
        Keywords are ordered by specificity. Weights range from 1.0 to 5.0.
        """
        return {
            Intent.GENERAL_KNOWLEDGE: [
                ("what is", 1.0), ("how does", 1.0), ("explain", 1.0), ("define", 1.5), 
                ("tell me about", 1.0), ("describe", 1.0), ("what are", 1.0), ("basics", 1.0),
                ("introduction", 1.0), ("overview", 1.0), ("farming practices", 2.0),
                ("agriculture fundamentals", 3.0), ("soil science", 3.0),
            ],

            Intent.INTERCROPPING: [
                ("plant alongside", 4.0), ("companion planting", 5.0), ("intercrop", 5.0),
                ("plant together", 4.0), ("with maize", 3.0), ("with corn", 3.0),
                ("plant with", 2.0), ("compatible crops", 4.0), ("crop combination", 4.5),
                ("mixed cropping", 5.0), ("polyculture", 5.0), ("associate", 2.0),
                ("alongside", 2.0), ("rotate", 2.0), ("succession planting", 4.0),
                ("growing together", 3.0), ("partner crop", 4.0),
            ],

            Intent.CROP_SELECTION: [
                ("what should i plant", 3.5), ("best crop", 3.0), ("suitable crop", 3.5),
                ("grow", 1.0), ("what to plant", 3.0), ("crop for", 2.0), ("which crop", 2.5),
                ("plant for", 2.0), ("growing season", 2.5), ("crop choice", 3.5),
                ("cultivar selection", 4.5), ("variety recommendation", 4.5),
                ("best variety", 3.5), ("should i grow", 3.0), ("plant in this soil", 3.0),
                ("variety", 3.0), ("suited for", 3.0), ("best suited", 3.5),
            ],

            Intent.SOIL_DIAGNOSIS: [
                ("soil condition", 3.5), ("soil is", 2.0), ("my soil", 2.0),
                ("soil problem", 3.5), ("soil issue", 3.5), ("what's wrong with", 2.5),
                ("diagnose soil", 5.0), ("soil analysis", 4.0), ("acidic soil", 3.5),
                ("alkaline soil", 3.5), ("poor soil", 2.5), ("soil quality", 3.0),
                ("heavy clay", 3.0), ("sandy soil", 3.0), ("compacted soil", 3.5),
                ("waterlogged", 3.0), ("soil health", 3.0),
            ],

            Intent.SENSOR_ANALYSIS: [
                ("sensor", 4.0), ("telemetry", 5.0), ("reading", 2.5), ("ph:", 4.0),
                ("nitrogen:", 4.0), ("phosphorus:", 4.0), ("potassium:", 4.0),
                ("moisture:", 4.0), ("temperature:", 3.5), ("salinity:", 4.0),
                ("ec:", 4.0), ("ppm", 3.0), ("percent", 1.5), ("degree", 1.5),
                ("data from sensor", 4.5), ("sensor shows", 4.5), ("my sensor says", 4.5),
                ("interpret data", 4.0), ("node_id", 4.5), ("hardware node", 4.5),
            ],

            Intent.FERTILIZER_RECOMMENDATION: [
                ("fertilizer", 4.5), ("fertilize", 4.0), ("nutrient", 3.0),
                ("nitrogen", 2.0), ("phosphorus", 2.0), ("potassium", 2.0),
                ("npk", 5.0), ("manure", 3.5), ("compost", 3.0), ("amendment", 3.5),
                ("deficiency", 3.0), ("feed", 1.5), ("boost", 1.5), ("enhance growth", 2.0),
                ("nutrient boost", 3.5), ("application rate", 4.0), ("dosage", 3.5),
                ("chemical fertilizer", 4.0), ("organic fertilizer", 4.0),
            ],

            Intent.DISEASE_DIAGNOSIS: [
                ("disease", 4.0), ("pest", 4.0), ("infection", 3.5), ("sick plant", 4.0),
                ("leaves yellow", 3.5), ("spots on leaves", 4.0), ("wilting", 3.0),
                ("blight", 4.5), ("rust", 3.5), ("mold", 3.5), ("insect", 3.0),
                ("bug", 2.0), ("infestation", 4.0), ("fungal", 4.0), ("bacterial", 4.0),
                ("viral", 4.0), ("symptom", 4.0), ("illness", 3.5), ("damage pattern", 4.0),
                ("powdery mildew", 5.0), ("aphid", 4.5), ("caterpillar", 4.0),
            ],

            Intent.YIELD_OPTIMIZATION: [
                ("yield", 4.0), ("productivity", 3.5), ("increase yield", 4.5),
                ("improve yield", 4.5), ("maximize production", 4.5), ("higher yield", 4.0),
                ("better harvest", 3.5), ("optimize", 2.5), ("efficiency", 2.0),
                ("production rate", 3.0), ("yield per hectare", 4.5), ("crop performance", 3.5),
                ("growth rate", 2.5), ("maximize", 2.0),
            ],

            Intent.IRRIGATION: [
                ("irrigation", 5.0), ("water", 1.5), ("watering", 2.0),
                ("drought", 2.5), ("drainage", 3.0), ("flooding", 2.5),
                ("water schedule", 4.0), ("irrigation schedule", 5.0), ("how often water", 4.0),
                ("water frequency", 4.0), ("water depth", 3.5), ("field capacity", 4.5),
                ("wilting point", 4.5), ("sprinkler", 3.5), ("drip system", 4.5),
            ],

            Intent.WEATHER_ADVICE: [
                ("weather", 4.0), ("rain", 2.0), ("rainfall", 2.5), ("forecast", 4.0),
                ("season", 2.0), ("climate", 3.0), ("frost", 3.5), ("monsoon", 4.5),
                ("wind", 2.0), ("humidity", 2.5), ("sunny", 2.0), ("cloudy", 2.0),
                ("seasonal", 2.5), ("climate suitability", 4.0), ("growing period", 3.0),
                ("frost risk", 4.5),
            ],
        }

    # ──────────────────────────────────────────────────────────────────
    # Private: Explanation Generation
    # ──────────────────────────────────────────────────────────────────

    def _build_explanation(
        self,
        intent: Intent,
        confidence: float,
        matched_keywords: list[str]
    ) -> str:
        """Generate human-readable explanation of classification."""
        confidence_level = "high" if confidence > 0.75 else "moderate" if confidence > 0.5 else "low"
        keywords_str = ", ".join(matched_keywords[:3])  # Top 3 keywords
        
        return f"{intent.value} ({confidence_level} confidence) — matched: {keywords_str}"

    def add_keywords(self, intent: Intent, keywords: list[str] | list[tuple[str, float]]) -> None:
        """
        Add keywords to an intent (extensibility).
        
        Args:
            intent: Target intent category.
            keywords: List of keywords (strings or tuples with weight) to add.
        """
        if intent in self.keywords:
            for kw in keywords:
                if isinstance(kw, tuple):
                    self.keywords[intent].append(kw)
                else:
                    self.keywords[intent].append((kw, 2.0))
            logger.info("Added %d keywords to %s", len(keywords), intent.value)
        else:
            logger.warning("Unknown intent: %s", intent)

    def update_keywords(self, intent: Intent, keywords: list[str] | list[tuple[str, float]]) -> None:
        """
        Replace keywords for an intent (extensibility).
        
        Args:
            intent: Target intent category.
            keywords: New keyword list (strings or tuples with weight).
        """
        if intent in self.keywords:
            new_kws = []
            for kw in keywords:
                if isinstance(kw, tuple):
                    new_kws.append(kw)
                else:
                    new_kws.append((kw, 2.0))
            self.keywords[intent] = new_kws
            logger.info("Updated keywords for %s", intent.value)
        else:
            logger.warning("Unknown intent: %s", intent)

    def register_custom_intent(self, intent_name: str, keywords: list[str] | list[tuple[str, float]]) -> None:
        """
        Register a custom intent (advanced extensibility).
        
        Args:
            intent_name: Custom intent identifier.
            keywords: Keywords for this intent.
        
        Note: Requires extension of Intent enum for full integration.
        """
        logger.warning(
            "Custom intent registration requires Intent enum extension. "
            "Recommended: Modify Intent enum in intent_classifier.py"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_classifier() -> IntentClassifier:
    """Get or create a singleton intent classifier."""
    if not hasattr(get_classifier, "_instance"):
        get_classifier._instance = IntentClassifier()
    return get_classifier._instance

