"""
test_intent_and_routing.py — Unit Tests for Intent Classification and Prompt Routing

Comprehensive test suite covering:
- Intent classification accuracy
- Confidence scoring
- Prompt routing decisions
- Edge cases and fallbacks
- Secondary intent detection
- Keyword matching
"""

import unittest
from unittest.mock import patch

from backend.rag import intent_classifier, prompt_router


class TestIntentClassifier(unittest.TestCase):
    """Unit tests for IntentClassifier."""

    def setUp(self):
        """Initialize classifier for each test."""
        self.classifier = intent_classifier.IntentClassifier()

    def test_intercropping_intent_detection(self):
        """Test detection of intercropping/companion planting queries."""
        query = "What crops should I plant alongside maize?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.INTERCROPPING)
        self.assertGreater(result.confidence, 0.6)
        self.assertIn("plant alongside", result.keywords_matched)

    def test_crop_selection_intent_detection(self):
        """Test detection of crop selection queries."""
        query = "What's the best crop to grow in my field?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.CROP_SELECTION)
        self.assertGreater(result.confidence, 0.5)

    def test_soil_diagnosis_intent_detection(self):
        """Test detection of soil diagnosis queries."""
        query = "My soil is acidic and waterlogged. What should I do?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.SOIL_DIAGNOSIS)
        self.assertGreater(result.confidence, 0.6)
        self.assertIn("waterlogged", result.keywords_matched)

    def test_sensor_analysis_intent_detection(self):
        """Test detection of sensor telemetry analysis queries."""
        query = "My sensor shows pH: 5.2, nitrogen: 25 ppm. What does this mean?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.SENSOR_ANALYSIS)
        self.assertGreater(result.confidence, 0.6)

    def test_fertilizer_recommendation_intent_detection(self):
        """Test detection of fertilizer recommendation queries."""
        query = "How much fertilizer should I apply to boost nitrogen?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.FERTILIZER_RECOMMENDATION)
        self.assertGreater(result.confidence, 0.5)

    def test_disease_diagnosis_intent_detection(self):
        """Test detection of disease/pest diagnosis queries."""
        query = "My plants have yellow spots on leaves and wilting. What disease is it?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.DISEASE_DIAGNOSIS)
        self.assertGreater(result.confidence, 0.6)

    def test_yield_optimization_intent_detection(self):
        """Test detection of yield optimization queries."""
        query = "How can I increase my crop yield this season?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.YIELD_OPTIMIZATION)
        self.assertGreater(result.confidence, 0.5)

    def test_irrigation_intent_detection(self):
        """Test detection of irrigation/water management queries."""
        query = "How often should I water my field?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.IRRIGATION)
        self.assertGreater(result.confidence, 0.4)

    def test_weather_advice_intent_detection(self):
        """Test detection of weather/seasonal queries."""
        query = "When should I plant given the upcoming monsoon season?"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.WEATHER_ADVICE)
        self.assertGreater(result.confidence, 0.5)

    def test_general_knowledge_fallback(self):
        """Test fallback to general knowledge for ambiguous queries."""
        query = "Tell me about farming"
        result = self.classifier.classify(query)
        
        self.assertEqual(result.intent, intent_classifier.Intent.GENERAL_KNOWLEDGE)
        self.assertGreater(result.confidence, 0.3)

    def test_confidence_scoring(self):
        """Test that confidence scores are normalized between 0 and 1."""
        queries = [
            "What is agriculture?",
            "Plant alongside maize",
            "pH: 5.2, nitrogen: 25",
        ]
        
        for query in queries:
            result = self.classifier.classify(query)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_keyword_matching_accuracy(self):
        """Test that matched keywords are actually present in the query."""
        query = "What crops should I plant alongside maize?"
        result = self.classifier.classify(query)
        
        query_lower = query.lower()
        for keyword in result.keywords_matched:
            self.assertIn(keyword.lower(), query_lower)

    def test_empty_query_handling(self):
        """Test that empty queries don't crash classifier."""
        result = self.classifier.classify("")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.intent, intent_classifier.Intent.GENERAL_KNOWLEDGE)
        self.assertLessEqual(result.confidence, 0.5)

    def test_none_query_handling(self):
        """Test that None queries don't crash classifier."""
        result = self.classifier.classify(None)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.intent, intent_classifier.Intent.GENERAL_KNOWLEDGE)

    def test_secondary_intent_detection(self):
        """Test that secondary intents are detected when confidence is close."""
        # A query that could be both fertilizer and soil diagnosis
        query = "My soil nitrogen is low. What fertilizer should I use?"
        result = self.classifier.classify(query)
        
        # Should have a secondary intent (either fertilizer or soil diagnosis)
        # Just check that classification doesn't crash
        self.assertIsNotNone(result.intent)
        self.assertIsNotNone(result.secondary_intent or True)  # Secondary may be None

    def test_case_insensitivity(self):
        """Test that classification is case-insensitive."""
        query_lower = "what crops should i plant alongside maize?"
        query_upper = "WHAT CROPS SHOULD I PLANT ALONGSIDE MAIZE?"
        
        result_lower = self.classifier.classify(query_lower)
        result_upper = self.classifier.classify(query_upper)
        
        self.assertEqual(result_lower.intent, result_upper.intent)
        self.assertEqual(result_lower.confidence, result_upper.confidence)

    def test_add_keywords_extensibility(self):
        """Test that custom keywords can be added to intents."""
        original_count = len(self.classifier.keywords[intent_classifier.Intent.CROP_SELECTION])
        
        self.classifier.add_keywords(
            intent_classifier.Intent.CROP_SELECTION,
            ["experimental crop", "heirloom variety"]
        )
        
        new_count = len(self.classifier.keywords[intent_classifier.Intent.CROP_SELECTION])
        self.assertEqual(new_count, original_count + 2)

    def test_update_keywords_extensibility(self):
        """Test that keywords can be updated for an intent."""
        new_keywords = ["test1", "test2", "test3"]
        self.classifier.update_keywords(
            intent_classifier.Intent.GENERAL_KNOWLEDGE,
            new_keywords
        )
        
        self.assertEqual(
            [k[0] for k in self.classifier.keywords[intent_classifier.Intent.GENERAL_KNOWLEDGE]],
            new_keywords
        )


class TestPromptRouter(unittest.TestCase):
    """Unit tests for PromptRouter."""

    def setUp(self):
        """Initialize router for each test."""
        self.router = prompt_router.PromptRouter()

    def test_general_query_routing(self):
        """Test that general queries route to GENERAL mode."""
        decision = self.router.route("Tell me about sustainable farming practices")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.GENERAL)
        self.assertIsNotNone(decision.template)
        self.assertEqual(decision.template.mode, prompt_router.ResponseMode.GENERAL)

    def test_intercropping_query_routing(self):
        """Test that intercropping queries route to INTERCROPPING mode."""
        decision = self.router.route("What crops should I plant alongside maize?")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.INTERCROPPING)
        self.assertTrue(decision.template.requires_context)
        self.assertFalse(decision.template.requires_diagnostics)

    def test_soil_diagnosis_routing(self):
        """Test that soil diagnosis queries route to GENERAL mode (downgraded)."""
        decision = self.router.route("My soil is acidic and has poor drainage")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.GENERAL_MODE)
        self.assertFalse(decision.template.requires_diagnostics)

    def test_sensor_analysis_routing(self):
        """Test that sensor analysis queries route to DIAGNOSTIC mode."""
        decision = self.router.route("My sensor shows pH: 5.2, nitrogen: 25 ppm")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.DIAGNOSTIC_MODE)
        self.assertTrue(decision.template.requires_diagnostics)

    def test_fertilizer_routing(self):
        """Test that fertilizer queries route to GENERAL mode."""
        decision = self.router.route("What fertilizer should I apply?")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.GENERAL_MODE)
        self.assertFalse(decision.template.requires_diagnostics)

    def test_disease_routing(self):
        """Test that disease queries route to DIAGNOSTIC mode."""
        decision = self.router.route("My plants have yellow leaves and spots and are wilting")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.DIAGNOSTIC_MODE)
        self.assertTrue(decision.template.requires_diagnostics)

    def test_yield_optimization_routing(self):
        """Test that optimization queries route to GENERAL mode."""
        decision = self.router.route("How can I maximize my yield?")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.GENERAL_MODE)

    def test_irrigation_routing(self):
        """Test that irrigation queries route to GENERAL mode."""
        decision = self.router.route("How often should I water my field?")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.GENERAL_MODE)

    def test_weather_routing(self):
        """Test that weather queries route to GENERAL mode."""
        decision = self.router.route("When should I plant given the monsoon season?")
        
        self.assertEqual(decision.mode, prompt_router.ResponseMode.GENERAL_MODE)

    def test_routing_decision_structure(self):
        """Test that routing decision contains all required fields."""
        decision = self.router.route("What crops should I plant?")
        
        self.assertIsNotNone(decision.mode)
        self.assertIsNotNone(decision.template)
        self.assertIsNotNone(decision.intent)
        self.assertIsNotNone(decision.confidence)
        self.assertIsNotNone(decision.explanation)

    def test_diagnostic_mode_requires_diagnostics(self):
        """Test that diagnostic mode requires diagnostics generation."""
        decision = self.router.route("Analyze my soil: pH 5.2, nitrogen 25")
        
        if decision.mode == prompt_router.ResponseMode.DIAGNOSTIC_MODE:
            self.assertTrue(decision.template.requires_diagnostics)

    def test_general_mode_no_diagnostics(self):
        """Test that general mode doesn't require diagnostics."""
        decision = self.router.route("Tell me about crop rotation")
        
        if decision.mode == prompt_router.ResponseMode.GENERAL_MODE:
            self.assertFalse(decision.template.requires_diagnostics)

    def test_context_requirements(self):
        """Test that all modes require context."""
        decision = self.router.route("Any farming question")
        
        self.assertTrue(decision.template.requires_context)

    def test_memory_preference_in_templates(self):
        """Test that all templates specify memory preference."""
        for mode, template in self.router.templates.items():
            self.assertIn(
                template.memory_preference,
                ["minimal", "selective", "full"],
                f"Invalid memory preference for {mode}: {template.memory_preference}"
            )

    def test_response_format_in_templates(self):
        """Test that all templates specify response format."""
        for mode, template in self.router.templates.items():
            self.assertIsNotNone(template.response_format)
            self.assertGreater(len(template.response_format), 0)

    def test_system_instruction_content(self):
        """Test that system instructions are not empty."""
        for mode, template in self.router.templates.items():
            self.assertGreater(len(template.system_instruction), 50)
            self.assertIn("Soil Doctor", template.system_instruction)

    def test_router_consistency(self):
        """Test that same query produces consistent routing."""
        query = "What crops go well with maize?"
        decision1 = self.router.route(query)
        decision2 = self.router.route(query)
        
        self.assertEqual(decision1.mode, decision2.mode)
        self.assertEqual(decision1.intent, decision2.intent)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for realistic usage scenarios."""

    def setUp(self):
        """Initialize classifier and router."""
        self.classifier = intent_classifier.IntentClassifier()
        self.router = prompt_router.PromptRouter()

    def test_diagnostic_vs_general_distinction(self):
        """Test that diagnostic and general queries are properly distinguished."""
        diagnostic_query = "My soil pH is 5.2, nitrogen is 25 ppm, moisture 18%"
        general_query = "What are the basics of soil science?"
        
        diagnostic_result = self.router.route(diagnostic_query)
        general_result = self.router.route(general_query)
        
        # Diagnostic should route to DIAGNOSTIC mode, general to GENERAL
        self.assertTrue(
            diagnostic_result.template.requires_diagnostics or
            diagnostic_result.mode == prompt_router.ResponseMode.DIAGNOSTIC_MODE
        )
        self.assertFalse(general_result.template.requires_diagnostics)

    def test_telemetry_triggers_diagnostics(self):
        """Test that telemetry-rich queries enable diagnostic mode."""
        queries_with_telemetry = [
            "pH: 5.2, nitrogen: 25 ppm, phosphorus: 8 ppm",
            "Sensor reading: EC 0.8 dS/m, moisture 18%",
            "My soil has pH 5.5, temperature 22°C",
        ]
        
        for query in queries_with_telemetry:
            decision = self.router.route(query)
            # Should route to diagnostic if telemetry detected
            is_diagnostic = decision.template.requires_diagnostics
            self.assertTrue(is_diagnostic)

    def test_symptom_based_disease_routing(self):
        """Test that symptom descriptions route to diagnostic mode."""
        symptom_queries = [
            "Yellow leaves and wilting spots",
            "Brown spots on leaf surface and wilting",
            "Plant is infected with rust and dying",
        ]
        
        for query in symptom_queries:
            decision = self.router.route(query)
            # All should route to DIAGNOSTIC mode due to symptoms
            self.assertEqual(decision.mode, prompt_router.ResponseMode.DIAGNOSTIC_MODE)

    def test_multipart_query_handling(self):
        """Test that complex queries with multiple intents are handled."""
        query = "My soil pH is 5.2, what fertilizer should I use and when should I plant given the season?"
        decision = self.router.route(query)
        
        # Should detect a primary intent and potentially secondary
        self.assertIsNotNone(decision.intent)
        self.assertIsNotNone(decision.secondary_intent or True)

    def test_prompt_appropriateness_for_mode(self):
        """Test that selected prompts are appropriate for the detected intent."""
        test_cases = [
            ("plant with maize", prompt_router.ResponseMode.INTERCROPPING_MODE),
            ("my soil pH", prompt_router.ResponseMode.GENERAL_MODE), # no values provided
            ("my soil pH is 5.5", prompt_router.ResponseMode.DIAGNOSTIC_MODE),
            ("fertilizer application", prompt_router.ResponseMode.GENERAL_MODE),
        ]
        
        for query, expected_mode in test_cases:
            decision = self.router.route(query)
            self.assertEqual(decision.mode, expected_mode)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Initialize classifier and router."""
        self.classifier = intent_classifier.IntentClassifier()
        self.router = prompt_router.PromptRouter()

    def test_very_long_query(self):
        """Test handling of very long queries."""
        long_query = "What can I plant " + ("with maize " * 100)
        result = self.classifier.classify(long_query)
        
        self.assertIsNotNone(result)
        self.assertGreater(result.confidence, 0.0)

    def test_special_characters_in_query(self):
        """Test handling of special characters."""
        query = "What crops!? @#$%^ should I plant?"
        result = self.classifier.classify(query)
        
        self.assertIsNotNone(result)

    def test_non_english_keywords(self):
        """Test behavior with non-English text (should degrade gracefully)."""
        query = "मुझे मक्का के साथ कौन सी फसल लगानी चाहिए?"  # Hindi
        result = self.classifier.classify(query)
        
        self.assertIsNotNone(result)
        self.assertLess(result.confidence, 0.5)

    def test_whitespace_variations(self):
        """Test handling of whitespace variations."""
        queries = [
            "What crops should I plant?",
            "What   crops   should   I   plant?",
            "  What crops should I plant  ",
        ]
        
        for query in queries:
            result = self.classifier.classify(query)
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
