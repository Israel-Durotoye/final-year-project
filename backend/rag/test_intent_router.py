"""
test_intent_router.py — Unit Tests for Soil Doctor Intent Classification & Routing

Verifies rule-based keyword matching, confidence scores, ResponseMode routing,
conditional diagnostic fallbacks, and conversation history filtering.
"""

import unittest
from typing import Any

from backend.rag.intent_classifier import Intent, IntentClassifier, get_classifier
from backend.rag.prompt_router import ResponseMode, PromptRouter, get_router
from backend.rag.chat_llm import _filter_conversation_history


class TestIntentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = get_classifier()

    def test_all_intents_detected(self):
        test_cases = [
            ("Explain the basics of soil science.", Intent.GENERAL_KNOWLEDGE),
            ("What can I plant alongside maize to maximize yield?", Intent.INTERCROPPING),
            ("Which crop variety is best suited for sandy soil?", Intent.CROP_SELECTION),
            ("How do I diagnose heavy clay soil conditions?", Intent.SOIL_DIAGNOSIS),
            ("What does the sensor telemetry pH: 5.2 reading mean?", Intent.SENSOR_ANALYSIS),
            ("What NPK dosage and application rate should I use?", Intent.FERTILIZER_RECOMMENDATION),
            ("Why are my potato plant leaves yellow with spots and wilting?", Intent.DISEASE_DIAGNOSIS),
            ("How can I increase the production rate and maximize crop yield?", Intent.YIELD_OPTIMIZATION),
            ("What drip irrigation watering schedule is best for drought?", Intent.IRRIGATION),
            ("What is the monsoon weather forecast for this season?", Intent.WEATHER_ADVICE),
        ]

        for query, expected_intent in test_cases:
            with self.subTest(query=query):
                result = self.classifier.classify(query)
                self.assertEqual(result.intent, expected_intent)
                self.assertGreaterEqual(result.confidence, 0.5)

    def test_output_dictionary_format(self):
        query = "What companion crops go well with corn?"
        result = self.classifier.classify(query)
        res_dict = result.to_dict()

        self.assertIn("intent", res_dict)
        self.assertIn("confidence", res_dict)
        self.assertEqual(res_dict["intent"], "intercropping")
        self.assertTrue(0.0 <= res_dict["confidence"] <= 1.0)

    def test_empty_query_fallback(self):
        result = self.classifier.classify("")
        self.assertEqual(result.intent, Intent.GENERAL_KNOWLEDGE)
        self.assertLessEqual(result.confidence, 0.5)


class TestPromptRouter(unittest.TestCase):
    def setUp(self):
        self.router = get_router()

    def test_routing_to_general_and_intercropping(self):
        # General knowledge
        decision = self.router.route("What is nitrogen?")
        self.assertEqual(decision.mode, ResponseMode.GENERAL_MODE)
        self.assertFalse(decision.template.requires_diagnostics)

        # Companion planting
        decision = self.router.route("What should I plant alongside maize?")
        self.assertEqual(decision.mode, ResponseMode.INTERCROPPING_MODE)
        self.assertFalse(decision.template.requires_diagnostics)

    def test_diagnostic_mode_activation_criteria(self):
        # Case 1: Soil diagnosis query with soil measurements provided
        decision = self.router.route("My soil pH is 5.2 and nitrogen is 45 ppm")
        self.assertEqual(decision.mode, ResponseMode.DIAGNOSTIC_MODE)
        self.assertTrue(decision.template.requires_diagnostics)

        # Case 2: Disease/pest query with symptoms reported
        decision = self.router.route("My plant is sick with yellow spots and wilting leaves")
        self.assertEqual(decision.mode, ResponseMode.DIAGNOSTIC_MODE)
        self.assertTrue(decision.template.requires_diagnostics)

        # Case 3: Telemetry parameters provided explicitly
        telemetry = {"ph": 6.2, "moisture": 25.0, "nitrogen": 30.0}
        decision = self.router.route("What is my soil status?", telemetry=telemetry)
        self.assertEqual(decision.mode, ResponseMode.DIAGNOSTIC_MODE)
        self.assertTrue(decision.template.requires_diagnostics)

        # Case 4: Diagnostic intent but NO telemetry, measurements, symptoms, or sensor values
        decision = self.router.route("How do you diagnose soil?")
        # Should fallback to GENERAL_MODE since no real-time telemetry or measurements are provided
        self.assertEqual(decision.mode, ResponseMode.GENERAL_MODE)
        self.assertFalse(decision.template.requires_diagnostics)
        self.assertIn("Downgraded to GENERAL_MODE", decision.explanation)


class TestMemoryRelevanceFiltering(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"role": "user", "content": "I want to plant maize in my field."},
            {"role": "assistant", "content": "Maize is a great crop. Loam soil is perfect for maize."},
        ]

    def test_memory_excluded_when_unrelated(self):
        # Unrelated query about rain, no maize/planting keywords or relative pronouns
        filtered = _filter_conversation_history(self.history, "selective", "Is it going to rain tomorrow?")
        self.assertIsNone(filtered)

    def test_memory_included_on_keyword_overlap(self):
        # Overlaps with keyword "maize"
        filtered = _filter_conversation_history(self.history, "selective", "When should I harvest my maize?")
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered), 2)

    def test_memory_included_on_contextual_cues(self):
        # Contains pronoun "it" which requires previous turn context
        filtered = _filter_conversation_history(self.history, "selective", "What about spacing details for it?")
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
