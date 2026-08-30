"""Acceptance tests for farm-level recommendation prioritization."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")

from backend.rag import chat_llm
from backend.rag.chat_llm import SYSTEM_INSTRUCTION, _build_user_content
from backend.rag.prescriptions import FarmRecommendationPlanner


class FarmRecommendationPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = FarmRecommendationPlanner()

    @staticmethod
    def node(**overrides):
        snapshot = {
            "node_id": "NODE_06",
            "timestamp_utc": "2026-08-30T12:00:00+00:00",
            "currently_planted_crop": "Maize",
            "ai_predicted_ideal_crop": "Maize",
            "season": "Peak Rainy season",
            "nitrogen_mg_kg": 30.0,
            "phosphorus_mg_kg": 20.0,
            "potassium_mg_kg": 150.0,
            "moisture_pct": 50.0,
            "temperature_c": 25.0,
            "humidity_pct": 65.0,
        }
        snapshot.update(overrides)
        return snapshot

    def test_scenario_a_one_critical_sensor_issue(self) -> None:
        brief = self.planner.build_node_brief(self.node(moisture_pct=85.0))

        self.assertEqual([item["domain"] for item in brief["priorities"]], ["excess_water"])
        priority = brief["priorities"][0]
        self.assertEqual(priority["parameters"], ["Moisture"])
        self.assertEqual(len(priority["supporting_steps"]), 3)
        self.assertTrue(any("irrigation" in step.lower() for step in priority["supporting_steps"]))
        self.assertTrue(any("drainage" in step.lower() for step in priority["supporting_steps"]))
        self.assertNotIn("Temperature", brief["internal_parameter_classification"]["critical_problematic"])

    def test_scenario_b_everything_acceptable(self) -> None:
        brief = self.planner.build_node_brief(self.node())

        self.assertTrue(brief["healthy_sensor_state"])
        self.assertEqual(brief["priorities"], [])
        self.assertLessEqual(len(brief["preventive_focus_if_no_actionable_problem"]), 3)
        self.assertIn("Moisture", brief["internal_parameter_classification"]["acceptable_optimal"])

    def test_scenario_c_related_wetness_signals_are_consolidated(self) -> None:
        brief = self.planner.build_node_brief(
            self.node(moisture_pct=85.0, humidity_pct=90.0)
        )

        self.assertEqual(len(brief["priorities"]), 1)
        priority = brief["priorities"][0]
        self.assertEqual(priority["domain"], "excess_water")
        self.assertTrue(any("Humidity" in evidence for evidence in priority["evidence"]))
        self.assertTrue(any("Rainy" in evidence for evidence in priority["evidence"]))
        self.assertTrue(any("disease" in step.lower() for step in priority["supporting_steps"]))

    def test_scenario_d_nutrient_signal_without_context_has_no_rate(self) -> None:
        brief = self.planner.build_node_brief(self.node(nitrogen_mg_kg=3.0))

        self.assertEqual([item["domain"] for item in brief["priorities"]], ["nutrient_verification"])
        priority = brief["priorities"][0]
        self.assertEqual(priority["severity"], "high_investigation")
        self.assertTrue(priority["context_limits"])
        self.assertIn("Nitrogen", brief["internal_parameter_classification"]["unknown_not_enough_context"])
        self.assertNotIn("kg/ha", json.dumps(priority))

    def test_scenario_e_crop_mismatch_remains_strategic(self) -> None:
        brief = self.planner.build_node_brief(
            self.node(ai_predicted_ideal_crop="Cassava")
        )

        self.assertTrue(brief["healthy_sensor_state"])
        self.assertEqual(brief["priorities"], [])
        self.assertEqual(len(brief["strategic_considerations"]), 1)
        self.assertIn("Cassava", brief["strategic_considerations"][0])

    def test_scenario_f_independent_critical_issues_remain_separate(self) -> None:
        brief = self.planner.build_node_brief(
            self.node(moisture_pct=85.0, soil_ph=3.5)
        )

        domains = {item["domain"] for item in brief["priorities"]}
        self.assertEqual(domains, {"excess_water", "soil_reaction"})
        self.assertEqual(len(brief["priorities"]), 2)

    def test_priority_brief_is_injected_without_changing_prompt_inputs(self) -> None:
        node = self.node(moisture_pct=85.0)
        content = _build_user_content(
            "What should I do for NODE_06?",
            "Agronomic drainage guidance.",
            True,
            {"status": "online", "node_count": 1, "nodes": [node]},
        )

        self.assertIn("LIVE FARM SNAPSHOT", content)
        self.assertIn("FARM-LEVEL PRIORITY BRIEF", content)
        self.assertIn('"domain": "excess_water"', content)
        self.assertIn("KNOWLEDGE-BASE CONTEXT", content)
        self.assertIn("CURRENT USER QUESTION", content)

    def test_system_instruction_enforces_farm_level_presentation(self) -> None:
        lowered = SYSTEM_INSTRUCTION.lower()
        self.assertIn("do not explain", lowered)
        self.assertIn("merge related observations", lowered)
        self.assertIn("one main priority", lowered)
        self.assertIn("do not invent a fertiliser", lowered)

    def test_generation_flow_and_rag_response_contract_are_preserved(self) -> None:
        node = self.node()
        chunk = SimpleNamespace(
            text="Maintain drainage during the rainy season.",
            source="Agronomic guide",
            page=12,
            rerank_score=1.0,
        )

        def fake_llm_call(_client, _model, messages, **_kwargs):
            self.assertIn("FARM-LEVEL PRIORITY BRIEF", messages[1]["content"])
            return "NODE_06 is generally healthy. No major sensor-based correction is required."

        with (
            patch.object(
                chat_llm,
                "_get_farm_snapshot",
                return_value={"status": "online", "node_count": 1, "nodes": [node]},
            ),
            patch.object(chat_llm, "_resolve_api_key", return_value=("test-key", "test")),
            patch.object(chat_llm, "_create_agentrouter_client", return_value=object()),
            patch.object(chat_llm, "_call_llm_with_retry", side_effect=fake_llm_call),
        ):
            response = chat_llm.generate_rag_response(
                "What should I do for NODE_06?",
                [chunk],
            )

        self.assertIsInstance(response, chat_llm.RAGResponse)
        self.assertEqual(response.sources, ["Agronomic guide"])
        self.assertEqual(response.chunks_used, 1)
        self.assertTrue(response.grounded)


if __name__ == "__main__":
    unittest.main()
