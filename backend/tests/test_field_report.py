"""Regression coverage for full reports and the offline generation path."""

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from backend.rag import chat_llm
from backend.rag.field_report import (
    build_report_evidence, report_content, usable_local_report, render_evidence_report,
    normalize_field_summary, render_field_summary, EVIDENCE_REPORT_NOTE,
)


def snapshot():
    return {"status": "online", "nodes": [
        {"node_id": "NODE_06", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
         "currently_planted_crop": "Maize", "moisture_pct": 85, "humidity_pct": 85,
         "temperature_c": 25, "nitrogen_mg_kg": 30, "phosphorus_mg_kg": 20,
         "potassium_mg_kg": 150, "season": "Rainy season"},
        {"node_id": "NODE_05", "currently_planted_crop": "Cassava", "moisture_pct": 20},
    ]}


class FieldReportTests(unittest.TestCase):
    def summary_response(self, answer, *, data=None, temporal=None):
        with (
            patch.object(chat_llm, "_get_farm_snapshot", return_value=data or snapshot()),
            patch.object(chat_llm, "_get_automatic_temporal_context", return_value=temporal or {}),
            patch.object(chat_llm, "_build_llm_providers", return_value=[chat_llm._LLMProvider("AgentRouter", object(), "test-model")]),
            patch.object(chat_llm, "_call_with_provider_fallback", return_value=(answer, 0)) as call,
        ):
            response = chat_llm.generate_rag_response("Summarise recent conditions", [], node_id="NODE_06", response_mode="field_summary")
        return response, call

    def test_summary_uses_the_actual_recorded_period_and_short_output_budget(self):
        sentence = "Soil moisture rose over the recorded period, so check drainage before adding more water."
        period = {"start": "2026-09-17T08:00:00Z", "end": "2026-09-17T09:00:00Z", "history_duration_hours": 1}
        response, call = self.summary_response(sentence, temporal={"nodes": {"NODE_06": {"history": period}}})
        messages = call.call_args.args[1]
        self.assertIn("exactly ONE short sentence", messages[0]["content"])
        self.assertIn('"reporting_period": ' + json.dumps(period), messages[1]["content"])
        self.assertLessEqual(call.call_args.kwargs["max_tokens"], 192)
        self.assertEqual(response.answer, sentence)

    def test_multi_sentence_summary_falls_back_without_method_note(self):
        response, _ = self.summary_response("The soil is wet. Improve drainage.")
        self.assertEqual(response.model_name, "sensor-screening-report")
        self.assertIsNotNone(normalize_field_summary(response.answer))
        self.assertIn("too wet", response.answer)
        self.assertNotIn(EVIDENCE_REPORT_NOTE, response.answer)

    def test_stale_readings_override_a_confident_summary(self):
        data = snapshot()
        data["nodes"][0]["timestamp_utc"] = "2020-01-01T00:00:00Z"
        response, _ = self.summary_response("The field is healthy and no action is needed.", data=data)
        self.assertIn("out of date", response.answer)
        self.assertIsNotNone(normalize_field_summary(response.answer))

    def test_summary_keeps_decimal_readings_but_rejects_paragraphs_and_fragments(self):
        self.assertIsNotNone(normalize_field_summary("Soil moisture is 32.5%, so check the root zone before adjusting irrigation."))
        for answer in ("### Assessment\nThe soil is wet.", "soil moisture", "The soil is wet. Clear drainage.", "word " * 46 + "."):
            self.assertIsNone(normalize_field_summary(answer))

    def test_fallback_summary_includes_observed_change_without_inventing_a_duration(self):
        data = snapshot()
        data["nodes"][0].update(moisture_pct=50, humidity_pct=65)
        evidence = build_report_evidence(data, {"nodes": {"NODE_06": {
            "historical_analysis": {"moisture": {"trend": "falling", "samples": 12}},
        }}}, "NODE_06")
        summary = render_field_summary(evidence)
        self.assertIn("falling over the recorded period", summary)
        self.assertNotIn("today", summary)
        self.assertIsNotNone(normalize_field_summary(summary))

    def test_local_summary_generates_once_and_does_not_expand_into_sections(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text)
            def decode(self, tokens, **kwargs):
                return "".join(tokens)
        client = chat_llm._LocalTransformersClient("test-model")
        client._tokenizer = Tokenizer()
        sentence = "The latest readings suggest excess moisture, so check drainage before irrigating."
        content = report_content(build_report_evidence(snapshot(), {}, "NODE_06"), "", summary=True)
        with patch.object(client, "_ensure_loaded"), patch.object(client, "_generate_text", return_value=sentence) as generate:
            answer = client.generate(messages=[{"role": "user", "content": content}])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(answer, sentence)

    def test_missing_forecast_and_selected_node_are_explicit(self):
        evidence = build_report_evidence(snapshot(), {"nodes": {}}, "NODE_06")
        self.assertEqual(evidence["forecast_status"], "unavailable")
        self.assertIsNone(evidence["future_estimates"])
        self.assertFalse(evidence["data_quality"]["stale"])
        self.assertNotIn("Cassava", json.dumps(evidence))
        self.assertIn("supporting_steps", evidence["priority_brief"]["priorities"][0])

    def test_old_readings_are_flagged_without_history(self):
        data = snapshot()
        data["nodes"][0]["timestamp_utc"] = "2020-01-01T00:00:00Z"
        evidence = build_report_evidence(data, {}, "NODE_06")
        self.assertTrue(evidence["data_quality"]["stale"])

    def test_unavailable_node_does_not_borrow_other_nodes(self):
        evidence = build_report_evidence(snapshot(), {}, "NODE_01")
        self.assertEqual(evidence["current_measurements"], {"status": "unavailable"})
        self.assertNotIn("priority_brief", evidence)
        self.assertNotIn("Maize", json.dumps(evidence))

    def test_report_mode_reaches_provider_with_focused_evidence(self):
        provider = chat_llm._LLMProvider("LocalLLM", object(), "test-model")
        with (
            patch.object(chat_llm, "_get_farm_snapshot", return_value=snapshot()),
            patch.object(chat_llm, "_get_automatic_temporal_context", return_value={"nodes": {}}),
            patch.object(chat_llm, "_build_llm_providers", return_value=[provider]),
            patch.object(chat_llm, "_call_with_provider_fallback", return_value=("Full report.", 0)) as call,
        ):
            result = chat_llm.generate_rag_response("Prepare a field report", [], node_id="NODE_06", response_mode="field_report")
        messages = call.call_args.args[1]
        self.assertIn("250–400", messages[0]["content"])
        self.assertIn("Recommended actions", messages[0]["content"])
        self.assertNotIn("NODE_05", messages[1]["content"])
        self.assertIn('"forecast_status": "unavailable"', messages[1]["content"])
        self.assertIsNone(call.call_args.kwargs["tools"])
        self.assertEqual(result.answer, "Full report.")

    def test_local_report_generates_all_four_sections_with_bounded_evidence(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text)

            def decode(self, tokens, **kwargs):
                return "".join(tokens)

        client = chat_llm._LocalTransformersClient("test-model")
        client._tokenizer = Tokenizer()
        evidence = build_report_evidence(snapshot(), {}, "NODE_06")
        messages = [{"role": "user", "content": report_content(evidence, "long guidance " * 2000)}]
        with (
            patch.object(client, "_ensure_loaded"),
            patch.object(client, "_generate_text", side_effect=["Assessment.", "Observations.", "Actions.", "Monitoring."]) as generate,
            patch.object(chat_llm, "LOCAL_LLM_MAX_INPUT_TOKENS", 4096),
        ):
            answer = client.generate(messages=messages)
        self.assertEqual(generate.call_count, 4)
        for heading in ("Overall assessment", "Supporting observations", "Recommended actions", "What to monitor"):
            self.assertIn(f"### {heading}\n", answer)
        for call in generate.call_args_list:
            prompt = call.args[0]
            self.assertLessEqual(len(prompt), 4096)
            self.assertIn('"moisture_pct": 85', prompt)
            self.assertIn('"status": "unavailable"', prompt)
            self.assertIn("Suspend avoidable irrigation", prompt)

        self.assertIn("AI-written interpretation was unavailable", answer)
        self.assertIn("soil moisture 85%", answer)
        self.assertIn("No usable forecast is available", answer)

    def test_malformed_model_output_is_not_presentable(self):
        self.assertFalse(usable_local_report(["The field is a stale false.", '"', 'Steps: ["monitor"]', '"']))
        self.assertFalse(usable_local_report(["repeat this phrase over and over again now " * 10] * 4))

    def test_stale_report_does_not_prescribe_from_old_measurements(self):
        data = snapshot()
        data["nodes"][0]["timestamp_utc"] = "2020-01-01T00:00:00Z"
        report = render_evidence_report(build_report_evidence(data, {}, "NODE_06"))
        self.assertIn("out of date", report)
        self.assertIn("obtain a fresh set of readings", report)
        self.assertNotIn("Suspend avoidable irrigation", report)

    def test_fallback_metadata_does_not_claim_an_llm_or_retrieved_sources(self):
        fallback = render_evidence_report(build_report_evidence(snapshot(), {}, "NODE_06"))
        with (
            patch.object(chat_llm, "_get_farm_snapshot", return_value=snapshot()),
            patch.object(chat_llm, "_get_automatic_temporal_context", return_value={}),
            patch.object(chat_llm, "_build_llm_providers", return_value=[chat_llm._LLMProvider("LocalLLM", object(), "flan-t5-small")]),
            patch.object(chat_llm, "_call_with_provider_fallback", return_value=(fallback, 0)),
        ):
            response = chat_llm.generate_rag_response("Field report", [], node_id="NODE_06", response_mode="field_report")
        self.assertEqual(response.model_name, "sensor-screening-report")
        self.assertFalse(response.grounded)
        self.assertEqual(response.sources, [])

    def test_unavailable_local_model_still_returns_a_labelled_report(self):
        client = chat_llm._LocalTransformersClient("missing-model")
        content = report_content(build_report_evidence(snapshot(), {}, "NODE_06"), "")
        with patch.object(client, "_ensure_loaded", side_effect=RuntimeError("Model unavailable")):
            answer = client.generate(messages=[{"role": "user", "content": content}])
        self.assertIn("AI-written interpretation was unavailable", answer)
        self.assertIn("soil moisture 85%", answer)


if __name__ == "__main__":
    unittest.main()
