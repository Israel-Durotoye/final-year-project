"""Tests for AgentRouter-primary and Conduit-fallback generation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.rag import chat_llm


class _StatusError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderFallbackTests(unittest.TestCase):
    def test_retryable_error_detection_is_scoped(self) -> None:
        self.assertTrue(chat_llm._is_retryable_provider_error(_StatusError(429, "rate limited")))
        self.assertTrue(chat_llm._is_retryable_provider_error(_StatusError(402, "usage limit reached")))
        self.assertTrue(chat_llm._is_retryable_provider_error(_StatusError(503, "unavailable")))
        self.assertTrue(chat_llm._is_retryable_provider_error(TimeoutError("timed out")))
        self.assertFalse(chat_llm._is_retryable_provider_error(_StatusError(401, "unauthorized")))
        self.assertFalse(chat_llm._is_retryable_provider_error(ValueError("invalid messages")))

    def test_rate_limit_retries_provider_three_times(self) -> None:
        calls = 0

        class Completions:
            def create(self, **_kwargs):
                nonlocal calls
                calls += 1
                raise _StatusError(429, "rate limit exceeded")

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with self.assertRaises(RuntimeError):
            chat_llm._call_llm_with_retry(
                client,
                "primary-model",
                [{"role": "user", "content": "hello"}],
                provider_name="AgentRouter",
            )
        self.assertEqual(calls, 3)

    def test_agentrouter_rate_limit_falls_back_to_conduit(self) -> None:
        primary_client = object()
        fallback_client = object()
        chunk = SimpleNamespace(
            text="Maintain drainage where excess moisture persists.",
            source="Agronomic guide",
            page=4,
            rerank_score=1.0,
        )

        def fake_call(client, _model, _messages, **_kwargs):
            if client is primary_client:
                raise RuntimeError("AgentRouter API call failed: 429 rate limit exceeded")
            return "Conduit fallback answer"

        with (
            patch.object(
                chat_llm,
                "_get_farm_snapshot",
                return_value={"status": "unavailable", "nodes": [], "node_count": 0},
            ),
            patch.object(
                chat_llm,
                "_get_automatic_temporal_context",
                return_value={"status": "not_requested", "nodes": {}},
            ),
            patch.object(chat_llm, "_resolve_api_key", return_value=("primary-key", "test")),
            patch.object(
                chat_llm,
                "_resolve_conduit_api_key",
                return_value=("fallback-key", "test"),
            ),
            patch.object(chat_llm, "_create_agentrouter_client", return_value=primary_client),
            patch.object(chat_llm, "_create_conduit_client", return_value=fallback_client),
            patch.object(chat_llm, "_call_llm_with_retry", side_effect=fake_call),
        ):
            response = chat_llm.generate_rag_response(
                "How should I manage wet soil?",
                [chunk],
                fallback_model_name="claude-opus-4.8",
            )

        self.assertEqual(response.answer, "Conduit fallback answer")
        self.assertEqual(response.model_name, "claude-opus-4.8")
        self.assertTrue(response.grounded)

    def test_nonretryable_primary_error_does_not_leak_to_fallback(self) -> None:
        providers = [
            chat_llm._LLMProvider("AgentRouter", object(), "primary-model"),
            chat_llm._LLMProvider("Conduit", object(), "fallback-model"),
        ]
        with patch.object(
            chat_llm,
            "_call_llm_with_retry",
            side_effect=RuntimeError("AgentRouter API call failed: 401 unauthorized"),
        ) as call:
            with self.assertRaises(RuntimeError):
                chat_llm._call_with_provider_fallback(
                    providers,
                    [{"role": "user", "content": "hello"}],
                )
        self.assertEqual(call.call_count, 1)

    def test_retryable_remote_error_falls_back_to_local_model(self) -> None:
        local_client = object()
        providers = [
            chat_llm._LLMProvider("AgentRouter", object(), "primary-model"),
            chat_llm._LLMProvider("LocalLLM", local_client, "local-model"),
        ]
        calls: list[object] = []

        def fake_call(client, *_args, **_kwargs):
            calls.append(client)
            if client is local_client:
                return "Offline answer"
            raise RuntimeError("AgentRouter API call failed: 503 service unavailable")

        with patch.object(chat_llm, "_call_llm_with_retry", side_effect=fake_call):
            response, provider_index = chat_llm._call_with_provider_fallback(
                providers,
                [{"role": "user", "content": "hello"}],
            )

        self.assertEqual(response, "Offline answer")
        self.assertEqual(provider_index, 1)
        self.assertEqual(calls, [providers[0].client, local_client])

    def test_tool_followup_can_switch_to_conduit(self) -> None:
        primary_client = object()
        fallback_client = object()
        primary_calls = 0
        tool_response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "get_live_sensor_data",
                                    "arguments": '{"node_id":"NODE_01"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        def fake_call(client, _model, _messages, **_kwargs):
            nonlocal primary_calls
            if client is primary_client:
                primary_calls += 1
                if primary_calls == 1:
                    return tool_response
                raise RuntimeError("AgentRouter API call failed: 503 service unavailable")
            return "Fallback completed the tool-grounded answer."

        with (
            patch.object(
                chat_llm,
                "_get_farm_snapshot",
                return_value={"status": "online", "nodes": [], "node_count": 0},
            ),
            patch.object(
                chat_llm,
                "_get_automatic_temporal_context",
                return_value={"status": "not_requested", "nodes": {}},
            ),
            patch.object(chat_llm, "_resolve_api_key", return_value=("primary-key", "test")),
            patch.object(
                chat_llm,
                "_resolve_conduit_api_key",
                return_value=("fallback-key", "test"),
            ),
            patch.object(chat_llm, "_create_agentrouter_client", return_value=primary_client),
            patch.object(chat_llm, "_create_conduit_client", return_value=fallback_client),
            patch.object(chat_llm, "_call_llm_with_retry", side_effect=fake_call),
            patch.object(chat_llm, "_execute_tool", return_value='{"status":"online"}'),
        ):
            response = chat_llm.generate_rag_response(
                "What is happening at NODE_01?",
                [],
                fallback_model_name="claude-opus-4.8",
            )

        self.assertEqual(response.answer, "Fallback completed the tool-grounded answer.")
        self.assertEqual(response.model_name, "claude-opus-4.8")
        self.assertEqual(primary_calls, 2)

    def test_conduit_can_run_when_agentrouter_key_is_absent(self) -> None:
        fallback_client = object()
        local_client = object()
        with (
            patch.object(
                chat_llm,
                "_resolve_api_key",
                side_effect=EnvironmentError("AgentRouter key missing"),
            ),
            patch.object(
                chat_llm,
                "_resolve_conduit_api_key",
                return_value=("fallback-key", "test"),
            ),
            patch.object(chat_llm, "_create_conduit_client", return_value=fallback_client),
            patch.object(chat_llm, "_create_local_llm_client", return_value=local_client),
        ):
            providers = chat_llm._build_llm_providers(
                agentrouter_api_key=None,
                agentrouter_model="primary-model",
                conduit_model="claude-opus-4.8",
            )
        self.assertEqual(len(providers), 2)
        self.assertEqual(providers[0].name, "Conduit")
        self.assertIs(providers[0].client, fallback_client)
        self.assertEqual(providers[1].name, "LocalLLM")
        self.assertIs(providers[1].client, local_client)

    def test_local_prompt_keeps_snapshot_and_current_question(self) -> None:
        prompt = chat_llm._build_local_prompt([{
            "role": "user",
            "content": (
                "LIVE FARM SNAPSHOT\n{\"node_id\": \"NODE_01\", \"moisture\": 32}\n\n"
                "TEMPORAL FARM ANALYSIS\n{}\n\n"
                "FARM-LEVEL PRIORITY BRIEF\nMonitor moisture.\n\n"
                "KNOWLEDGE-BASE CONTEXT\nApply crop-specific nutrient guidance.\n\n"
                "CURRENT USER QUESTION\nGive me a field report."
            ),
        }])

        self.assertIn("NODE_01", prompt)
        self.assertIn("Monitor moisture", prompt)
        self.assertIn("Give me a field report", prompt)
        self.assertIn("Apply crop-specific nutrient guidance", prompt)
        self.assertNotIn("representative", prompt.casefold())


if __name__ == "__main__":
    unittest.main()
