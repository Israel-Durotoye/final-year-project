import types

from backend.rag import chat_llm


class DummyRouter:
    class Template:
        system_instruction = "You are a helpful assistant."
        memory_preference = "none"
        requires_diagnostics = False
        mode = types.SimpleNamespace(name="chat")

    def route(self, user_query, context_block, telemetry_params):
        return types.SimpleNamespace(template=self.Template())


class DummyCompletions:
    def create(self, *args, **kwargs):
        raise RuntimeError("Groq API call failed after 3 retries: Error code: 429 - rate_limit_exceeded")


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace(completions=DummyCompletions())


def test_generate_rag_response_falls_back_on_groq_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(chat_llm.prompt_router, "get_router", lambda: DummyRouter())
    monkeypatch.setattr(chat_llm, "_fetch_farm_snapshot", lambda: "Farm snapshot")
    monkeypatch.setattr(chat_llm, "_build_context_block", lambda chunks: ("Context block", ["source.txt"]))
    monkeypatch.setattr(chat_llm, "_truncate_context_if_needed", lambda context, query, max_tokens=None: (context, False))
    monkeypatch.setattr(chat_llm, "_extract_telemetry_from_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_llm, "_filter_conversation_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(chat_llm, "_build_user_content", lambda *args, **kwargs: "User content")
    monkeypatch.setattr(chat_llm, "OpenAI", DummyClient)

    chunk = types.SimpleNamespace(rerank_score=0.5, text="soil moisture is low", source="source.txt", page=1)

    response = chat_llm.generate_rag_response("hello", [chunk], conversation_history=[])

    assert response.answer
    assert "temporarily" in response.answer.lower() or "rate limit" in response.answer.lower() or "available context" in response.answer.lower()
