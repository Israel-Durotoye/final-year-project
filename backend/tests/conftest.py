"""Shared test setup: no cached telemetry may leak from one test into another."""

import os

os.environ.setdefault("USE_TF", "0")

import pytest

from backend.ml import lstm_crop_inference, node_data
from backend.rag import chat_llm


@pytest.fixture(autouse=True)
def _clear_telemetry_cache():
    node_data.clear_cache()
    lstm_crop_inference.clear_cache()
    chat_llm.reset_provider_cooldowns()
    yield
    node_data.clear_cache()
    lstm_crop_inference.clear_cache()
