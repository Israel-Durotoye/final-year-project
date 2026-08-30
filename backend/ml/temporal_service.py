"""Public service boundary for Soil Doctor temporal farm intelligence."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from backend.ml import lstm_forecaster
from backend.ml.temporal_analysis import analyze_history
from backend.ml.temporal_data import (
    DEFAULT_SEQUENCE_LENGTH,
    HISTORY_ROWS,
    MIN_ANALYSIS_SAMPLES,
    complete_feature_matrix,
    fetch_node_history,
    prepare_temporal_rows,
)


logger = logging.getLogger(__name__)
MAX_FARM_NODES = int(os.getenv("TEMPORAL_MAX_FARM_NODES", "6"))


def get_temporal_farm_intelligence(
    node_id: str,
    *,
    rows: list[dict[str, Any]] | None = None,
    history_rows: int = HISTORY_ROWS,
    fetcher: Callable[..., dict[str, Any]] = fetch_node_history,
) -> dict[str, Any]:
    """Return audited history, observations and an optional deployed forecast.

    ``rows`` is a dependency-injection seam for tests and offline backtesting.
    Runtime callers normally omit it, causing exactly one Supabase query.
    """
    cleaned_node = str(node_id).strip()
    if rows is None:
        fetched = fetcher(cleaned_node, limit=history_rows)
        if fetched.get("status") == "unavailable":
            return {
                "status": "unavailable",
                "node_id": cleaned_node,
                "reason": fetched.get("reason", "Sensor history is unavailable."),
                "history": {"samples_used": 0},
                "historical_analysis": {},
                "events": [],
                "forecast": None,
                "model": lstm_forecaster.artifact_status(),
            }
        rows = list(fetched.get("rows") or [])

    prepared = prepare_temporal_rows(rows, node_id=cleaned_node)
    analysis = analyze_history(prepared) if len(prepared.rows) >= MIN_ANALYSIS_SAMPLES else {
        "historical_analysis": {},
        "events": [],
        "cross_sensor_observations": [],
    }

    tail_matrix = complete_feature_matrix(prepared.contiguous_tail)
    required = int(
        lstm_forecaster.artifact_status().get("sequence_length")
        or DEFAULT_SEQUENCE_LENGTH
    )
    if len(prepared.contiguous_tail) < required:
        forecast_result = {
            "status": "insufficient_history",
            "forecast": None,
            "samples_available": len(prepared.contiguous_tail),
            "samples_required": required,
            "model": lstm_forecaster.artifact_status(),
        }
    elif tail_matrix is None:
        forecast_result = {
            "status": "invalid_input",
            "forecast": None,
            "reason": "The latest contiguous model window contains unresolved missing or anomalous values.",
            "model": lstm_forecaster.artifact_status(),
        }
    else:
        forecast_result = lstm_forecaster.forecast(
            tail_matrix,
            median_interval_minutes=prepared.median_interval_minutes,
        )

    if len(prepared.rows) < MIN_ANALYSIS_SAMPLES:
        overall_status = "insufficient_history"
    elif forecast_result.get("status") == "success":
        overall_status = "success"
    elif forecast_result.get("status") in {"not_trained", "insufficient_history", "cadence_mismatch", "invalid_input"}:
        # Historical analysis is still a successful fallback capability.
        overall_status = "historical_only"
    else:
        overall_status = "partial"

    result: dict[str, Any] = {
        "status": overall_status,
        "node_id": cleaned_node,
        "history": prepared.history,
        "data_quality": prepared.data_quality,
        **analysis,
        "forecast_status": forecast_result.get("status"),
        "forecast": forecast_result.get("forecast"),
        "forecast_trends": forecast_result.get("forecast_trends", {}),
        "uncertainty_note": forecast_result.get("uncertainty_note"),
        "model": forecast_result.get("model", lstm_forecaster.artifact_status()),
    }
    if forecast_result.get("reason"):
        result["forecast_unavailable_reason"] = forecast_result["reason"]
    if "samples_required" in forecast_result:
        result["samples_required_for_forecast"] = forecast_result["samples_required"]
    return result


def get_multi_node_temporal_intelligence(
    node_ids: list[str],
    *,
    history_rows: int = HISTORY_ROWS,
    max_nodes: int = MAX_FARM_NODES,
    row_sets: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Analyze node histories independently and never mix their sequences."""
    unique: list[str] = []
    for value in node_ids:
        node = str(value).strip()
        if node and node not in unique:
            unique.append(node)
    limited = unique[:max_nodes]
    results = {
        node: get_temporal_farm_intelligence(
            node,
            rows=(row_sets or {}).get(node) if row_sets is not None else None,
            history_rows=history_rows,
        )
        for node in limited
    }
    return {
        "status": "success" if results else "no_nodes",
        "nodes_requested": len(unique),
        "nodes_analyzed": len(results),
        "truncated": len(unique) > len(limited),
        "nodes": results,
    }
