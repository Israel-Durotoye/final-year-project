"""Public service boundary for Soil Doctor temporal farm intelligence.

For one node this service turns raw telemetry into four kinds of evidence:

    historical_analysis   deterministic trends and events (what already happened)
    forecast              multivariate LSTM estimates (what may happen next)
    anomaly_screening     LSTM autoencoder check of the newest window
    forecast_outlook      forecast values compared with the crop's reference
                          ranges, so an emerging risk is stated explicitly

The chat layer passes this evidence to the language model and uses it to choose
which agronomic knowledge to retrieve.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from backend.ml import lstm_anomaly_inference, lstm_forecaster, node_data, soil_health
from backend.ml.temporal_analysis import analyze_history
from backend.ml.temporal_data import (
    DEFAULT_SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
    FEATURE_OUTPUT_NAMES,
    FEATURE_SHORT_NAMES,
    HISTORY_ROWS,
    MIN_ANALYSIS_SAMPLES,
    fetch_node_history,
    latest_complete_window,
    prepare_temporal_rows,
)


logger = logging.getLogger(__name__)
MAX_FARM_NODES = int(os.getenv("TEMPORAL_MAX_FARM_NODES", "8"))


def _anomaly_screening(prepared: Any) -> dict[str, Any]:
    """Screen the newest readings, preferring a window with no timestamp gap."""
    model_status = lstm_anomaly_inference.artifact_status()
    if model_status["status"] != "available":
        return {"status": model_status["status"]}
    required = int(model_status.get("sequence_length") or 24)
    # Raw readings are screened on purpose: the values that the trend analysis
    # sets aside as spikes are exactly what this check should see.
    tail = prepared.rows[prepared.contiguous_tail_start :]
    spans_gap = len(tail) < required <= len(prepared.rows)
    result = lstm_anomaly_inference.score_rows(prepared.rows if spans_gap else tail)
    if result.get("status") == "success":
        result["spans_data_gap"] = spans_gap
    return result


def _forecast_outlook(
    forecast: dict[str, Any] | None,
    crop: Any,
    skill: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare forecast checkpoints with the crop's reference ranges."""
    if not forecast:
        return {"status": "no_forecast", "risks": []}
    try:
        profile = soil_health.resolve_crop_profile(crop)
        parameters = soil_health.crop_parameters(profile) if profile else {}
    except (FileNotFoundError, ValueError):
        profile, parameters = None, {}
    if not parameters:
        return {
            "status": "no_crop_reference",
            "risks": [],
            "note": "No reference ranges are configured for this node's crop, so forecasts are not screened.",
        }

    informative = set((skill or {}).get("informative_sensors") or [])
    risks: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        spec = parameters.get(soil_health.PARAM_MAP[feature])
        output_name = FEATURE_OUTPUT_NAMES[feature]
        if not spec:
            continue
        low, high = float(spec["optimal_min"]), float(spec["optimal_max"])
        for label, point in forecast.items():
            predicted = (point.get(output_name) or {}).get("predicted")
            if predicted is None or low <= predicted <= high:
                continue
            risks.append(
                {
                    "sensor": FEATURE_SHORT_NAMES[feature],
                    "direction": "below_reference_range" if predicted < low else "above_reference_range",
                    "first_horizon": label,
                    "predicted": predicted,
                    "reference_range": [low, high],
                    "forecast_reliability": (
                        "model beat naive baselines for this sensor"
                        if output_name in informative
                        else "level estimate only; treat as weak evidence"
                    ),
                }
            )
            break  # Report the earliest horizon at which the sensor leaves its range.
    return {"status": "screened", "crop_profile": profile, "risks": risks}


def get_temporal_farm_intelligence(
    node_id: str,
    *,
    rows: list[dict[str, Any]] | None = None,
    history_rows: int = HISTORY_ROWS,
    fetcher: Callable[..., dict[str, Any]] = fetch_node_history,
) -> dict[str, Any]:
    """Return audited history, observations and LSTM evidence for one node.

    ``rows`` is a dependency-injection seam for tests and offline backtesting.
    Runtime callers normally omit it, causing at most one telemetry query.
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
    crop = next(
        (row.get("Target_Crop") for row in reversed(prepared.rows) if row.get("Target_Crop")),
        None,
    ) or node_data.declared_crop(cleaned_node)
    analysis = analyze_history(prepared, crop) if len(prepared.rows) >= MIN_ANALYSIS_SAMPLES else {
        "historical_analysis": {},
        "events": [],
        "cross_sensor_observations": [],
    }

    model_status = lstm_forecaster.artifact_status()
    required = int(model_status.get("sequence_length") or DEFAULT_SEQUENCE_LENGTH)
    tail = prepared.contiguous_tail
    if len(tail) < required:
        forecast_result = {
            "status": "insufficient_history",
            "forecast": None,
            "samples_available": len(tail),
            "samples_required": required,
            "model": model_status,
        }
    elif model_status.get("status") != "available":
        forecast_result = {
            "status": model_status.get("status", "not_trained"),
            "forecast": None,
            "reason": (
                "No validated multivariate forecast artifact is deployed. "
                "Historical analysis remains available."
            ),
            "model": model_status,
        }
    else:
        # Only the model's own input window has to be complete; an older
        # missing value elsewhere in the history does not block the forecast.
        window = latest_complete_window(tail, required)
        if window is None:
            forecast_result = {
                "status": "invalid_input",
                "forecast": None,
                "reason": "The latest contiguous model window contains unresolved missing or anomalous values.",
                "model": model_status,
            }
        else:
            forecast_result = lstm_forecaster.forecast(
                window,
                median_interval_minutes=prepared.median_interval_minutes,
            )

    try:
        anomaly = _anomaly_screening(prepared)
    except Exception as exc:  # The screen must never break the analysis.
        logger.warning("Anomaly screening failed for %s: %s", cleaned_node, exc)
        anomaly = {"status": "unavailable", "reason": "Anomaly screening could not be completed."}

    outlook = _forecast_outlook(
        forecast_result.get("forecast"),
        crop,
        forecast_result.get("forecast_skill"),
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
        "forecast_skill": forecast_result.get("forecast_skill"),
        "forecast_outlook": outlook,
        "uncertainty_note": forecast_result.get("uncertainty_note"),
        "anomaly_screening": anomaly,
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
    if row_sets is None:
        # One shared query warms the cache for every simulator node.
        node_data.prefetch_simulator_windows(limited, history_rows)
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
