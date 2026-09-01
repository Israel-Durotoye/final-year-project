"""Deterministic historical trend and event analysis for farm telemetry."""

from __future__ import annotations

import math
import os
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

from backend.ml.temporal_data import (
    FEATURE_COLUMNS,
    FEATURE_SHORT_NAMES,
    PreparedTemporalData,
    parse_timestamp,
)


ROLLING_SAMPLES = int(os.getenv("TEMPORAL_ROLLING_SAMPLES", "6"))
MOISTURE_WET_THRESHOLD = float(os.getenv("TEMPORAL_MOISTURE_WET_THRESHOLD", "70"))
MOISTURE_SATURATION_THRESHOLD = float(
    os.getenv("TEMPORAL_MOISTURE_SATURATION_THRESHOLD", "80")
)

# Minimum rate per hour before ordinary sensor movement is called a trend.
# A minimum net movement is also required, making classification conservative.
TREND_DEADBAND_PER_HOUR: dict[str, float] = {
    "Nitrogen_mg_k": 0.10,
    "Phosphorus_m": 0.06,
    "Potassium_mg_": 0.08,
    "Moisture_%": 0.25,
    "Temperature_C": 0.10,
    "Humidity_%": 0.25,
}
TREND_MIN_NET_CHANGE: dict[str, float] = {
    "Nitrogen_mg_k": 1.0,
    "Phosphorus_m": 0.6,
    "Potassium_mg_": 0.8,
    "Moisture_%": 2.0,
    "Temperature_C": 0.8,
    "Humidity_%": 2.0,
}
ABRUPT_CHANGE: dict[str, float] = {
    "Nitrogen_mg_k": 18.0,
    "Phosphorus_m": 10.0,
    "Potassium_mg_": 14.0,
    "Moisture_%": 12.0,
    "Temperature_C": 6.0,
    "Humidity_%": 15.0,
}


def _series(
    prepared: PreparedTemporalData,
    feature: str,
) -> tuple[list[float], list[float], list[str]]:
    """Return valid values/times from the newest contiguous history segment."""
    rows = prepared.contiguous_tail
    valid: list[tuple[float, float, str]] = []
    origin = None
    for row in rows:
        timestamp = parse_timestamp(row.get("Timestamp"))
        value = row.get(feature)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if timestamp is None or not math.isfinite(numeric):
            continue
        origin = origin or timestamp
        valid.append(((timestamp - origin).total_seconds() / 3600.0, numeric, timestamp.isoformat()))
    return (
        [item[1] for item in valid],
        [item[0] for item in valid],
        [item[2] for item in valid],
    )


def _consecutive_direction(values: list[float]) -> tuple[int, int]:
    rising = falling = 0
    for before, after in reversed(list(zip(values, values[1:]))):
        delta = after - before
        if delta > 0:
            if falling:
                break
            rising += 1
        elif delta < 0:
            if rising:
                break
            falling += 1
        else:
            break
    return rising, falling


def _trend(feature: str, values: list[float], times: list[float], slope: float) -> str:
    if len(values) < 3 or len(times) < 3:
        return "unknown"
    net = values[-1] - values[0]
    deadband = TREND_DEADBAND_PER_HOUR[feature]
    minimum_net = TREND_MIN_NET_CHANGE[feature]
    # A regression slope that disagrees with the window's net direction usually
    # indicates oscillation rather than a defensible directional trend.
    if abs(net) < minimum_net or abs(slope) < deadband or slope * net <= 0:
        return "stable"
    strength = "strongly_" if abs(slope) >= deadband * 3.0 and abs(net) >= minimum_net * 2 else ""
    return f"{strength}{'rising' if slope > 0 else 'falling'}"


def _sensor_metrics(
    prepared: PreparedTemporalData,
    feature: str,
) -> dict[str, Any]:
    values, times, _timestamps = _series(prepared, feature)
    if not values:
        return {"status": "unavailable", "trend": "unknown", "samples": 0}

    slope = 0.0
    if len(values) >= 2 and times[-1] > times[0]:
        slope = float(np.polyfit(np.asarray(times), np.asarray(values), 1)[0])
    previous = values[-2] if len(values) > 1 else None
    first = values[0]
    percent_change = None if math.isclose(first, 0.0, abs_tol=1e-9) else ((values[-1] - first) / abs(first)) * 100.0
    rising, falling = _consecutive_direction(values)
    rolling = values[-min(ROLLING_SAMPLES, len(values)) :]
    deltas_per_hour: list[float] = []
    for index in range(1, len(values)):
        elapsed = times[index] - times[index - 1]
        if elapsed > 0:
            deltas_per_hour.append((values[index] - values[index - 1]) / elapsed)

    result: dict[str, Any] = {
        "current": round(values[-1], 4),
        "previous": round(previous, 4) if previous is not None else None,
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "standard_deviation": round(pstdev(values), 4) if len(values) > 1 else 0.0,
        "rolling_mean": round(mean(rolling), 4),
        "slope_per_hour": round(slope, 4),
        "rate_of_change_per_hour": round(mean(deltas_per_hour), 4) if deltas_per_hour else None,
        "percentage_change": round(percent_change, 3) if percent_change is not None else None,
        "volatility": round(pstdev(deltas_per_hour), 4) if len(deltas_per_hour) > 1 else 0.0,
        "consecutive_rising_samples": rising,
        "consecutive_falling_samples": falling,
        "trend": _trend(feature, values, times, slope),
        "samples": len(values),
    }
    if feature in {"Nitrogen_mg_k", "Phosphorus_m", "Potassium_mg_"}:
        if result["trend"] in {"falling", "strongly_falling"} and falling >= 3:
            result["event"] = "sustained_decline"
        elif result["trend"] in {"rising", "strongly_rising"} and rising >= 3:
            result["event"] = "sustained_increase"
        else:
            result["event"] = "stable_level" if result["trend"] == "stable" else None
    return result


def _consecutive_duration(
    values: list[float],
    timestamps: list[str],
    predicate: Any,
    median_interval_minutes: float | None,
) -> tuple[int, float, str | None]:
    count = 0
    start_index = len(values)
    for index in range(len(values) - 1, -1, -1):
        if predicate(values[index]):
            count += 1
            start_index = index
        else:
            break
    if count == 0:
        return 0, 0.0, None
    start = parse_timestamp(timestamps[start_index])
    end = parse_timestamp(timestamps[-1])
    duration = (end - start).total_seconds() / 3600.0 if start and end else 0.0
    if count > 1 and median_interval_minutes:
        duration += median_interval_minutes / 60.0
    return count, max(0.0, duration), timestamps[start_index]


def _moisture_events(prepared: PreparedTemporalData, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    values, _times, timestamps = _series(prepared, "Moisture_%")
    if len(values) < 3:
        return []
    events: list[dict[str, Any]] = []
    deltas = [after - before for before, after in zip(values, values[1:])]
    sharp_indices = [index + 1 for index, delta in enumerate(deltas) if delta >= ABRUPT_CHANGE["Moisture_%"]]
    wet_count, wet_hours, wet_started = _consecutive_duration(
        values,
        timestamps,
        lambda value: value >= MOISTURE_WET_THRESHOLD,
        prepared.median_interval_minutes,
    )
    saturation_count, saturation_hours, saturation_started = _consecutive_duration(
        values,
        timestamps,
        lambda value: value >= MOISTURE_SATURATION_THRESHOLD,
        prepared.median_interval_minutes,
    )

    if wet_count >= 3 and (sharp_indices or metrics.get("trend") in {"rising", "strongly_rising"}):
        supporting = min(0.85, 0.45 + min(wet_count, 8) * 0.05 + (0.10 if sharp_indices else 0.0))
        events.append(
            {
                "type": "sustained_wetting",
                "started_at": wet_started,
                "duration_hours": round(wet_hours, 3),
                "detail": "sharp_wetting_followed_by_sustained_high_moisture" if sharp_indices else "rising_then_sustained_high_moisture",
                "likely_cause": "external_water_input",
                "cause_detail": "rain_or_irrigation_unknown",
                "cause_confidence": round(supporting, 2),
                "epistemic_note": "Soil sensors alone cannot distinguish rainfall from irrigation.",
            }
        )
    elif sharp_indices:
        events.append(
            {
                "type": "sudden_wetting",
                "started_at": timestamps[sharp_indices[-1]],
                "duration_hours": None,
                "likely_cause": "external_water_input",
                "cause_detail": "rain_or_irrigation_unknown",
                "cause_confidence": 0.55,
                "epistemic_note": "Soil sensors alone cannot distinguish rainfall from irrigation.",
            }
        )

    if saturation_count >= 3:
        events.append(
            {
                "type": "prolonged_saturation",
                "started_at": saturation_started,
                "duration_hours": round(saturation_hours, 3),
                "threshold_pct": MOISTURE_SATURATION_THRESHOLD,
                "likely_cause": "uncertain_water_source",
                "cause_detail": "rain_or_irrigation_unknown",
                "cause_confidence": None,
            }
        )

    declining = sum(delta < -0.25 for delta in deltas[-6:])
    total_recent_change = values[-1] - values[max(0, len(values) - 6)]
    if declining >= 3 and total_recent_change <= -5:
        event_type = "rapid_drying" if total_recent_change <= -15 else "gradual_drying"
        events.append(
            {
                "type": event_type,
                "started_at": timestamps[max(0, len(values) - 6)],
                "duration_hours": round(
                    (parse_timestamp(timestamps[-1]) - parse_timestamp(timestamps[max(0, len(values) - 6)])).total_seconds() / 3600.0,
                    3,
                ),
                "likely_cause": "unknown",
                "cause_confidence": None,
            }
        )

    repeated_wetting = sum(delta >= ABRUPT_CHANGE["Moisture_%"] for delta in deltas)
    if repeated_wetting >= 2:
        events.append(
            {
                "type": "repeated_wetting",
                "occurrences": repeated_wetting,
                "likely_cause": "repeated_external_water_input",
                "cause_detail": "rain_or_irrigation_unknown",
                "cause_confidence": round(min(0.8, 0.45 + repeated_wetting * 0.1), 2),
            }
        )

    recent_deltas = deltas[-min(12, len(deltas)) :]
    sign_changes = sum(a * b < 0 for a, b in zip(recent_deltas, recent_deltas[1:]))
    if sign_changes >= 4 and max(values[-12:]) - min(values[-12:]) >= 15:
        events.append(
            {
                "type": "unusual_moisture_oscillation",
                "sign_changes": sign_changes,
                "likely_cause": "unknown_or_sensor_instability",
                "cause_confidence": None,
            }
        )
    return events


def _nutrient_step_events(prepared: PreparedTemporalData) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for feature in ("Nitrogen_mg_k", "Phosphorus_m", "Potassium_mg_"):
        values, _times, timestamps = _series(prepared, feature)
        if len(values) < 2:
            continue
        threshold = ABRUPT_CHANGE[feature]
        sensor_events: list[dict[str, Any]] = []
        for index, (before, after) in enumerate(zip(values, values[1:])):
            delta = after - before
            if abs(delta) < threshold:
                continue
            direction = "increase" if delta > 0 else "drop"
            sensor_events.append(
                {
                    "type": f"abrupt_nutrient_{direction}",
                    "sensor": FEATURE_SHORT_NAMES[feature],
                    "observed_at": timestamps[index + 1],
                    "change": round(delta, 4),
                    "interpretation": (
                        "nutrient_input_like_event_or_measurement_change; cause_unknown"
                        if delta > 0
                        else "abrupt_decrease_or_measurement_change; cause_unknown"
                    ),
                }
            )
        if len(sensor_events) >= 3:
            events.append(
                {
                    "type": "unusual_nutrient_volatility",
                    "sensor": FEATURE_SHORT_NAMES[feature],
                    "occurrences": len(sensor_events),
                    "start": sensor_events[0]["observed_at"],
                    "end": sensor_events[-1]["observed_at"],
                    "largest_absolute_change": max(
                        abs(float(event["change"])) for event in sensor_events
                    ),
                    "interpretation": "repeated abrupt changes; input events, sensor instability, or another cause remain unconfirmed",
                }
            )
        else:
            events.extend(sensor_events)
    return events


def analyze_history(prepared: PreparedTemporalData) -> dict[str, Any]:
    """Create historical observations only; this function never forecasts."""
    per_sensor = {
        FEATURE_SHORT_NAMES[feature]: _sensor_metrics(prepared, feature)
        for feature in FEATURE_COLUMNS
    }
    events = _moisture_events(prepared, per_sensor["moisture"])
    events.extend(_nutrient_step_events(prepared))

    observations: list[dict[str, Any]] = []
    moisture_trend = per_sensor["moisture"].get("trend")
    nitrogen_trend = per_sensor["nitrogen"].get("trend")
    potassium_trend = per_sensor["potassium"].get("trend")
    if moisture_trend in {"rising", "strongly_rising"} and (
        nitrogen_trend in {"falling", "strongly_falling"}
        or potassium_trend in {"falling", "strongly_falling"}
    ):
        observations.append(
            {
                "type": "simultaneous_sensor_changes",
                "observed": {
                    "moisture": moisture_trend,
                    "nitrogen": nitrogen_trend,
                    "potassium": potassium_trend,
                },
                "interpretation": "deferred_to_soil_doctor_and_rag_evidence",
            }
        )

    return {
        "historical_analysis": per_sensor,
        "events": events,
        "cross_sensor_observations": observations,
    }
