"""Telemetry loading and leakage-safe preprocessing for temporal intelligence.

This module is deliberately independent of Keras.  Historical summaries remain
available when no forecasting model has been trained or deployed.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median
from typing import Any, Callable, Iterable

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - optional in unit-test environments
    Client = None
    create_client = None


FARM_DATA_TABLE = os.getenv("FARM_DATA_TABLE", "capstone_dataset")
HISTORY_ROWS = int(os.getenv("TEMPORAL_HISTORY_ROWS", "100"))
MIN_ANALYSIS_SAMPLES = int(os.getenv("TEMPORAL_MIN_ANALYSIS_SAMPLES", "3"))
DEFAULT_SEQUENCE_LENGTH = int(os.getenv("TEMPORAL_SEQUENCE_LENGTH", "48"))
MAX_SHORT_GAP_SAMPLES = int(os.getenv("TEMPORAL_MAX_INTERPOLATION_SAMPLES", "2"))
GAP_MULTIPLIER = float(os.getenv("TEMPORAL_GAP_MULTIPLIER", "3.0"))

# This is the single canonical order used by preprocessing, training, inference,
# metadata validation, and forecast presentation.
FEATURE_COLUMNS: tuple[str, ...] = (
    "Nitrogen_mg_k",
    "Phosphorus_m",
    "Potassium_mg_",
    "Moisture_%",
    "Temperature_C",
    "Humidity_%",
)

FEATURE_OUTPUT_NAMES: dict[str, str] = {
    "Nitrogen_mg_k": "nitrogen_mg_kg",
    "Phosphorus_m": "phosphorus_mg_kg",
    "Potassium_mg_": "potassium_mg_kg",
    "Moisture_%": "moisture_pct",
    "Temperature_C": "temperature_c",
    "Humidity_%": "humidity_pct",
}

FEATURE_SHORT_NAMES: dict[str, str] = {
    "Nitrogen_mg_k": "nitrogen",
    "Phosphorus_m": "phosphorus",
    "Potassium_mg_": "potassium",
    "Moisture_%": "moisture",
    "Temperature_C": "temperature",
    "Humidity_%": "humidity",
}

# Broad physical/sensor sanity bounds, not crop-optimal agronomic thresholds.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "Nitrogen_mg_k": (0.0, 1000.0),
    "Phosphorus_m": (0.0, 1000.0),
    "Potassium_mg_": (0.0, 1000.0),
    "Moisture_%": (0.0, 100.0),
    "Temperature_C": (-20.0, 70.0),
    "Humidity_%": (0.0, 100.0),
}

# A point must depart this far from both neighbours before it is a candidate
# isolated spike. It remains in the audit record and is excluded only from
# deterministic trend/model inputs.
ISOLATED_SPIKE_DEADBANDS: dict[str, float] = {
    "Nitrogen_mg_k": 25.0,
    "Phosphorus_m": 15.0,
    "Potassium_mg_": 20.0,
    "Moisture_%": 15.0,
    "Temperature_C": 8.0,
    "Humidity_%": 20.0,
}


def parse_timestamp(value: Any) -> datetime | None:
    """Parse Supabase/ISO timestamps as timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        parsed = value
    elif value is None:
        return None
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass
class PreparedTemporalData:
    node_id: str
    rows: list[dict[str, Any]]
    analysis_rows: list[dict[str, Any]]
    history: dict[str, Any]
    data_quality: dict[str, Any]
    median_interval_minutes: float | None
    contiguous_tail_start: int
    excluded_points: set[tuple[int, str]] = field(default_factory=set)

    @property
    def contiguous_tail(self) -> list[dict[str, Any]]:
        return self.analysis_rows[self.contiguous_tail_start :]


def _resolve_credentials() -> tuple[str | None, str | None]:
    return (
        os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL"),
        os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY"),
    )


def fetch_node_history(
    node_id: str,
    *,
    limit: int = HISTORY_ROWS,
    client: Any | None = None,
) -> dict[str, Any]:
    """Fetch one full sensor row-set, newest first in SQL then oldest first here."""
    cleaned_node = str(node_id).strip()
    if not cleaned_node:
        return {"status": "unavailable", "reason": "node_id is required.", "rows": []}

    if client is None:
        if create_client is None:
            return {
                "status": "unavailable",
                "reason": "Supabase package is not installed.",
                "rows": [],
            }
        url, key = _resolve_credentials()
        if not url or not key:
            return {
                "status": "unavailable",
                "reason": "Supabase credentials are not configured.",
                "rows": [],
            }
        client = create_client(url, key)

    try:
        # select("*") is intentional: PostgREST's select parser requires special
        # quoting for production columns containing '%'. This is still one query
        # for all measured parameters and preserves Target_Crop for context.
        result = (
            client.table(FARM_DATA_TABLE)
            .select("*")
            .eq("Node_ID", cleaned_node)
            .order("Timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        rows = getattr(result, "data", None) or []
    except Exception as exc:  # pragma: no cover - network-specific
        return {
            "status": "unavailable",
            "reason": f"Unable to retrieve sensor history: {str(exc)[:160]}",
            "rows": [],
        }

    return {
        "status": "ok" if rows else "insufficient_history",
        "node_id": cleaned_node,
        "rows": list(reversed(rows)),
        "count": len(rows),
    }


def _deduplicate(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    parsed: list[tuple[datetime, int, dict[str, Any]]] = []
    invalid_timestamps = 0
    for position, source in enumerate(rows):
        timestamp = parse_timestamp(source.get("Timestamp"))
        if timestamp is None:
            invalid_timestamps += 1
            continue
        row = dict(source)
        row["Timestamp"] = timestamp.isoformat()
        parsed.append((timestamp, position, row))

    parsed.sort(key=lambda item: (item[0], item[1]))
    best_by_timestamp: dict[datetime, tuple[int, dict[str, Any]]] = {}
    for timestamp, position, row in parsed:
        completeness = sum(_number(row.get(feature)) is not None for feature in FEATURE_COLUMNS)
        previous = best_by_timestamp.get(timestamp)
        if previous is None or completeness >= previous[0]:
            best_by_timestamp[timestamp] = (completeness, row)

    ordered = [best_by_timestamp[timestamp][1] for timestamp in sorted(best_by_timestamp)]
    return ordered, len(parsed) - len(ordered), invalid_timestamps


def _find_isolated_spikes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        deadband = ISOLATED_SPIKE_DEADBANDS[feature]
        for index in range(1, len(rows) - 1):
            before = _number(rows[index - 1].get(feature))
            value = _number(rows[index].get(feature))
            after = _number(rows[index + 1].get(feature))
            if before is None or value is None or after is None:
                continue
            neighbour_level = (before + after) / 2.0
            neighbours_agree = abs(after - before) <= deadband * 0.35
            if neighbours_agree and abs(value - neighbour_level) >= deadband:
                findings.append(
                    {
                        "type": "isolated_spike",
                        "feature": FEATURE_SHORT_NAMES[feature],
                        "timestamp": rows[index]["Timestamp"],
                        "value": value,
                        "neighbour_mean": round(neighbour_level, 4),
                        "policy": "preserved_in_audit_excluded_from_trends_and_forecast_input",
                        "row_index": index,
                        "column": feature,
                    }
                )
    return findings


def _longest_repeated_run(rows: list[dict[str, Any]], feature: str) -> int:
    longest = current = 0
    previous: float | None = None
    for row in rows:
        value = _number(row.get(feature))
        if value is None:
            current = 0
            previous = None
        elif previous is not None and math.isclose(value, previous, abs_tol=1e-9):
            current += 1
        else:
            current = 1
        previous = value
        longest = max(longest, current)
    return longest


def prepare_temporal_rows(
    rows: Iterable[dict[str, Any]],
    *,
    node_id: str | None = None,
    now: datetime | None = None,
) -> PreparedTemporalData:
    """Sort, deduplicate, audit and cautiously interpolate a telemetry history."""
    source_rows = list(rows)
    ordered, duplicate_count, invalid_timestamps = _deduplicate(source_rows)
    resolved_node = str(node_id or (ordered[-1].get("Node_ID") if ordered else "") or "").strip()

    timestamps = [parse_timestamp(row["Timestamp"]) for row in ordered]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    intervals = [
        (later - earlier).total_seconds() / 60.0
        for earlier, later in zip(timestamps, timestamps[1:])
        if later > earlier
    ]
    median_interval = float(median(intervals)) if intervals else None
    gap_threshold = (median_interval * GAP_MULTIPLIER) if median_interval else None
    gaps: list[dict[str, Any]] = []
    gap_after_indices: set[int] = set()
    if gap_threshold is not None:
        for index, interval in enumerate(intervals):
            if interval > gap_threshold:
                gap_after_indices.add(index)
                gaps.append(
                    {
                        "after": ordered[index]["Timestamp"],
                        "before": ordered[index + 1]["Timestamp"],
                        "duration_minutes": round(interval, 3),
                    }
                )

    missing_counts = {
        feature: sum(_number(row.get(feature)) is None for row in ordered)
        for feature in FEATURE_COLUMNS
    }
    impossible: list[dict[str, Any]] = []
    excluded: set[tuple[int, str]] = set()
    for index, row in enumerate(ordered):
        for feature, (lower, upper) in PHYSICAL_BOUNDS.items():
            value = _number(row.get(feature))
            if value is not None and not lower <= value <= upper:
                excluded.add((index, feature))
                impossible.append(
                    {
                        "type": "impossible_reading",
                        "feature": FEATURE_SHORT_NAMES[feature],
                        "timestamp": row["Timestamp"],
                        "value": value,
                        "physical_bounds": [lower, upper],
                        "row_index": index,
                        "column": feature,
                    }
                )

    spike_findings = _find_isolated_spikes(ordered)
    for finding in spike_findings:
        excluded.add((int(finding["row_index"]), str(finding["column"])))

    analysis_rows = [dict(row) for row in ordered]
    for index, feature in excluded:
        analysis_rows[index][feature] = None

    interpolated: list[dict[str, Any]] = []
    if median_interval and ordered:
        for feature in FEATURE_COLUMNS:
            index = 0
            while index < len(analysis_rows):
                if _number(analysis_rows[index].get(feature)) is not None:
                    index += 1
                    continue
                start = index
                while index < len(analysis_rows) and _number(analysis_rows[index].get(feature)) is None:
                    index += 1
                end = index - 1
                run_length = end - start + 1
                before_index = start - 1
                after_index = index
                if (
                    run_length <= MAX_SHORT_GAP_SAMPLES
                    and before_index >= 0
                    and after_index < len(analysis_rows)
                    and not any(gap in gap_after_indices for gap in range(before_index, after_index))
                ):
                    before_value = _number(analysis_rows[before_index].get(feature))
                    after_value = _number(analysis_rows[after_index].get(feature))
                    before_time = parse_timestamp(analysis_rows[before_index]["Timestamp"])
                    after_time = parse_timestamp(analysis_rows[after_index]["Timestamp"])
                    if before_value is not None and after_value is not None and before_time and after_time:
                        span = (after_time - before_time).total_seconds()
                        expected_span = median_interval * 60.0 * (run_length + 1)
                        if span <= expected_span * 1.5:
                            for fill_index in range(start, end + 1):
                                fill_time = parse_timestamp(analysis_rows[fill_index]["Timestamp"])
                                if fill_time is None or span <= 0:
                                    continue
                                fraction = (fill_time - before_time).total_seconds() / span
                                value = before_value + (after_value - before_value) * fraction
                                analysis_rows[fill_index][feature] = value
                                interpolated.append(
                                    {
                                        "feature": FEATURE_SHORT_NAMES[feature],
                                        "timestamp": analysis_rows[fill_index]["Timestamp"],
                                        "method": "linear_short_gap",
                                    }
                                )

    # Only the segment after the latest large timestamp gap is eligible for an
    # LSTM window. Old data remain available in the audit/history summary.
    contiguous_tail_start = (max(gap_after_indices) + 1) if gap_after_indices else 0

    irregular_count = 0
    if median_interval:
        irregular_count = sum(
            abs(interval - median_interval) > median_interval * 0.5
            for interval in intervals
        )

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    age_minutes = (
        (now_utc.astimezone(timezone.utc) - timestamps[-1]).total_seconds() / 60.0
        if timestamps else None
    )
    stale_threshold = max(30.0, (median_interval or 10.0) * 5.0)

    repeated = {
        FEATURE_SHORT_NAMES[feature]: run
        for feature in FEATURE_COLUMNS
        if (run := _longest_repeated_run(ordered, feature)) >= 10
    }

    history = {
        "samples_received": len(source_rows),
        "samples_used": len(ordered),
        "start": ordered[0]["Timestamp"] if ordered else None,
        "end": ordered[-1]["Timestamp"] if ordered else None,
        "median_interval_minutes": round(median_interval, 3) if median_interval else None,
        "history_duration_hours": round((timestamps[-1] - timestamps[0]).total_seconds() / 3600.0, 3)
        if len(timestamps) > 1
        else 0.0,
        "contiguous_tail_samples": len(ordered) - contiguous_tail_start,
        "contiguous_tail_duration_hours": round(
            (timestamps[-1] - timestamps[contiguous_tail_start]).total_seconds() / 3600.0,
            3,
        )
        if len(timestamps) - contiguous_tail_start > 1
        else 0.0,
    }
    data_quality = {
        "duplicate_timestamps_removed": duplicate_count,
        "invalid_timestamps_removed": invalid_timestamps,
        "missing_values": {FEATURE_SHORT_NAMES[key]: value for key, value in missing_counts.items()},
        "interpolated_values": interpolated,
        "gap_count": len(gaps),
        "gaps": gaps,
        "irregular_intervals": irregular_count,
        "is_irregular": bool(intervals and irregular_count / len(intervals) > 0.2),
        "impossible_readings": [
            {key: value for key, value in finding.items() if key not in {"row_index", "column"}}
            for finding in impossible
        ],
        "possible_anomalies": [
            {key: value for key, value in finding.items() if key not in {"row_index", "column"}}
            for finding in spike_findings
        ],
        "repeated_identical_runs": repeated,
        "latest_sample_age_minutes": round(age_minutes, 3) if age_minutes is not None else None,
        "stale": bool(age_minutes is not None and age_minutes > stale_threshold),
        "analysis_filter_policy": (
            "Impossible readings and isolated one-sample spikes are preserved in this audit "
            "but excluded from trend calculations and forecast input. Only short bounded "
            "missing runs are linearly interpolated; timestamp gaps are never filled."
        ),
    }
    return PreparedTemporalData(
        node_id=resolved_node,
        rows=ordered,
        analysis_rows=analysis_rows,
        history=history,
        data_quality=data_quality,
        median_interval_minutes=median_interval,
        contiguous_tail_start=contiguous_tail_start,
        excluded_points=excluded,
    )


def complete_feature_matrix(rows: list[dict[str, Any]]) -> list[list[float]] | None:
    """Return a numeric feature matrix only when every measured value is valid."""
    matrix: list[list[float]] = []
    for row in rows:
        values = [_number(row.get(feature)) for feature in FEATURE_COLUMNS]
        if any(value is None for value in values):
            return None
        matrix.append([float(value) for value in values if value is not None])
    return matrix
