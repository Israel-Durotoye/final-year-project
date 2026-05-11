"""
evaluator.py — Soil Doctor Prescriptive Engine: Threshold Evaluator

Responsibility:
    Takes a raw telemetry payload (from a LoRa node or mock generator) and
    evaluates every parameter against the ground-truth thresholds in
    `optimal_thresholds.json`. Produces a structured EvaluationReport containing:
      - Per-parameter flags (status, deviation, severity)
      - A weighted composite Field Health Score (FHS, 0–100)
      - A curated list of alert strings ready to be injected into the LLM prompt

This module contains ZERO LLM logic. It is purely mathematical.
All downstream prescription generation lives in prescriptive_llm.py.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the ground-truth schema
# ---------------------------------------------------------------------------
_THRESHOLDS_PATH = Path(__file__).parent.parent / "data" / "optimal_thresholds.json"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ParameterStatus(str, Enum):
    """
    The health status of a single telemetry parameter.

    OK       — value falls inside the optimal range
    WARNING  — value is outside the optimal range but inside the critical range
    CRITICAL — value is outside the critical range (requires immediate action)
    MISSING  — sensor did not report this parameter
    """
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MISSING = "MISSING"


class DeviationDirection(str, Enum):
    """Which side of the optimal band the reading lies on."""
    LOW = "LOW"    # reading is below optimal_min
    HIGH = "HIGH"  # reading is above optimal_max
    NONE = "NONE"  # reading is within the optimal band


# ---------------------------------------------------------------------------
# Data classes (structured output contracts)
# ---------------------------------------------------------------------------

@dataclass
class ParameterFlag:
    """
    Complete evaluation result for a single telemetry parameter.

    Attributes:
        param_key          : Machine-readable parameter name (matches JSON key).
        display_name       : Human-readable label from the schema.
        unit               : Physical unit string (e.g. 'ppm', '°C').
        observed_value     : The raw sensor reading.
        optimal_min        : Lower bound of the optimal window.
        optimal_max        : Upper bound of the optimal window.
        status             : ParameterStatus enum value.
        direction          : DeviationDirection enum value.
        absolute_deviation : |observed_value - nearest_optimal_bound|.
                             Zero if status is OK.
        relative_deviation : absolute_deviation expressed as a percentage of
                             the optimal range width. Zero if status is OK.
        direction_label    : Human label for the deviation direction
                             (e.g. 'acidic', 'nitrogen deficient').
        alert_summary      : A single, concise, factual alert sentence
                             (injected verbatim into the LLM prompt).
        intervention_hint  : The raw hint string from the schema — NOT a
                             prescription. The LLM uses this to build one.
        parameter_score    : Individual parameter health score 0–100.
                             100 = perfectly centred in optimal range.
    """
    param_key: str
    display_name: str
    unit: str
    observed_value: float
    optimal_min: float
    optimal_max: float
    status: ParameterStatus
    direction: DeviationDirection
    absolute_deviation: float
    relative_deviation: float
    direction_label: str
    alert_summary: str
    intervention_hint: str
    parameter_score: float


@dataclass
class EvaluationReport:
    """
    Complete evaluation output for one telemetry snapshot.

    Attributes:
        crop_profile          : Crop profile key used for evaluation.
        telemetry_snapshot    : The raw input dict (preserved for audit).
        flags                 : List of ParameterFlag — one per evaluated param.
        ok_parameters         : Names of parameters within optimal range.
        warning_parameters    : Names of parameters with WARNING status.
        critical_parameters   : Names of parameters with CRITICAL status.
        missing_parameters    : Names of parameters not present in telemetry.
        field_health_score    : Weighted composite FHS (0–100).
        health_band           : Label string (EXCELLENT / GOOD / FAIR / POOR / CRITICAL).
        health_band_label     : Full human-readable description from the schema.
        health_color          : Hex color code for the health band.
        requires_intervention : True if any CRITICAL or WARNING flags exist.
        llm_alert_block       : Pre-formatted multi-line string of all alerts
                                ready to be injected into the LLM system prompt.
    """
    crop_profile: str
    telemetry_snapshot: dict[str, Any]
    flags: list[ParameterFlag] = field(default_factory=list)
    ok_parameters: list[str] = field(default_factory=list)
    warning_parameters: list[str] = field(default_factory=list)
    critical_parameters: list[str] = field(default_factory=list)
    missing_parameters: list[str] = field(default_factory=list)
    field_health_score: float = 0.0
    health_band: str = "UNKNOWN"
    health_band_label: str = ""
    health_color: str = "#95a5a6"
    requires_intervention: bool = False
    llm_alert_block: str = ""

    # Convenience helpers --------------------------------------------------

    def has_critical_flags(self) -> bool:
        return len(self.critical_parameters) > 0

    def has_any_flags(self) -> bool:
        return len(self.critical_parameters) > 0 or len(self.warning_parameters) > 0

    def flagged_flags(self) -> list[ParameterFlag]:
        """Return only WARNING and CRITICAL ParameterFlag objects."""
        return [f for f in self.flags if f.status in (ParameterStatus.WARNING, ParameterStatus.CRITICAL)]

    def as_summary_dict(self) -> dict[str, Any]:
        """Serializable summary for API responses and logging."""
        return {
            "crop_profile": self.crop_profile,
            "field_health_score": round(self.field_health_score, 1),
            "health_band": self.health_band,
            "requires_intervention": self.requires_intervention,
            "critical_count": len(self.critical_parameters),
            "warning_count": len(self.warning_parameters),
            "critical_parameters": self.critical_parameters,
            "warning_parameters": self.warning_parameters,
            "ok_parameters": self.ok_parameters,
            "missing_parameters": self.missing_parameters,
        }


# ---------------------------------------------------------------------------
# Core Evaluator Class
# ---------------------------------------------------------------------------

class ThresholdEvaluator:
    """
    Stateless (after initialization) evaluator that compares telemetry
    readings against the optimal_thresholds.json ground truth.

    Usage:
        evaluator = ThresholdEvaluator()
        report    = evaluator.evaluate(telemetry_dict, crop="maize_corn")

    The evaluator is intentionally deterministic: same input → same output.
    No LLM calls, no randomness, no side effects after __init__.
    """

    def __init__(self, thresholds_path: Path = _THRESHOLDS_PATH) -> None:
        self._thresholds = self._load_thresholds(thresholds_path)
        logger.info("ThresholdEvaluator initialized. Schema version: %s",
                    self._thresholds.get("_schema_version", "unknown"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        telemetry: dict[str, Any],
        crop: str = "maize_corn",
    ) -> EvaluationReport:
        """
        Evaluate a telemetry snapshot against the crop's optimal thresholds.

        Args:
            telemetry : Dict of {param_key: float_value}. Keys must match the
                        parameter keys in optimal_thresholds.json.
            crop      : Key of the crop profile to use (default: 'maize_corn').

        Returns:
            A fully populated EvaluationReport dataclass.

        Raises:
            KeyError  : If the requested crop profile does not exist.
            TypeError : If a telemetry value cannot be cast to float.
        """
        crop_schema = self._get_crop_schema(crop)
        parameters = crop_schema["parameters"]
        weight_map = crop_schema["composite_health_scoring"]["weights"]
        score_bands = crop_schema["composite_health_scoring"]["score_bands"]

        report = EvaluationReport(
            crop_profile=crop,
            telemetry_snapshot=telemetry,
        )

        weighted_score_sum = 0.0
        total_weight_used = 0.0

        # ---- Per-parameter evaluation loop --------------------------------
        for param_key, param_schema in parameters.items():
            display_name = param_schema["display_name"]
            unit = param_schema["unit"]

            # -- Handle missing sensor data ---------------------------------
            if param_key not in telemetry or telemetry[param_key] is None:
                logger.warning("Parameter '%s' not found in telemetry payload.", param_key)
                report.missing_parameters.append(display_name)
                # Missing params contribute 0 score — penalizes incomplete data
                total_weight_used += weight_map.get(param_key, 0.0)
                continue

            # -- Extract and validate the observed value --------------------
            try:
                observed = float(telemetry[param_key])
            except (ValueError, TypeError) as exc:
                logger.error("Non-numeric value for '%s': %s", param_key, exc)
                report.missing_parameters.append(display_name)
                total_weight_used += weight_map.get(param_key, 0.0)
                continue

            # -- Pull threshold bounds --------------------------------------
            opt_min = float(param_schema["optimal_min"])
            opt_max = float(param_schema["optimal_max"])
            crit_min = float(param_schema["critical_min"])
            crit_max = float(param_schema["critical_max"])

            # -- Classify status --------------------------------------------
            status, direction = self._classify(observed, opt_min, opt_max, crit_min, crit_max)

            # -- Compute deviation metrics ----------------------------------
            abs_dev, rel_dev = self._compute_deviation(observed, opt_min, opt_max)

            # -- Resolve direction-specific labels and hints ----------------
            dir_key = direction.value.lower()  # "low" | "high" | "none"
            direction_labels = param_schema.get("direction_labels", {})
            intervention_hints = param_schema.get("intervention_hints", {})

            direction_label = direction_labels.get(dir_key, "")
            intervention_hint = intervention_hints.get(dir_key, "No specific intervention hint available.")

            # -- Build concise alert summary --------------------------------
            alert_summary = self._build_alert_summary(
                display_name=display_name,
                unit=unit,
                observed=observed,
                opt_min=opt_min,
                opt_max=opt_max,
                status=status,
                direction=direction,
                direction_label=direction_label,
                abs_dev=abs_dev,
                rel_dev=rel_dev,
            )

            # -- Compute per-parameter score (0–100) ------------------------
            param_score = self._compute_parameter_score(
                observed, opt_min, opt_max, crit_min, crit_max
            )

            # -- Accumulate weighted score ----------------------------------
            weight = weight_map.get(param_key, 0.0)
            weighted_score_sum += param_score * weight
            total_weight_used += weight

            # -- Assemble ParameterFlag ------------------------------------
            flag = ParameterFlag(
                param_key=param_key,
                display_name=display_name,
                unit=unit,
                observed_value=observed,
                optimal_min=opt_min,
                optimal_max=opt_max,
                status=status,
                direction=direction,
                absolute_deviation=round(abs_dev, 3),
                relative_deviation=round(rel_dev, 1),
                direction_label=direction_label,
                alert_summary=alert_summary,
                intervention_hint=intervention_hint,
                parameter_score=round(param_score, 1),
            )
            report.flags.append(flag)

            # -- Bucket into OK / WARNING / CRITICAL lists -----------------
            if status == ParameterStatus.OK:
                report.ok_parameters.append(display_name)
            elif status == ParameterStatus.WARNING:
                report.warning_parameters.append(display_name)
            elif status == ParameterStatus.CRITICAL:
                report.critical_parameters.append(display_name)

        # ---- Compute composite Field Health Score -------------------------
        if total_weight_used > 0:
            report.field_health_score = round(weighted_score_sum / total_weight_used, 1)
        else:
            report.field_health_score = 0.0

        # ---- Resolve health band ------------------------------------------
        band_key, band_meta = self._resolve_health_band(
            report.field_health_score, score_bands
        )
        report.health_band = band_key
        report.health_band_label = band_meta.get("label", "")
        report.health_color = band_meta.get("color", "#95a5a6")

        # ---- Set intervention flag ----------------------------------------
        report.requires_intervention = report.has_any_flags()

        # ---- Build the LLM alert block ------------------------------------
        report.llm_alert_block = self._build_llm_alert_block(report)

        logger.info(
            "Evaluation complete | Crop: %s | FHS: %.1f (%s) | "
            "CRITICAL: %d | WARNING: %d | OK: %d | MISSING: %d",
            crop,
            report.field_health_score,
            report.health_band,
            len(report.critical_parameters),
            len(report.warning_parameters),
            len(report.ok_parameters),
            len(report.missing_parameters),
        )
        return report

    def list_supported_crops(self) -> list[str]:
        """Return a list of available crop profile keys."""
        return list(self._thresholds.get("crop_profiles", {}).keys())

    def get_parameter_schema(self, param_key: str, crop: str = "maize_corn") -> dict:
        """Expose the raw JSON schema for a single parameter (useful for the frontend)."""
        crop_schema = self._get_crop_schema(crop)
        parameters = crop_schema["parameters"]
        if param_key not in parameters:
            raise KeyError(f"Parameter '{param_key}' not found in crop profile '{crop}'.")
        return parameters[param_key]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_thresholds(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"Thresholds file not found at: {path}\n"
                "Ensure optimal_thresholds.json is in backend/data/"
            )
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug("Loaded thresholds from %s", path)
        return data

    def _get_crop_schema(self, crop: str) -> dict:
        profiles = self._thresholds.get("crop_profiles", {})
        if crop not in profiles:
            available = list(profiles.keys())
            raise KeyError(
                f"Crop profile '{crop}' not found. "
                f"Available profiles: {available}"
            )
        return profiles[crop]

    @staticmethod
    def _classify(
        observed: float,
        opt_min: float,
        opt_max: float,
        crit_min: float,
        crit_max: float,
    ) -> tuple[ParameterStatus, DeviationDirection]:
        """
        Map an observed value to (ParameterStatus, DeviationDirection).

        Classification priority (most severe wins):
            1. CRITICAL — outside critical bounds
            2. WARNING  — outside optimal bounds but inside critical bounds
            3. OK       — inside optimal bounds (direction = NONE)
        """
        if observed <= crit_min:
            return ParameterStatus.CRITICAL, DeviationDirection.LOW
        if observed >= crit_max:
            return ParameterStatus.CRITICAL, DeviationDirection.HIGH
        if observed < opt_min:
            return ParameterStatus.WARNING, DeviationDirection.LOW
        if observed > opt_max:
            return ParameterStatus.WARNING, DeviationDirection.HIGH
        return ParameterStatus.OK, DeviationDirection.NONE

    @staticmethod
    def _compute_deviation(
        observed: float,
        opt_min: float,
        opt_max: float,
    ) -> tuple[float, float]:
        """
        Compute absolute and relative deviation from the optimal range.

        Absolute deviation: distance from the nearest optimal boundary.
            - Zero when observed is inside [opt_min, opt_max].
        Relative deviation: absolute_deviation as a % of (opt_max - opt_min).
            - Represents 'how many optimal-range widths outside' the reading is.

        Returns:
            (absolute_deviation, relative_deviation_percent)
        """
        optimal_range = opt_max - opt_min
        if optimal_range <= 0:
            return 0.0, 0.0

        if observed < opt_min:
            abs_dev = opt_min - observed
        elif observed > opt_max:
            abs_dev = observed - opt_max
        else:
            return 0.0, 0.0

        rel_dev = (abs_dev / optimal_range) * 100.0
        return abs_dev, rel_dev

    @staticmethod
    def _compute_parameter_score(
        observed: float,
        opt_min: float,
        opt_max: float,
        crit_min: float,
        crit_max: float,
    ) -> float:
        """
        Compute a continuous 0–100 score for a single parameter.

        Scoring model:
          - Score = 100 when the observed value is exactly at the optimal midpoint.
          - Score decays linearly from 100 → 30 as the value moves from the
            optimal boundary to the critical boundary.
          - Score decays linearly from 30 → 0 as the value moves beyond the
            critical boundary (penalises extreme exceedances).

        This produces a smooth, monotonic decay on each side of the optimal
        window — suitable for weighted averaging into the FHS.
        """
        opt_mid = (opt_min + opt_max) / 2.0

        # Value is inside the optimal range: score interpolates 70–100
        if opt_min <= observed <= opt_max:
            half_opt = (opt_max - opt_min) / 2.0
            if half_opt == 0:
                return 100.0
            dist_from_mid = abs(observed - opt_mid)
            # At midpoint → 100; at boundary → 70
            score = 100.0 - (dist_from_mid / half_opt) * 30.0
            return max(70.0, score)

        # Value is between optimal and critical bounds: score 30–70
        if observed < opt_min:
            opt_boundary = opt_min
            crit_boundary = crit_min
        else:
            opt_boundary = opt_max
            crit_boundary = crit_max

        crit_range = abs(opt_boundary - crit_boundary)
        if crit_range == 0:
            return 30.0

        dist_from_opt = abs(observed - opt_boundary)
        if dist_from_opt <= crit_range:
            score = 70.0 - (dist_from_opt / crit_range) * 40.0
            return max(30.0, score)

        # Value is beyond the critical boundary: score 0–30
        beyond_crit = dist_from_opt - crit_range
        # Decay over one additional critical_range width
        decay_range = crit_range if crit_range > 0 else 1.0
        score = 30.0 - (beyond_crit / decay_range) * 30.0
        return max(0.0, score)

    @staticmethod
    def _build_alert_summary(
        display_name: str,
        unit: str,
        observed: float,
        opt_min: float,
        opt_max: float,
        status: ParameterStatus,
        direction: DeviationDirection,
        direction_label: str,
        abs_dev: float,
        rel_dev: float,
    ) -> str:
        """
        Build a single, dense, factual alert string for LLM injection.

        Format:
            [CRITICAL|WARNING] {display_name}: Observed {X} {unit}
            (optimal: {min}–{max} {unit}). {deviation} {direction_label} —
            {relative_deviation}% outside optimal range.

        Example:
            [CRITICAL] Soil pH: Observed 4.7 pH units (optimal: 5.8–7.0). 
            1.10 units acidic — 91.7% outside optimal range.
        """
        if status == ParameterStatus.OK:
            return (
                f"[OK] {display_name}: {observed} {unit} "
                f"(within optimal range {opt_min}–{opt_max} {unit})."
            )

        label_suffix = f" — {direction_label}" if direction_label else ""
        return (
            f"[{status.value}] {display_name}: Observed {observed} {unit} "
            f"(optimal: {opt_min}–{opt_max} {unit}). "
            f"Deviation: {abs_dev:.3f} {unit} {direction.value.lower()}{label_suffix} "
            f"({rel_dev:.1f}% outside optimal range)."
        )

    @staticmethod
    def _resolve_health_band(
        score: float, score_bands: dict
    ) -> tuple[str, dict]:
        """
        Find which score band the FHS falls into.
        Returns (band_key, band_metadata_dict).
        Falls back to CRITICAL band if no match found.
        """
        for band_key, meta in score_bands.items():
            if meta["min"] <= score <= meta["max"]:
                return band_key, meta
        # Fallback — should never trigger if schema is valid
        logger.error("FHS %.1f fell outside all defined score bands.", score)
        return "CRITICAL", score_bands.get("CRITICAL", {})

    @staticmethod
    def _build_llm_alert_block(report: EvaluationReport) -> str:
        """
        Assemble a structured text block containing all evaluation results,
        formatted for clean injection into the LLM system prompt.

        The block is intentionally terse and factual — it is NOT a prescription.
        The LLM will use this as the exclusive source of truth.
        """
        lines: list[str] = []

        lines.append("=== FIELD EVALUATION REPORT ===")
        lines.append(f"Crop Profile  : {report.crop_profile}")
        lines.append(f"Field Health  : {report.field_health_score}/100 ({report.health_band})")
        lines.append(f"Status        : {'INTERVENTION REQUIRED' if report.requires_intervention else 'ALL PARAMETERS NOMINAL'}")
        lines.append("")

        if report.critical_parameters:
            lines.append("--- CRITICAL FLAGS (Immediate action required) ---")
            for flag in report.flags:
                if flag.status == ParameterStatus.CRITICAL:
                    lines.append(f"  • {flag.alert_summary}")
                    lines.append(f"    Hint: {flag.intervention_hint}")
            lines.append("")

        if report.warning_parameters:
            lines.append("--- WARNING FLAGS (Action recommended) ---")
            for flag in report.flags:
                if flag.status == ParameterStatus.WARNING:
                    lines.append(f"  • {flag.alert_summary}")
                    lines.append(f"    Hint: {flag.intervention_hint}")
            lines.append("")

        if report.ok_parameters:
            lines.append("--- NOMINAL PARAMETERS ---")
            for flag in report.flags:
                if flag.status == ParameterStatus.OK:
                    lines.append(f"  ✓ {flag.display_name}: {flag.observed_value} {flag.unit}")
            lines.append("")

        if report.missing_parameters:
            lines.append("--- MISSING SENSOR DATA ---")
            for name in report.missing_parameters:
                lines.append(f"  ? {name}: No reading received")
            lines.append("")

        lines.append("=== END EVALUATION REPORT ===")
        return "\n".join(lines)
