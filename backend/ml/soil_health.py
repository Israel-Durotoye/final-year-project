"""
soil_health.py — Crop-Aware Soil Suitability Scoring / Labeling

Single source of truth for turning a raw sensor reading into a soil-suitability
verdict for the crop a node is dedicated to.

It is used in three places, and MUST behave identically in all of them so the
model's learned target and the live threshold verdict never drift apart:

    1. lstm_suitability_trainer.py  — to LABEL training sequences.
    2. backend/api/routes/ml.py     — to return an auditable threshold verdict
                                      alongside the LSTM prediction.
    3. backend/rag/chat_llm.py      — as a fallback verdict when the trained
                                      model is unavailable.

Scoring model
-------------
Each measured parameter is scored 0-100 against the crop's optimal / critical
range from ``optimal_thresholds.json``:

    * 100 when the value sits inside [optimal_min, optimal_max].
    * Decays linearly from 100 (at the optimal bound) to 0 (at the critical
      bound) in the sub-optimal shoulder.
    * 0 beyond the critical bound.

The composite score is the weighted mean over the parameters actually present in
the reading (the sensor measures only 6 of the profile's parameters; pH, EC and
organic matter are not measured and are simply skipped). Composite score is then
banded into a 3-class label: Good / Fair / Poor.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns — identical order to lstm_anomaly_trainer.FEATURES so the
# MinMaxScaler feature axis stays consistent across the ML pipeline.
# ---------------------------------------------------------------------------

FEATURES: list[str] = [
    "Nitrogen_mg_k",
    "Phosphorus_m",
    "Potassium_mg_",
    "Moisture_%",
    "Temperature_C",
    "Humidity_%",
]

# Map each dataset column to its parameter key inside optimal_thresholds.json.
# Columns without a corresponding profile parameter are simply not scored.
PARAM_MAP: dict[str, str] = {
    "Nitrogen_mg_k": "nitrogen_ppm",
    "Phosphorus_m": "phosphorus_ppm",
    "Potassium_mg_": "potassium_ppm",
    "Moisture_%": "soil_moisture",
    "Temperature_C": "soil_temperature",
    "Humidity_%": "ambient_humidity",
}

# ---------------------------------------------------------------------------
# Label encoding — shared by trainer (targets) and inference (index -> name).
# Index order is fixed: 0=Poor, 1=Fair, 2=Good.
# ---------------------------------------------------------------------------

LABELS: list[str] = ["Poor", "Fair", "Good"]

# Composite-score band edges (3-class collapse of the JSON's 5 EXCELLENT..CRITICAL
# bands). Tune these two constants to reshape the classes without retraining logic.
GOOD_MIN: float = 70.0   # score >= GOOD_MIN            -> Good
FAIR_MIN: float = 45.0   # FAIR_MIN <= score < GOOD_MIN -> Fair
#                          score < FAIR_MIN             -> Poor

# Default crop profile used when a node's Target_Crop has no dedicated profile.
DEFAULT_CROP_KEY: str = "maize_corn"

# ---------------------------------------------------------------------------
# Threshold profile loading (path-relative, mirrors lstm_inference.py idiom)
# ---------------------------------------------------------------------------

_ML_DIR = pathlib.Path(__file__).parent.resolve()
_THRESHOLDS_PATH = _ML_DIR.parent / "data" / "optimal_thresholds.json"

_profiles_cache: Optional[dict[str, Any]] = None
_unknown_crops_seen: set[str] = set()


def _load_profiles() -> dict[str, Any]:
    """Load and cache the crop_profiles block from optimal_thresholds.json."""

    global _profiles_cache

    if _profiles_cache is not None:
        return _profiles_cache

    if not _THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Optimal thresholds file not found at: {_THRESHOLDS_PATH}"
        )

    with _THRESHOLDS_PATH.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    profiles = payload.get("crop_profiles") or {}

    if not profiles:
        raise ValueError(
            "optimal_thresholds.json contains no crop_profiles."
        )

    _profiles_cache = profiles
    return profiles


# ---------------------------------------------------------------------------
# Crop name normalisation
# ---------------------------------------------------------------------------

# Aliases mapping common Target_Crop spellings to a profile key.
_CROP_ALIASES: dict[str, str] = {
    "maize": "maize_corn",
    "corn": "maize_corn",
    "maize_corn": "maize_corn",
    "maize/corn": "maize_corn",
}


def normalize_crop(target_crop: Optional[str]) -> str:
    """
    Resolve a raw Target_Crop value to a profile key present in the thresholds.

    Unknown or missing crops fall back to DEFAULT_CROP_KEY and are logged once.
    """

    profiles = _load_profiles()

    raw = (target_crop or "").strip().lower()

    if not raw:
        return DEFAULT_CROP_KEY if DEFAULT_CROP_KEY in profiles else next(iter(profiles))

    # Direct profile-key match.
    if raw in profiles:
        return raw

    # Alias match.
    if raw in _CROP_ALIASES and _CROP_ALIASES[raw] in profiles:
        return _CROP_ALIASES[raw]

    # Loose match: normalise separators (e.g. "sweet corn" -> "sweet_corn").
    collapsed = raw.replace("/", "_").replace(" ", "_").replace("-", "_")
    if collapsed in profiles:
        return collapsed
    if collapsed in _CROP_ALIASES and _CROP_ALIASES[collapsed] in profiles:
        return _CROP_ALIASES[collapsed]

    fallback = DEFAULT_CROP_KEY if DEFAULT_CROP_KEY in profiles else next(iter(profiles))

    if raw not in _unknown_crops_seen:
        _unknown_crops_seen.add(raw)
        logger.warning(
            "No threshold profile for crop '%s'; falling back to '%s'.",
            target_crop,
            fallback,
        )

    return fallback


# ---------------------------------------------------------------------------
# Per-parameter and composite scoring
# ---------------------------------------------------------------------------

def _score_parameter(value: float, param_spec: dict[str, Any]) -> Optional[float]:
    """
    Score a single parameter value 0-100 against its optimal/critical range.

    Returns None if the spec lacks the bounds needed to score.
    """

    try:
        opt_min = float(param_spec["optimal_min"])
        opt_max = float(param_spec["optimal_max"])
        crit_min = float(param_spec["critical_min"])
        crit_max = float(param_spec["critical_max"])
    except (KeyError, TypeError, ValueError):
        return None

    # Inside the optimal band -> full marks.
    if opt_min <= value <= opt_max:
        return 100.0

    # Below optimal: decay from 100 at opt_min down to 0 at crit_min.
    if value < opt_min:
        if value <= crit_min or opt_min <= crit_min:
            return 0.0 if value <= crit_min else 100.0
        fraction = (value - crit_min) / (opt_min - crit_min)
        return max(0.0, min(100.0, fraction * 100.0))

    # Above optimal: decay from 100 at opt_max down to 0 at crit_max.
    if value >= crit_max or crit_max <= opt_max:
        return 0.0 if value >= crit_max else 100.0
    fraction = (crit_max - value) / (crit_max - opt_max)
    return max(0.0, min(100.0, fraction * 100.0))


def _band_label(score: float) -> str:
    """Collapse a 0-100 composite score into a Good/Fair/Poor label."""

    if score >= GOOD_MIN:
        return "Good"
    if score >= FAIR_MIN:
        return "Fair"
    return "Poor"


def score_reading(
    row: dict[str, Any],
    crop: Optional[str],
) -> tuple[str, float, dict[str, float]]:
    """
    Score one sensor reading against its crop's optimal thresholds.

    Args:
        row:  A reading dict using dataset column names (see FEATURES).
        crop: The raw Target_Crop value (resolved via normalize_crop()).

    Returns:
        (label, composite_score, per_parameter_scores)
        where label is one of LABELS and composite_score is 0-100.
    """

    profiles = _load_profiles()
    crop_key = normalize_crop(crop)
    parameters = profiles.get(crop_key, {}).get("parameters", {})
    weights = (
        profiles.get(crop_key, {})
        .get("composite_health_scoring", {})
        .get("weights", {})
    )

    per_param: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for column, param_key in PARAM_MAP.items():
        if column not in row or row[column] is None:
            continue

        param_spec = parameters.get(param_key)
        if not param_spec:
            continue

        try:
            value = float(row[column])
        except (TypeError, ValueError):
            continue

        sub_score = _score_parameter(value, param_spec)
        if sub_score is None:
            continue

        weight = float(weights.get(param_key, 1.0))
        per_param[param_key] = round(sub_score, 2)
        weighted_sum += sub_score * weight
        weight_total += weight

    if weight_total == 0.0:
        # No scorable parameters — treat as worst case rather than fabricate.
        return "Poor", 0.0, per_param

    composite = weighted_sum / weight_total
    return _band_label(composite), round(composite, 2), per_param


def label_index(label: str) -> int:
    """Return the class index (0/1/2) for a Good/Fair/Poor label."""

    return LABELS.index(label)


def label_map() -> dict[int, str]:
    """Return the class-index -> label-name map (persisted with the model)."""

    return {index: name for index, name in enumerate(LABELS)}
