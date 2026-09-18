"""Focused evidence and writing contract for the dashboard's field report."""

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

from backend.rag.prescriptions import FarmRecommendationPlanner

logger = logging.getLogger(__name__)
EVIDENCE_REPORT_NOTE = "*Assessment method: sensor evidence and configured screening rules. AI-written interpretation was unavailable.*"

FIELD_SUMMARY_INSTRUCTION = """FIELD SUMMARY MODE
Write exactly ONE short sentence (20–40 words, at most 45) in plain language for
a farmer, covering only the most important condition or change in the selected
field area during the supplied reporting period, plus one practical action if
needed. If no action is justified, say to continue monitoring without inventing
a problem. Use the latest reading and observed trends within reporting_period;
do not call a single reading a trend or invent a duration such as today or this
week. Mention a future estimate only when available and relevant to an immediate
decision, using uncertain language. Old or missing readings take priority over
agronomic advice. Do not infer stability from a missing forecast.
No headings, bullets, greeting, source/method notes, sensor list, generic seasonal
tips, or unrelated limitations. Include a measurement only if it explains the
main finding. Do not invent crops, causes, rainfall, diagnoses or treatment rates.
"""


FIELD_REPORT_INSTRUCTION = """FIELD REPORT MODE
Write a professional field assessment of the selected node, approximately 250–400
words when evidence supports that detail. Use these four Markdown headings:
### Overall assessment
Explain the main finding and its significance in two or three sentences.
### Supporting observations
Connect the important measurements (with their units) to the assessment. Identify
the crop if known, and distinguish observed history from uncertain forecasts.
### Recommended actions
Give two to four specific, prioritized steps justified by the evidence. Explain
their purpose. If no correction is justified, say so and give monitoring steps.
### What to monitor
Describe relevant changes or field signs to watch and material data limitations.

Use only the supplied evidence. Unknown crop, missing measurements, old readings,
and unavailable forecasts must stay explicit; absence of a forecast never means
conditions are stable. Generic screening is provisional, not a crop diagnosis.
Never invent a crop, disease, weather event, trend, treatment rate or watering
schedule. Do not fill space by repeating readings or advice. No greeting or
discussion of these instructions. Keep the four sections even with limited data,
but make the report shorter when there is insufficient evidence.
"""

REPORT_SECTIONS = (
    ("Overall assessment", "Explain the main finding and its significance in 2–3 sentences. State uncertainty if evidence is limited."),
    ("Supporting observations", "Explain 2–3 important measured observations with units. Distinguish observations from forecasts; do not list every sensor."),
    ("Recommended actions", "Give 2–4 practical steps supported by the supplied priority brief and guidance. Explain why. Do not invent treatment rates."),
    ("What to monitor", "Describe relevant changes or field signs to watch. State missing crop information, stale data or unavailable forecasts when applicable."),
)


def build_report_evidence(snapshot: dict[str, Any], temporal: dict[str, Any], node_id: str) -> dict[str, Any]:
    """Keep unrelated nodes and verbose model metadata out of the report input."""
    node = next((item for item in snapshot.get("nodes", [])
                 if str(item.get("node_id", "")).upper() == node_id.upper()), None)
    result = (temporal.get("nodes") or {}).get(node_id, {})
    quality = dict(result.get("data_quality") or {})
    # History can fail independently of the current snapshot. Still expose the
    # age of that snapshot instead of silently describing old readings as live.
    if node and "stale" not in quality:
        try:
            timestamp = datetime.fromisoformat(str(node.get("timestamp_utc")).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - timestamp).total_seconds() / 60
            quality.update(latest_sample_age_minutes=round(age, 1), stale=age > 30)
        except (ValueError, TypeError):
            quality["timestamp_status"] = "unknown"
    evidence: dict[str, Any] = {
        "selected_node": node_id,
        "current_measurements": node or {"status": "unavailable"},
        "data_quality": quality or {"status": "unknown"},
        "reporting_period": result.get("history") or {
            "scope": "latest available reading only",
            "end": node.get("timestamp_utc") if node else None,
        },
        "observed_history": result.get("historical_analysis") or {"status": "unavailable"},
        "forecast_status": result.get("forecast_status", "unavailable"),
        "future_estimates": result.get("forecast"),
    }
    if node:
        try:
            brief = FarmRecommendationPlanner().build_node_brief(node)
            evidence["priority_brief"] = {
                "screening_basis": brief["screening_basis"],
                "priorities": [{key: item[key] for key in (
                    "domain", "label", "evidence", "supporting_steps", "monitor", "context_limits",
                )} for item in brief["priorities"][:2]],
                "preventive_focus": brief["preventive_focus_if_no_actionable_problem"] if not brief["priorities"] else [],
            }
        except Exception:
            logger.warning("Field report priority brief unavailable", exc_info=True)
            evidence["priority_brief"] = {"status": "unavailable"}
    return evidence


def report_content(evidence: dict[str, Any], knowledge: str, *, summary: bool = False) -> str:
    return (
        ("FIELD SUMMARY EVIDENCE\n" if summary else "FIELD REPORT EVIDENCE\n")
        + json.dumps(evidence, ensure_ascii=False, default=str)
        + "\n\nSUPPORTING GUIDANCE\n" + (knowledge or "No relevant guidance retrieved.")
    )


def normalize_field_summary(answer: str) -> str | None:
    """Require a complete, short sentence rather than chopping off an action."""
    cleaned = answer.strip().strip('"').replace("**", "")
    if any(token in cleaned for token in ("\n", "#", "{", "}", "[", "]", "`")):
        return None
    cleaned = " ".join(cleaned.split())
    if not 6 <= len(cleaned.split()) <= 45 or not cleaned.endswith((".", "!")):
        return None
    # Decimal points in readings are not sentence boundaries.
    if re.search(r"[.!?](?!\d)", cleaned[:-1]):
        return None
    return cleaned


def render_field_summary(evidence: dict[str, Any]) -> str:
    """Single-sentence fallback; only report supported conditions or trends."""
    node = evidence.get("current_measurements") or {}
    if node.get("status") == "unavailable":
        return "Recent field readings are unavailable, so check the sensor connection before deciding whether any action is needed."
    if (evidence.get("data_quality") or {}).get("stale"):
        return "The latest field readings are out of date, so obtain fresh readings before changing irrigation or applying treatments."
    priorities = (evidence.get("priority_brief") or {}).get("priorities") or []
    if priorities:
        domain = priorities[0].get("domain")
        summaries = {
            "excess_water": "The latest readings suggest the soil is too wet, so check for standing water and clear drainage before adding more irrigation.",
            "water_deficit": "The latest readings suggest the soil is too dry, so check moisture around the roots before adjusting irrigation.",
            "temperature_stress": "The latest readings suggest soil temperature needs attention, so inspect the crop for stress before deciding on a correction.",
            "nutrient_verification": "The latest nutrient readings need checking against the crop's requirements before changing fertiliser application.",
            "soil_reaction": "The latest soil acidity reading needs attention, so confirm the crop's requirements before applying amendments.",
        }
        return summaries.get(domain, "The latest readings flag a possible soil condition problem, so inspect the field and verify the readings before applying a correction.")
    moisture = (evidence.get("observed_history") or {}).get("moisture") or {}
    trend = moisture.get("trend")
    if trend in {"falling", "strongly_falling", "rising", "strongly_rising"}:
        direction = "falling" if "falling" in trend else "rising"
        return f"Soil moisture has been {direction} over the recorded period, so check the root zone and subsequent readings before changing irrigation."
    return "The latest readings do not identify an immediate correction, so continue monitoring the field for changes."


def usable_local_report(sections: list[str]) -> bool:
    """Reject fragments, copied JSON and repetitive output from tiny models.

    This is a presentation check, not a claim to verify arbitrary model facts.
    """
    if len(sections) != 4 or sum(len(section.split()) for section in sections) < 120:
        return False
    for section in sections:
        words = section.lower().split()
        if len(words) < 12 or any(token in section for token in ('{', '}', '["', '":', '###')):
            return False
        phrases = [tuple(words[index:index + 8]) for index in range(len(words) - 7)]
        if len(phrases) != len(set(phrases)):
            return False
    return True


def render_evidence_report(evidence: dict[str, Any]) -> str:
    """Readable fallback assembled from measured facts and existing screening rules.

    It deliberately does not pretend to be an LLM interpretation or a forecast.
    """
    node_id = str(evidence["selected_node"])
    node = evidence.get("current_measurements") or {}
    quality = evidence.get("data_quality") or {}
    brief = evidence.get("priority_brief") or {}
    priorities = brief.get("priorities") or []
    stale = quality.get("stale") is True
    unavailable = node.get("status") == "unavailable"

    if unavailable:
        assessment = f"Current measurements for {node_id} are unavailable. A reliable field-condition assessment cannot be made from the available evidence."
    elif stale:
        assessment = f"The latest available readings for {node_id} are out of date. They describe the last recorded conditions and should not be used as a current field assessment."
    elif priorities:
        labels = "; ".join(str(item["label"]).lower() for item in priorities)
        assessment = f"Screening of {node_id} identifies {labels} as the main management priority. Confirm these indications through field observations before attributing them to a specific cause of crop stress."
    else:
        assessment = f"The available screening checks for {node_id} identify no priority corrective action. This does not establish overall crop health or confirm that unmeasured conditions are suitable."

    observations = []
    if not unavailable:
        observations.append(f"Latest reading: {node.get('timestamp_utc') or 'timestamp unavailable'}. Crop: {node.get('currently_planted_crop') or 'not recorded'}.")
    measurements = []
    for key, label, unit in (
        ("moisture_pct", "soil moisture", "%"), ("temperature_c", "temperature", "°C"),
        ("humidity_pct", "humidity", "%"), ("nitrogen_mg_kg", "nitrogen", " mg/kg"),
        ("phosphorus_mg_kg", "phosphorus", " mg/kg"), ("potassium_mg_kg", "potassium", " mg/kg"),
        ("ph", "pH", ""),
    ):
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            measurements.append(f"{label} {value:g}{unit}")
    if measurements:
        observations.append("Recorded measurements: " + "; ".join(measurements) + ".")
    if brief.get("screening_basis"):
        observations.append("Interpretation basis: " + brief["screening_basis"] + ".")
    if not observations:
        observations.append("No usable current measurements were retrieved for the selected node.")

    if unavailable or stale:
        actions = ["Check the selected node's connection and obtain a fresh set of readings.",
                   "Inspect the field before changing irrigation or applying amendments on the basis of this report."]
    else:
        actions = list(dict.fromkeys(step for item in priorities for step in item.get("supporting_steps", [])))[:4]
        if not actions:
            actions = brief.get("preventive_focus") or ["Inspect sensor placement and collect further readings before deciding on a correction."]

    monitor = list(dict.fromkeys(sign for item in priorities for sign in item.get("monitor", [])))
    monitoring = ["Watch " + ", ".join(monitor) + "; compare subsequent readings with field observations." if monitor
                  else "Compare subsequent readings and field observations before concluding that conditions are stable."]
    history = evidence.get("observed_history") or {}
    observed_trends = [f"{key.replace('_', ' ')}: {metrics['trend']}"
                       for key, metrics in history.items()
                       if isinstance(metrics, dict) and metrics.get("trend") not in (None, "unknown")]
    if observed_trends:
        monitoring.append("Observed historical trends (not predictions): " + "; ".join(observed_trends) + ".")
    else:
        monitoring.append("Historical trends are unavailable, so improvement or deterioration cannot be established here.")
    if evidence.get("forecast_status") != "success" or not evidence.get("future_estimates"):
        monitoring.append("No usable forecast is available; future conditions have not been assessed.")
    else:
        monitoring.append("Model estimates are available but are uncertain; this screening report does not interpret them as confirmed outcomes.")
    if not node.get("currently_planted_crop"):
        monitoring.append("The planted crop is not recorded, which limits crop-specific interpretation.")
    if not unavailable:
        monitoring.append("Exact irrigation volumes and amendment rates are not established by these readings alone.")

    report = "\n\n".join((
        f"### Overall assessment\n{assessment}",
        "### Supporting observations\n" + "\n".join(f"- {line}" for line in observations),
        "### Recommended actions\n" + "\n".join(f"{index}. {line}" for index, line in enumerate(actions, 1)),
        "### What to monitor\n" + " ".join(monitoring),
    ))
    return report + "\n\n" + EVIDENCE_REPORT_NOTE
