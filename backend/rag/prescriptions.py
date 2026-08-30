"""
prescriptions.py — Soil Management Action Recommendation Engine

Generates prioritized corrective actions and interventions based on
soil diagnoses. Actions are ranked by impact and dependencies.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.prescriptive.evaluator import ThresholdEvaluator
from backend.rag import diagnostics, thresholds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prescription Output Models
# ---------------------------------------------------------------------------

@dataclass
class CorrectiveAction:
    """A single recommended action."""
    action: str                          # E.g., "Apply agricultural lime"
    target_parameter: str                # E.g., "pH"
    priority: int                        # 1 = highest priority, ascending
    severity: thresholds.Severity        # Severity of the issue being addressed
    impact: str                          # Expected outcome ("High", "Moderate", "Low")
    dosage: Optional[str] = None        # Recommended rate (e.g., "2–3 tons/ha")
    timeline: Optional[str] = None      # When to apply (e.g., "Immediate", "Before next planting")
    interaction_dependencies: list[str] = field(default_factory=list)  # Other actions that must happen first
    reasoning: str = ""                  # Why this action is recommended


@dataclass
class ActionPlan:
    """Complete soil management action plan."""
    diagnosis_id: str
    diagnosis_timestamp: str
    issues_count: int
    corrective_actions: list[CorrectiveAction]  # Ranked by priority
    expected_timeline: str                       # E.g., "1 month" or "Full season"
    critical_first_steps: list[str]             # Top 3–5 actions to start immediately
    monitoring_parameters: list[str]            # What to monitor after interventions


# ---------------------------------------------------------------------------
# Prescription Engine
# ---------------------------------------------------------------------------

class PrescriptionEngine:
    """
    Generates prioritized corrective actions from soil diagnoses.
    
    Usage:
        engine = PrescriptionEngine()
        plan = engine.generate_action_plan(diagnosis)
    """

    def __init__(self):
        """Initialize prescription engine."""
        self.action_registry = self._build_action_registry()

    def generate_action_plan(
        self,
        diagnosis: diagnostics.SoilDiagnosis,
        diagnosis_id: str = "default",
    ) -> ActionPlan:
        """
        Generate a prioritized action plan from a soil diagnosis.
        
        Args:
            diagnosis: SoilDiagnosis object from the diagnostic engine.
            diagnosis_id: Unique identifier for this diagnosis.
        
        Returns:
            ActionPlan with ranked actions and expected timeline.
        """
        actions = []

        # Generate actions for each diagnosed issue
        for issue in diagnosis.issues:
            issue_actions = self._prescribe_actions_for_issue(issue)
            actions.extend(issue_actions)

        # Remove duplicates (same action for multiple issues)
        unique_actions = self._deduplicate_actions(actions)

        # Rank actions by priority considering dependencies and interactions
        ranked_actions = self._rank_actions(unique_actions, diagnosis)

        # Identify critical first steps
        critical_steps = [a.action for a in ranked_actions[:3]]

        # Estimate timeline
        timeline = self._estimate_timeline(diagnosis.severity_summary)

        # Identify monitoring parameters
        monitoring = self._monitoring_parameters_for_issues(diagnosis.issues)

        return ActionPlan(
            diagnosis_id=diagnosis_id,
            diagnosis_timestamp=diagnosis.timestamp,
            issues_count=len(diagnosis.issues),
            corrective_actions=ranked_actions,
            expected_timeline=timeline,
            critical_first_steps=critical_steps,
            monitoring_parameters=monitoring,
        )

    # ──────────────────────────────────────────────────────────────────
    # Private: Action Generation
    # ──────────────────────────────────────────────────────────────────

    def _prescribe_actions_for_issue(self, issue: diagnostics.SoilIssue) -> list[CorrectiveAction]:
        """Generate corrective actions for a specific soil issue."""
        actions = []

        if issue.parameter == "pH":
            if issue.measured_value < issue.optimal_range[0]:
                # Acidic soil — apply lime
                dosage = self._calculate_lime_dosage(issue.measured_value, issue.optimal_range[0])
                actions.append(CorrectiveAction(
                    action="Apply agricultural lime (CaCO₃)",
                    target_parameter="pH",
                    priority=1 if issue.severity == thresholds.Severity.CRITICAL else 2,
                    severity=issue.severity,
                    impact="High",
                    dosage=dosage,
                    timeline="Before next planting cycle",
                    reasoning="Lime neutralizes soil acidity and raises pH to optimal range for nutrient availability."
                ))
            else:
                # Alkaline soil — apply sulfur
                dosage = self._calculate_sulfur_dosage(issue.measured_value, issue.optimal_range[1])
                actions.append(CorrectiveAction(
                    action="Apply elemental sulfur (S)",
                    target_parameter="pH",
                    priority=1 if issue.severity == thresholds.Severity.CRITICAL else 2,
                    severity=issue.severity,
                    impact="High",
                    dosage=dosage,
                    timeline="2–3 weeks before planting",
                    reasoning="Sulfur oxidation lowers pH. Slow-acting but long-lasting."
                ))

        elif issue.parameter == "Nitrogen":
            if issue.measured_value < issue.optimal_range[0]:
                dosage = self._calculate_fertilizer_dosage("N", issue.measured_value, issue.optimal_range[0])
                actions.append(CorrectiveAction(
                    action="Apply nitrogen fertilizer (urea or ammonium nitrate)",
                    target_parameter="Nitrogen",
                    priority=3,  # Usually lower priority if pH is also an issue
                    severity=issue.severity,
                    impact="High",
                    dosage=dosage,
                    timeline="At planting or early vegetative stage",
                    interaction_dependencies=["pH correction must be completed first if acidic."],
                    reasoning="Nitrogen is mobile and readily leached. Apply in split doses during growing season."
                ))

        elif issue.parameter == "Phosphorus":
            if issue.measured_value < issue.optimal_range[0]:
                dosage = self._calculate_fertilizer_dosage("P", issue.measured_value, issue.optimal_range[0])
                actions.append(CorrectiveAction(
                    action="Apply phosphorus fertilizer (superphosphate or rock phosphate)",
                    target_parameter="Phosphorus",
                    priority=2,
                    severity=issue.severity,
                    impact="High",
                    dosage=dosage,
                    timeline="Pre-plant or at planting",
                    reasoning="Phosphorus is relatively immobile. Apply before planting for full-season availability."
                ))

        elif issue.parameter == "Potassium":
            if issue.measured_value < issue.optimal_range[0]:
                dosage = self._calculate_fertilizer_dosage("K", issue.measured_value, issue.optimal_range[0])
                actions.append(CorrectiveAction(
                    action="Apply potassium fertilizer (muriate of potash or potassium nitrate)",
                    target_parameter="Potassium",
                    priority=2,
                    severity=issue.severity,
                    impact="Moderate",
                    dosage=dosage,
                    timeline="At planting or early growth",
                    reasoning="Potassium supports plant vigor and disease resistance. Apply in split doses if high rates needed."
                ))

        elif issue.parameter == "Moisture":
            if issue.measured_value < issue.optimal_range[0]:
                actions.append(CorrectiveAction(
                    action="Increase irrigation frequency or duration",
                    target_parameter="Moisture",
                    priority=1 if issue.severity == thresholds.Severity.CRITICAL else 2,
                    severity=issue.severity,
                    impact="High",
                    dosage="Target 20–25% volumetric water content",
                    timeline="Immediate, maintain throughout growing season",
                    reasoning="Adequate moisture is critical for nutrient uptake and plant function."
                ))
                actions.append(CorrectiveAction(
                    action="Apply organic mulch (2–4 cm layer)",
                    target_parameter="Moisture",
                    priority=3,
                    severity=issue.severity,
                    impact="Moderate",
                    dosage="2–4 cm of organic material (straw, compost)",
                    timeline="After rain or irrigation",
                    reasoning="Mulch reduces evaporation and maintains consistent soil moisture."
                ))
            else:
                actions.append(CorrectiveAction(
                    action="Improve soil drainage (raised beds, drainage tiles, or aeration)",
                    target_parameter="Moisture",
                    priority=1 if issue.severity == thresholds.Severity.CRITICAL else 2,
                    severity=issue.severity,
                    impact="High",
                    timeline="Before next planting cycle",
                    reasoning="Excess water suffocates roots. Drainage is structural and must be fixed before planting."
                ))

        elif issue.parameter == "Temperature":
            if issue.measured_value < issue.optimal_range[0]:
                actions.append(CorrectiveAction(
                    action="Mulch soil surface to retain heat",
                    target_parameter="Temperature",
                    priority=3,
                    severity=issue.severity,
                    impact="Moderate",
                    reasoning="Dark mulch absorbs and retains solar heat."
                ))
            else:
                actions.append(CorrectiveAction(
                    action="Increase irrigation and apply reflective mulch",
                    target_parameter="Temperature",
                    priority=3,
                    severity=issue.severity,
                    impact="Moderate",
                    reasoning="Water and light-colored mulch cool the soil by evaporation and reflection."
                ))

        elif issue.parameter == "Salinity (EC)":
            actions.append(CorrectiveAction(
                action="Increase irrigation to leach salts downward",
                target_parameter="Salinity (EC)",
                priority=1 if issue.severity == thresholds.Severity.CRITICAL else 2,
                severity=issue.severity,
                impact="High",
                dosage="30–50% more water than normal irrigation",
                timeline="Before planting or immediately if already growing",
                reasoning="Salt leaching requires drainage. Excess water is applied to move salts below root zone."
            ))
            actions.append(CorrectiveAction(
                action="Use low-salinity irrigation water source if available",
                target_parameter="Salinity (EC)",
                priority=2,
                severity=issue.severity,
                impact="High",
                reasoning="High-salt irrigation water perpetuates the problem. Switch to better source if possible."
            ))

        elif issue.parameter == "Organic Matter":
            actions.append(CorrectiveAction(
                action="Incorporate crop residues and apply compost or aged manure",
                target_parameter="Organic Matter",
                priority=3,
                severity=issue.severity,
                impact="Moderate",
                dosage="5–10 tons/ha of compost or well-rotted manure annually",
                timeline="Before planting, continue annually",
                reasoning="Organic matter improves structure, water retention, and nutrient cycling. Build organic matter over time."
            ))

        return actions

    # ──────────────────────────────────────────────────────────────────
    # Private: Dosage Calculations
    # ──────────────────────────────────────────────────────────────────

    def _calculate_lime_dosage(self, current_ph: float, target_ph: float) -> str:
        """Estimate agricultural lime requirement (tons/ha)."""
        ph_deficit = target_ph - current_ph
        # Rough formula: 1 pH unit ≈ 5–8 tons/ha of lime (depends on soil type)
        lime_tons = max(2, min(8, ph_deficit * 6))
        return f"{lime_tons:.1f}–{lime_tons + 1:.1f} tons/ha"

    def _calculate_sulfur_dosage(self, current_ph: float, target_ph: float) -> str:
        """Estimate elemental sulfur requirement (kg/ha)."""
        ph_excess = current_ph - target_ph
        # Rough formula: 1 pH unit ≈ 500–1000 kg/ha of sulfur
        sulfur_kg = max(500, min(2000, ph_excess * 750))
        return f"{sulfur_kg:.0f}–{sulfur_kg + 500:.0f} kg/ha"

    def _calculate_fertilizer_dosage(self, nutrient: str, current: float, target: float) -> str:
        """Estimate fertilizer requirement to reach target nutrient level."""
        deficit = target - current
        # Rough conversion: 1 ppm deficit ≈ 2 kg/ha of pure nutrient
        kg_pure = deficit * 2
        
        # Adjust for fertilizer grade (e.g., urea is 46% N)
        if nutrient == "N":
            fertilizer_kg = kg_pure / 0.46
            product = "urea"
        elif nutrient == "P":
            fertilizer_kg = kg_pure / 0.20
            product = "superphosphate"
        elif nutrient == "K":
            fertilizer_kg = kg_pure / 0.60
            product = "muriate of potash"
        else:
            return "Contact agronomist for custom calculations"
        
        return f"{fertilizer_kg:.0f} kg/ha of {product}"

    # ──────────────────────────────────────────────────────────────────
    # Private: Action Prioritization
    # ──────────────────────────────────────────────────────────────────

    def _rank_actions(
        self,
        actions: list[CorrectiveAction],
        diagnosis: diagnostics.SoilDiagnosis,
    ) -> list[CorrectiveAction]:
        """
        Rank actions by priority, considering:
        - Severity of the issue
        - Dependencies (e.g., pH before fertilizing)
        - Interactions identified in the diagnosis
        """
        # pH correction is nearly always first priority if it's critical
        ph_critical = any(
            issue.severity == thresholds.Severity.CRITICAL and issue.parameter == "pH"
            for issue in diagnosis.issues
        )
        
        def action_sort_key(action):
            score = 0
            
            # Severity contributes to priority
            severity_scores = {
                thresholds.Severity.CRITICAL: 0,
                thresholds.Severity.HIGH: 100,
                thresholds.Severity.MODERATE: 200,
                thresholds.Severity.LOW: 300,
            }
            score += severity_scores.get(action.severity, 999)
            
            # pH correction is almost always first
            if "lime" in action.action.lower() or "sulfur" in action.action.lower():
                score -= 50 if ph_critical else 10
            
            # Structural fixes (drainage, mulch) come before fertilizing
            if "drainage" in action.action.lower() or "mulch" in action.action.lower():
                score -= 20
            
            # Moisture management is critical
            if "irrigation" in action.action.lower() or "moisture" in action.action.lower():
                score -= 15
            
            # Fertilizer comes later
            if "fertilizer" in action.action.lower():
                score += 50
            
            return score
        
        actions.sort(key=action_sort_key)
        
        # Assign priority numbers
        for i, action in enumerate(actions, 1):
            action.priority = i
        
        return actions

    def _deduplicate_actions(self, actions: list[CorrectiveAction]) -> list[CorrectiveAction]:
        """Collapse actions that express the same management objective."""
        seen: set[str] = set()
        unique: list[CorrectiveAction] = []

        for action in actions:
            key = self._semantic_action_family(action)
            if key in seen:
                continue
            seen.add(key)
            unique.append(action)

        return unique

    @staticmethod
    def _semantic_action_family(action: CorrectiveAction) -> str:
        """Return a stable family key for semantically overlapping actions."""
        text = action.action.lower()

        families = {
            "excess_water_management": (
                "drainage", "standing water", "waterlogging", "aeration",
            ),
            "add_water": (
                "increase irrigation", "increase watering", "irrigate",
            ),
            "mulch": ("mulch",),
            "soil_acidity": ("lime", "liming"),
            "soil_alkalinity": ("elemental sulfur", "lower soil ph"),
            "nitrogen_fertilisation": ("nitrogen fertilizer", "urea", "ammonium nitrate"),
            "phosphorus_fertilisation": ("phosphorus fertilizer", "superphosphate"),
            "potassium_fertilisation": ("potassium fertilizer", "potash"),
            "salinity_leaching": ("leach salts", "salt leaching"),
            "organic_matter": ("crop residues", "compost", "aged manure"),
        }

        for family, phrases in families.items():
            if any(phrase in text for phrase in phrases):
                return family

        normalized = " ".join(text.split())
        return f"{action.target_parameter.lower()}::{normalized}"

    # ──────────────────────────────────────────────────────────────────
    # Private: Timeline and Monitoring
    # ──────────────────────────────────────────────────────────────────

    def _estimate_timeline(self, severity: thresholds.Severity) -> str:
        """Estimate total timeline to resolve issues."""
        if severity == thresholds.Severity.CRITICAL:
            return "2–4 weeks for immediate interventions; 1–2 months for full correction"
        elif severity == thresholds.Severity.HIGH:
            return "1–2 months with consistent management"
        elif severity == thresholds.Severity.MODERATE:
            return "2–4 months for full correction"
        else:
            return "Ongoing monitoring; improvements expected within growing season"

    def _monitoring_parameters_for_issues(self, issues: list[diagnostics.SoilIssue]) -> list[str]:
        """Identify only parameters connected to diagnosed issues."""
        params: list[str] = []

        for issue in issues:
            if issue.parameter not in params:
                params.append(issue.parameter)

        return params

    # ──────────────────────────────────────────────────────────────────
    # Private: Action Registry (extensible for future models)
    # ──────────────────────────────────────────────────────────────────

    def _build_action_registry(self) -> dict:
        """Build a registry of possible actions (extensible for ML)."""
        return {
            "pH": ["lime", "sulfur", "compost"],
            "Nitrogen": ["urea", "ammonium_nitrate", "compost"],
            "Phosphorus": ["superphosphate", "rock_phosphate"],
            "Potassium": ["potassium_chloride", "potassium_nitrate"],
            "Moisture": ["irrigation", "drainage", "mulch"],
            "Temperature": ["mulch", "irrigation"],
            "Salinity": ["leaching", "water_source"],
            "Organic Matter": ["compost", "manure", "crop_residue"],
        }


# ---------------------------------------------------------------------------
# Farm-level recommendation planning
# ---------------------------------------------------------------------------

class FarmRecommendationPlanner:
    """
    Convert parameter-level diagnostics into a concise farm-management brief.

    The brief is decision support for the generation prompt. It deliberately
    keeps acceptable readings out of the action list, merges related conditions
    into one management objective, and treats nutrient signals as provisional
    when crop stage, pH, or a crop-specific target is unavailable.
    """

    _FIELD_ALIASES = {
        "pH": ("soil_ph", "ph"),
        "Nitrogen": ("nitrogen_mg_kg", "nitrogen"),
        "Phosphorus": ("phosphorus_mg_kg", "phosphorus"),
        "Potassium": ("potassium_mg_kg", "potassium"),
        "Moisture": ("moisture_pct", "moisture"),
        "Temperature": ("temperature_c", "temperature"),
        "Salinity (EC)": ("salinity_ds_m", "salinity", "ec"),
        "Organic Matter": ("organic_matter_pct", "organic_matter"),
        "Humidity": ("humidity_pct", "humidity"),
    }

    _DIAGNOSTIC_ARGUMENTS = {
        "pH": "ph",
        "Nitrogen": "nitrogen",
        "Phosphorus": "phosphorus",
        "Potassium": "potassium",
        "Moisture": "moisture",
        "Temperature": "temperature",
        "Salinity (EC)": "salinity",
        "Organic Matter": "organic_matter",
    }

    _NUTRIENTS = {"Nitrogen", "Phosphorus", "Potassium"}
    _CROP_PROFILE_ALIASES = {
        "maize": "maize_corn",
        "corn": "maize_corn",
        "maize corn": "maize_corn",
        "maize/corn": "maize_corn",
    }
    _PROFILE_PARAMETER_KEYS = {
        "pH": "soil_ph",
        "Nitrogen": "nitrogen_ppm",
        "Phosphorus": "phosphorus_ppm",
        "Potassium": "potassium_ppm",
        "Moisture": "soil_moisture",
        "Temperature": "soil_temperature",
        "Salinity (EC)": "electrical_conductivity",
        "Organic Matter": "organic_matter_percent",
        "Humidity": "ambient_humidity",
    }
    _SEVERITY_RANK = {
        thresholds.Severity.CRITICAL: 0,
        thresholds.Severity.HIGH: 1,
        thresholds.Severity.MODERATE: 2,
        thresholds.Severity.LOW: 3,
    }

    def __init__(self) -> None:
        self._diagnostic_engine = diagnostics.SoilDiagnosticEngine()
        self._prescription_engine = PrescriptionEngine()
        self._crop_threshold_evaluator = ThresholdEvaluator()

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _extract_values(self, node: dict[str, Any]) -> dict[str, float]:
        values: dict[str, float] = {}

        for parameter, aliases in self._FIELD_ALIASES.items():
            for alias in aliases:
                if alias not in node:
                    continue
                value = self._number(node.get(alias))
                if value is not None:
                    values[parameter] = value
                    break

        return values

    @staticmethod
    def _format_value(parameter: str, value: float) -> str:
        units = {
            "pH": "",
            "Nitrogen": " mg/kg",
            "Phosphorus": " mg/kg",
            "Potassium": " mg/kg",
            "Moisture": "%",
            "Temperature": "°C",
            "Salinity (EC)": " dS/m",
            "Organic Matter": "%",
            "Humidity": "%",
        }
        rendered = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{parameter} {rendered}{units.get(parameter, '')}"

    @staticmethod
    def _is_known(value: Any) -> bool:
        text = str(value or "").strip().lower()
        return bool(text and text not in {"unknown", "none", "null", "n/a"})

    @staticmethod
    def _severity_label(severity: thresholds.Severity) -> str:
        return severity.name.lower()

    @staticmethod
    def _is_above(issue: diagnostics.SoilIssue) -> bool:
        return issue.measured_value > issue.optimal_range[1]

    @staticmethod
    def _is_below(issue: diagnostics.SoilIssue) -> bool:
        return issue.measured_value < issue.optimal_range[0]

    def _prescription_for(
        self,
        action_plan: ActionPlan,
        parameter: str,
    ) -> str | None:
        for action in action_plan.corrective_actions:
            if action.target_parameter == parameter:
                return action.action
        return None

    def _crop_profile_for(self, crop: Any) -> str | None:
        normalized = " ".join(str(crop or "").strip().casefold().split())
        return self._CROP_PROFILE_ALIASES.get(normalized)

    def _diagnose_node(
        self,
        node: dict[str, Any],
        values: dict[str, float],
    ) -> tuple[diagnostics.SoilDiagnosis, str]:
        """Prefer an existing crop profile; retain generic diagnostics as fallback."""
        crop_profile = self._crop_profile_for(node.get("currently_planted_crop"))
        timestamp = str(node.get("timestamp_utc") or "")
        context = {
            "crop": node.get("currently_planted_crop"),
            "season": node.get("season"),
            "soil_type": node.get("soil_type"),
            "growth_stage": node.get("growth_stage"),
        }

        if crop_profile is None:
            diagnostic_values = {
                argument: values[parameter]
                for parameter, argument in self._DIAGNOSTIC_ARGUMENTS.items()
                if parameter in values
            }
            return (
                self._diagnostic_engine.diagnose(
                    **diagnostic_values,
                    timestamp=timestamp,
                    context=context,
                ),
                "generic agronomic screening; validate against crop-specific knowledge",
            )

        crop_issues: list[diagnostics.SoilIssue] = []
        for parameter, value in values.items():
            profile_key = self._PROFILE_PARAMETER_KEYS.get(parameter)
            if profile_key is None:
                continue

            schema = self._crop_threshold_evaluator.get_parameter_schema(
                profile_key,
                crop_profile,
            )
            optimal_low = float(schema["optimal_min"])
            optimal_high = float(schema["optimal_max"])
            critical_low = float(schema["critical_min"])
            critical_high = float(schema["critical_max"])

            if optimal_low <= value <= optimal_high:
                continue

            below = value < optimal_low
            severity = (
                thresholds.Severity.CRITICAL
                if value <= critical_low or value >= critical_high
                else thresholds.Severity.MODERATE
            )
            direction = "below" if below else "above"
            crop_issues.append(diagnostics.SoilIssue(
                issue=f"{parameter} outside the {crop_profile} reference range",
                parameter=parameter,
                measured_value=value,
                optimal_range=(optimal_low, optimal_high),
                severity=severity,
                description=(
                    f"{self._format_value(parameter, value)} is {direction} the "
                    f"configured {crop_profile} screening range."
                ),
                root_cause=None,
            ))

        crop_issues.sort(key=lambda issue: self._SEVERITY_RANK[issue.severity])
        overall_severity = (
            crop_issues[0].severity
            if crop_issues
            else thresholds.Severity.LOW
        )
        return (
            diagnostics.SoilDiagnosis(
                timestamp=timestamp,
                issues=crop_issues,
                severity_summary=overall_severity,
                interactions=[],
                context=context,
            ),
            f"crop-specific profile: {crop_profile}",
        )

    def build_node_brief(self, node: dict[str, Any]) -> dict[str, Any]:
        """Build a serializable, prioritized management brief for one node."""
        values = self._extract_values(node)
        diagnosis, screening_basis = self._diagnose_node(node, values)
        action_plan = self._prescription_engine.generate_action_plan(diagnosis)
        issues = {issue.parameter: issue for issue in diagnosis.issues}

        classification: dict[str, list[str]] = {
            "critical_problematic": [],
            "borderline_watch": [],
            "acceptable_optimal": [],
            "unknown_not_enough_context": [],
        }

        for parameter, value in values.items():
            if parameter == "Humidity":
                bucket = "borderline_watch" if value >= 80.0 else "acceptable_optimal"
                classification[bucket].append(parameter)
                continue

            issue = issues.get(parameter)
            if issue is None:
                classification["acceptable_optimal"].append(parameter)
            elif parameter in self._NUTRIENTS:
                classification["unknown_not_enough_context"].append(parameter)
            elif issue.severity in {thresholds.Severity.CRITICAL, thresholds.Severity.HIGH}:
                classification["critical_problematic"].append(parameter)
            else:
                classification["borderline_watch"].append(parameter)

        priorities: list[dict[str, Any]] = []
        handled_parameters: set[str] = set()
        season = str(node.get("season") or "Unknown")
        rainy_season = "rain" in season.lower()
        humidity = values.get("Humidity")

        moisture_issue = issues.get("Moisture")
        temperature_issue = issues.get("Temperature")

        if moisture_issue is not None:
            wet = self._is_above(moisture_issue)
            evidence = [self._format_value("Moisture", moisture_issue.measured_value)]
            monitor = ["soil moisture"]
            supporting_steps: list[str]

            if wet:
                if humidity is not None and humidity >= 80.0:
                    evidence.append(self._format_value("Humidity", humidity))
                if rainy_season:
                    evidence.append(season)

                supporting_steps = [
                    "Suspend avoidable irrigation while the field remains excessively wet.",
                    "Clear or maintain drainage routes so excess water can leave the root zone.",
                ]
                if (humidity is not None and humidity >= 80.0) or rainy_season:
                    supporting_steps.append(
                        "Scout the crop for wet-weather disease symptoms while the field dries."
                    )
                    monitor.append("standing water and disease symptoms")

                label = "Excess wetness"
                objective = "Restore root-zone aeration through coordinated drainage and water management."
                domain = "excess_water"
            else:
                if temperature_issue is not None and self._is_above(temperature_issue):
                    evidence.append(self._format_value("Temperature", temperature_issue.measured_value))
                    handled_parameters.add("Temperature")
                    monitor.append("soil temperature")

                supporting_steps = [
                    "Restore moisture with a controlled irrigation adjustment rather than a large one-off application.",
                    "Reduce avoidable evaporation with suitable soil cover where agronomically appropriate.",
                ]
                label = "Root-zone water deficit"
                objective = "Restore stable crop-available moisture and limit further water stress."
                domain = "water_deficit"

            prescribed = self._prescription_for(action_plan, "Moisture")
            if prescribed and not wet:
                supporting_steps[0] = prescribed + "."

            priorities.append({
                "domain": domain,
                "label": label,
                "severity": self._severity_label(moisture_issue.severity),
                "severity_rank": self._SEVERITY_RANK[moisture_issue.severity],
                "parameters": ["Moisture"],
                "evidence": evidence,
                "recommended_objective": objective,
                "supporting_steps": supporting_steps[:3],
                "monitor": monitor,
                "context_limits": [],
                "prescription_basis": prescribed,
            })
            handled_parameters.add("Moisture")

            if (
                wet
                and temperature_issue is not None
                and self._is_above(temperature_issue)
                and temperature_issue.severity != thresholds.Severity.CRITICAL
            ):
                priorities[-1]["evidence"].append(
                    self._format_value("Temperature", temperature_issue.measured_value)
                )
                priorities[-1]["parameters"].append("Temperature")
                handled_parameters.add("Temperature")

        if temperature_issue is not None and "Temperature" not in handled_parameters:
            hot = self._is_above(temperature_issue)
            priorities.append({
                "domain": "temperature_stress",
                "label": "Heat stress" if hot else "Cold stress",
                "severity": self._severity_label(temperature_issue.severity),
                "severity_rank": self._SEVERITY_RANK[temperature_issue.severity],
                "parameters": ["Temperature"],
                "evidence": [self._format_value("Temperature", temperature_issue.measured_value)],
                "recommended_objective": (
                    "Reduce root-zone heat load without worsening current moisture conditions."
                    if hot
                    else "Protect the root zone and avoid management that further delays warming."
                ),
                "supporting_steps": [
                    "Use crop-appropriate soil cover and adjust field operations to the temperature trend."
                ],
                "monitor": ["soil temperature"],
                "context_limits": ["Confirm crop growth stage before changing water management."],
            })
            handled_parameters.add("Temperature")

        ph_issue = issues.get("pH")
        if ph_issue is not None:
            acidic = self._is_below(ph_issue)
            context_limits = []
            if not self._is_known(node.get("soil_type")):
                context_limits.append("Soil type or buffering capacity is unavailable.")

            priorities.append({
                "domain": "soil_reaction",
                "label": "Acidic soil reaction" if acidic else "Alkaline soil reaction",
                "severity": self._severity_label(ph_issue.severity),
                "severity_rank": self._SEVERITY_RANK[ph_issue.severity],
                "parameters": ["pH"],
                "evidence": [self._format_value("pH", ph_issue.measured_value)],
                "recommended_objective": "Confirm the soil reaction and correct it before relying on nutrient additions.",
                "supporting_steps": [
                    "Confirm pH with a calibrated soil test.",
                    "Determine any amendment type and rate from soil texture/buffering information and crop requirement.",
                ],
                "monitor": ["soil pH"],
                "context_limits": context_limits,
            })

        for parameter, domain, label in (
            ("Salinity (EC)", "salinity", "Salinity stress"),
            ("Organic Matter", "soil_structure", "Low organic matter"),
        ):
            issue = issues.get(parameter)
            if issue is None:
                continue
            prescribed = self._prescription_for(action_plan, parameter)
            priorities.append({
                "domain": domain,
                "label": label,
                "severity": self._severity_label(issue.severity),
                "severity_rank": self._SEVERITY_RANK[issue.severity],
                "parameters": [parameter],
                "evidence": [self._format_value(parameter, issue.measured_value)],
                "recommended_objective": issue.description,
                "supporting_steps": [prescribed + "."] if prescribed else [],
                "monitor": [parameter.lower()],
                "context_limits": [],
            })

        nutrient_issues = [
            issues[parameter]
            for parameter in ("Nitrogen", "Phosphorus", "Potassium")
            if parameter in issues
        ]
        if nutrient_issues:
            worst_nutrient_severity = min(
                (issue.severity for issue in nutrient_issues),
                key=lambda severity: self._SEVERITY_RANK[severity],
            )
            evidence = [
                self._format_value(issue.parameter, issue.measured_value)
                for issue in nutrient_issues
            ]
            missing_context = [
                label
                for key, label in (
                    ("growth_stage", "crop growth stage"),
                    ("soil_type", "soil type"),
                )
                if not self._is_known(node.get(key))
            ]
            if "pH" not in values:
                missing_context.insert(0, "soil pH")
            if screening_basis.startswith("generic"):
                missing_context.insert(0, "crop-specific nutrient target")
            priorities.append({
                "domain": "nutrient_verification",
                "label": "Nutrient result requiring crop-specific verification",
                "severity": (
                    "high_investigation"
                    if worst_nutrient_severity == thresholds.Severity.CRITICAL
                    else "watch"
                ),
                "severity_rank": 2,
                "parameters": [issue.parameter for issue in nutrient_issues],
                "evidence": evidence,
                "recommended_objective": (
                    "Verify the nutrient concern against crop- and growth-stage-specific targets before treatment."
                ),
                "supporting_steps": [
                    "Obtain or confirm soil pH, crop growth stage, and a representative soil test.",
                    "Use the confirmed crop requirement to select an amendment; do not infer a rate from relative N-P-K sizes.",
                ],
                "monitor": [issue.parameter.lower() for issue in nutrient_issues],
                "context_limits": (
                    ["Missing: " + ", ".join(missing_context) + "."]
                    if missing_context
                    else []
                ),
            })

        priorities.sort(key=lambda item: (item["severity_rank"], item["domain"]))

        planted_crop = node.get("currently_planted_crop")
        ideal_crop = node.get("ai_predicted_ideal_crop")
        strategic_considerations: list[str] = []
        if (
            self._is_known(planted_crop)
            and self._is_known(ideal_crop)
            and str(planted_crop).strip().casefold() != str(ideal_crop).strip().casefold()
        ):
            strategic_considerations.append(
                f"The currently planted crop is {planted_crop}, while the ML suitability prediction is {ideal_crop}. "
                "Treat this as a strategic crop-soil comparison, not as proof that the current crop is failing."
            )

        if rainy_season:
            preventive_focus = [
                "Keep drainage routes serviceable ahead of further rain.",
                "Scout the crop for wet-weather disease and weed pressure.",
            ]
        elif "dry" in season.lower():
            preventive_focus = [
                "Track the soil-moisture trend before changing irrigation frequency.",
                "Maintain suitable soil cover to reduce avoidable water loss.",
            ]
        else:
            preventive_focus = [
                "Continue routine field scouting and record changes in the important trends.",
                "Keep drainage and sensor placement in working order.",
            ]

        return {
            "node_id": node.get("node_id") or "Unknown node",
            "currently_planted_crop": planted_crop or "Unknown",
            "season": season,
            "screening_basis": screening_basis,
            "internal_parameter_classification": classification,
            "healthy_sensor_state": len(priorities) == 0,
            "priorities": priorities[:3],
            "strategic_considerations": strategic_considerations,
            "preventive_focus_if_no_actionable_problem": preventive_focus,
            "output_constraints": [
                "Do not narrate acceptable parameters.",
                "Present one consolidated recommendation per independent priority.",
                "Do not state an exact nutrient or amendment rate without the required crop and soil context.",
            ],
        }
