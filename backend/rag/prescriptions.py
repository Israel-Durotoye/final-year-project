"""
prescriptions.py — Soil Management Action Recommendation Engine

Generates prioritized corrective actions and interventions based on
soil diagnoses. Actions are ranked by impact and dependencies.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

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
        """Remove duplicate actions (same target parameter)."""
        seen = {}
        unique = []
        for action in actions:
            key = action.target_parameter
            if key not in seen:
                seen[key] = action
                unique.append(action)
        return unique

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
        """Identify which parameters to monitor after interventions."""
        params = set()
        for issue in issues:
            params.add(issue.parameter)
        
        monitoring = list(params)
        # Always monitor soil moisture and temperature
        if "Moisture" not in monitoring:
            monitoring.append("Moisture")
        if "Temperature" not in monitoring:
            monitoring.append("Temperature")
        
        return monitoring

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
