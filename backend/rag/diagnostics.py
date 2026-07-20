"""
diagnostics.py — Soil Condition Diagnostic Engine

Analyzes soil measurements against agronomic thresholds and generates
structured diagnoses with severity classifications and identified causes.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.rag import thresholds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic Output Models
# ---------------------------------------------------------------------------

@dataclass
class SoilIssue:
    """A single diagnosed soil issue."""
    issue: str                           # E.g., "Acidic Soil", "Nitrogen Deficiency"
    parameter: str                       # E.g., "pH", "Nitrogen"
    measured_value: float               # Actual measured value
    optimal_range: tuple[float, float]  # (low, high) target
    severity: thresholds.Severity       # CRITICAL, HIGH, MODERATE, LOW
    description: str                    # Human-readable explanation
    root_cause: Optional[str] = None    # Why this is happening


@dataclass
class SoilDiagnosis:
    """Complete soil diagnostic result."""
    timestamp: str                      # ISO format timestamp
    issues: list[SoilIssue]            # Diagnosed issues, sorted by severity
    severity_summary: thresholds.Severity  # Overall severity (worst of all issues)
    interactions: list[str] = field(default_factory=list)  # Parameter interactions
    context: dict = field(default_factory=dict)  # Sensor data, crop info, etc.


# ---------------------------------------------------------------------------
# Diagnostic Engine
# ---------------------------------------------------------------------------

class SoilDiagnosticEngine:
    """
    Analyzes soil sensor data and generates diagnoses.
    
    Usage:
        engine = SoilDiagnosticEngine()
        diagnosis = engine.diagnose(
            ph=4.8,
            nitrogen=12,
            phosphorus=8,
            potassium=90,
            moisture=18,
            temperature=25,
        )
    """

    def __init__(self):
        """Initialize diagnostic engine with default thresholds."""
        self.ph_threshold = thresholds.PH_GENERAL
        self.nutrient_n = thresholds.NITROGEN
        self.nutrient_p = thresholds.PHOSPHORUS
        self.nutrient_k = thresholds.POTASSIUM
        self.moisture_threshold = thresholds.MOISTURE_SANDY
        self.temperature_threshold = thresholds.TEMPERATURE_WARM_SEASON
        self.salinity_threshold = thresholds.SALINITY_GENERAL
        self.organic_matter_threshold = thresholds.ORGANIC_MATTER

    def diagnose(
        self,
        ph: Optional[float] = None,
        nitrogen: Optional[float] = None,
        phosphorus: Optional[float] = None,
        potassium: Optional[float] = None,
        moisture: Optional[float] = None,
        temperature: Optional[float] = None,
        salinity: Optional[float] = None,
        organic_matter: Optional[float] = None,
        timestamp: str = None,
        context: dict = None,
    ) -> SoilDiagnosis:
        """
        Generate a complete soil diagnosis from measured parameters.
        
        Args:
            ph: Soil pH (0–14).
            nitrogen: N concentration (ppm).
            phosphorus: P concentration (ppm).
            potassium: K concentration (ppm).
            moisture: Volumetric water content (%).
            temperature: Soil temperature (°C).
            salinity: Electrical conductivity (dS/m).
            organic_matter: Organic matter content (%).
            timestamp: ISO timestamp of measurement.
            context: Additional context (crop, soil type, etc.).
        
        Returns:
            SoilDiagnosis with ranked issues and severity.
        """
        issues = []
        interactions = []

        # Analyze pH
        if ph is not None:
            ph_issue = self._diagnose_ph(ph)
            if ph_issue:
                issues.append(ph_issue)

        # Analyze nutrients
        if nitrogen is not None:
            n_issue = self._diagnose_nutrient("Nitrogen", nitrogen, self.nutrient_n)
            if n_issue:
                issues.append(n_issue)

        if phosphorus is not None:
            p_issue = self._diagnose_nutrient("Phosphorus", phosphorus, self.nutrient_p)
            if p_issue:
                issues.append(p_issue)

        if potassium is not None:
            k_issue = self._diagnose_nutrient("Potassium", potassium, self.nutrient_k)
            if k_issue:
                issues.append(k_issue)

        # Analyze moisture
        if moisture is not None:
            moisture_issue = self._diagnose_moisture(moisture)
            if moisture_issue:
                issues.append(moisture_issue)

        # Analyze temperature
        if temperature is not None:
            temp_issue = self._diagnose_temperature(temperature)
            if temp_issue:
                issues.append(temp_issue)

        # Analyze salinity
        if salinity is not None:
            salinity_issue = self._diagnose_salinity(salinity)
            if salinity_issue:
                issues.append(salinity_issue)

        # Analyze organic matter
        if organic_matter is not None:
            om_issue = self._diagnose_organic_matter(organic_matter)
            if om_issue:
                issues.append(om_issue)

        # Identify interactions
        interactions = self._identify_interactions(
            ph=ph, nitrogen=nitrogen, moisture=moisture, temperature=temperature
        )

        # Rank issues by severity
        severity_order = {
            thresholds.Severity.CRITICAL: 0,
            thresholds.Severity.HIGH: 1,
            thresholds.Severity.MODERATE: 2,
            thresholds.Severity.LOW: 3,
        }
        issues.sort(key=lambda x: severity_order.get(x.severity, 999))

        # Determine overall severity
        overall_severity = thresholds.Severity.LOW
        if issues:
            overall_severity = issues[0].severity

        return SoilDiagnosis(
            timestamp=timestamp or "",
            issues=issues,
            severity_summary=overall_severity,
            interactions=interactions,
            context=context or {},
        )

    # ──────────────────────────────────────────────────────────────────
    # Private: Individual parameter diagnosis
    # ──────────────────────────────────────────────────────────────────

    def _diagnose_ph(self, ph: float) -> Optional[SoilIssue]:
        """Diagnose pH-related issues."""
        severity = thresholds.classify_ph_severity(ph, self.ph_threshold)
        if severity == thresholds.Severity.LOW:
            return None  # Optimal

        if ph < self.ph_threshold.optimal_low:
            issue_name = "Acidic Soil" if ph > self.ph_threshold.critical_low else "Critically Acidic Soil"
            description = (
                f"Soil pH is {ph:.1f}, which is below the optimal range. "
                f"Acidic conditions reduce nutrient availability and may inhibit root growth."
            )
            root_cause = "Natural soil type, organic acid accumulation, or excessive acid rain"
        else:
            issue_name = "Alkaline Soil" if ph < self.ph_threshold.critical_high else "Critically Alkaline Soil"
            description = (
                f"Soil pH is {ph:.1f}, which is above the optimal range. "
                f"Alkaline conditions can lock up micronutrients and reduce their availability."
            )
            root_cause = "High limestone content, calcite deposits, or excessive irrigation water"

        return SoilIssue(
            issue=issue_name,
            parameter="pH",
            measured_value=ph,
            optimal_range=(self.ph_threshold.optimal_low, self.ph_threshold.optimal_high),
            severity=severity,
            description=description,
            root_cause=root_cause,
        )

    def _diagnose_nutrient(
        self, nutrient_name: str, value: float, threshold
    ) -> Optional[SoilIssue]:
        """Diagnose nutrient deficiency or excess."""
        severity = thresholds.classify_nutrient_severity(value, threshold)
        if severity == thresholds.Severity.LOW:
            return None  # Optimal

        optimal_range = (threshold.optimal_low, threshold.optimal_high)

        if value < threshold.optimal_low:
            issue_name = f"{nutrient_name} Deficiency"
            description = (
                f"{nutrient_name} concentration is {value:.1f} {threshold.unit}, "
                f"below the optimal range ({threshold.optimal_low}–{threshold.optimal_high} {threshold.unit}). "
                f"Plants will show growth stunting and reduced yield."
            )
            root_cause = f"Soil naturally low in {nutrient_name}, or excessive leaching due to high rainfall or irrigation"
        else:
            issue_name = f"{nutrient_name} Excess"
            description = (
                f"{nutrient_name} concentration is {value:.1f} {threshold.unit}, "
                f"above the optimal range. High levels can cause nutrient imbalances and reduce crop quality."
            )
            root_cause = f"Over-fertilization or repeated heavy applications of {nutrient_name}-rich manure"

        return SoilIssue(
            issue=issue_name,
            parameter=nutrient_name,
            measured_value=value,
            optimal_range=optimal_range,
            severity=severity,
            description=description,
            root_cause=root_cause,
        )

    def _diagnose_moisture(self, moisture: float) -> Optional[SoilIssue]:
        """Diagnose moisture stress."""
        severity = thresholds.classify_moisture_severity(moisture, self.moisture_threshold)
        if severity == thresholds.Severity.LOW:
            return None  # Optimal

        if moisture < self.moisture_threshold.optimal_low:
            issue_name = "Drought Stress" if moisture > self.moisture_threshold.dry else "Severe Drought"
            description = (
                f"Soil moisture is {moisture:.1f}%, below the optimal range. "
                f"Plants will experience water stress, reduced growth, and potential yield loss."
            )
            root_cause = "Insufficient rainfall, poor irrigation, or high evaporation rates"
        else:
            issue_name = "Waterlogging" if moisture < self.moisture_threshold.critical_wet else "Severe Waterlogging"
            description = (
                f"Soil moisture is {moisture:.1f}%, above the optimal range. "
                f"Excess water reduces oxygen availability, inhibiting root respiration and promoting disease."
            )
            root_cause = "Poor drainage, excessive rainfall, or over-irrigation"

        return SoilIssue(
            issue=issue_name,
            parameter="Moisture",
            measured_value=moisture,
            optimal_range=(self.moisture_threshold.optimal_low, self.moisture_threshold.optimal_high),
            severity=severity,
            description=description,
            root_cause=root_cause,
        )

    def _diagnose_temperature(self, temperature: float) -> Optional[SoilIssue]:
        """Diagnose temperature stress."""
        severity = thresholds.classify_nutrient_severity(
            temperature, 
            thresholds.NutrientThreshold(
                name="temp", unit="°C",
                critical_low=self.temperature_threshold.critical_low,
                low=self.temperature_threshold.low,
                moderate_low=self.temperature_threshold.moderate_low,
                optimal_low=self.temperature_threshold.optimal_low,
                optimal_high=self.temperature_threshold.optimal_high,
                high=self.temperature_threshold.high,
                critical_high=self.temperature_threshold.critical_high,
            )
        )
        if severity == thresholds.Severity.LOW:
            return None  # Optimal

        if temperature < self.temperature_threshold.optimal_low:
            issue_name = "Cold Stress"
            description = (
                f"Soil temperature is {temperature:.1f}°C, below optimal. "
                f"Microbial activity and nutrient mineralization are reduced."
            )
        else:
            issue_name = "Heat Stress"
            description = (
                f"Soil temperature is {temperature:.1f}°C, above optimal. "
                f"High temperatures can damage root systems and promote pathogenic organisms."
            )

        return SoilIssue(
            issue=issue_name,
            parameter="Temperature",
            measured_value=temperature,
            optimal_range=(self.temperature_threshold.optimal_low, self.temperature_threshold.optimal_high),
            severity=severity,
            description=description,
            root_cause="Seasonal variation or lack of organic matter/mulch cover",
        )

    def _diagnose_salinity(self, ec: float) -> Optional[SoilIssue]:
        """Diagnose salt stress."""
        if ec <= self.salinity_threshold.optimal_high:
            return None  # Optimal

        if ec <= self.salinity_threshold.moderate_high:
            severity = thresholds.Severity.MODERATE
            issue_name = "Mild Salinity"
            description = f"Electrical conductivity is {ec:.2f} dS/m. Some salt-sensitive crops may show stress."
        elif ec <= self.salinity_threshold.high:
            severity = thresholds.Severity.HIGH
            issue_name = "Moderate Salinity"
            description = f"Electrical conductivity is {ec:.2f} dS/m. Most crops will show reduced growth."
        else:
            severity = thresholds.Severity.CRITICAL
            issue_name = "High Salinity"
            description = f"Electrical conductivity is {ec:.2f} dS/m. Severe salt stress affecting most crops."

        return SoilIssue(
            issue=issue_name,
            parameter="Salinity (EC)",
            measured_value=ec,
            optimal_range=(0, self.salinity_threshold.optimal_high),
            severity=severity,
            description=description,
            root_cause="Poor irrigation water quality, salt accumulation, or saline groundwater intrusion",
        )

    def _diagnose_organic_matter(self, om: float) -> Optional[SoilIssue]:
        """Diagnose organic matter levels."""
        if om >= self.organic_matter_threshold.optimal_low:
            return None  # Optimal or adequate

        if om >= self.organic_matter_threshold.moderate_low:
            severity = thresholds.Severity.MODERATE
            issue_name = "Low Organic Matter"
        elif om >= self.organic_matter_threshold.low:
            severity = thresholds.Severity.HIGH
            issue_name = "Very Low Organic Matter"
        else:
            severity = thresholds.Severity.CRITICAL
            issue_name = "Critically Low Organic Matter"

        description = (
            f"Organic matter is {om:.1f}%. Low organic matter reduces soil structure, "
            f"water retention, and nutrient availability."
        )

        return SoilIssue(
            issue=issue_name,
            parameter="Organic Matter",
            measured_value=om,
            optimal_range=(self.organic_matter_threshold.optimal_low, self.organic_matter_threshold.optimal_high),
            severity=severity,
            description=description,
            root_cause="Insufficient crop residue incorporation, limited manure application, or erosion",
        )

    # ──────────────────────────────────────────────────────────────────
    # Private: Identify interactions
    # ──────────────────────────────────────────────────────────────────

    def _identify_interactions(
        self,
        ph: Optional[float] = None,
        nitrogen: Optional[float] = None,
        moisture: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """
        Identify critical interactions between parameters.
        
        Example:
            - Low pH reduces nutrient availability even if levels are adequate
            - High temperature increases evaporation, worsening drought stress
        """
        interactions = []

        # pH × Nutrient interaction
        if ph is not None and ph < self.ph_threshold.moderate_low and nitrogen is not None:
            if nitrogen > self.nutrient_n.optimal_low:
                interactions.append(
                    "CRITICAL: Acidic pH (4.8) locks up available nitrogen. "
                    "Even adequate nitrogen levels will not be plant-available until pH is corrected."
                )

        # Moisture × Temperature interaction
        if moisture is not None and temperature is not None:
            if moisture < self.moisture_threshold.optimal_low and temperature > self.temperature_threshold.optimal_high:
                interactions.append(
                    "CRITICAL: Combined drought and heat stress. High temperature increases evaporation "
                    "and exacerbates water stress. Priority: Restore moisture before correcting other issues."
                )

        # Salinity × Moisture interaction
        if moisture is not None and moisture < self.moisture_threshold.optimal_low:
            interactions.append(
                "Drought stress reduces salt leaching. Once moisture is restored, "
                "plan salt remediation through managed leaching."
            )

        return interactions
