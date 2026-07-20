"""
thresholds.py — Agronomic Reference Thresholds

Defines acceptable ranges, critical thresholds, and severity classifications
for soil parameters commonly monitored in precision agriculture.

All ranges are based on standard agronomic guidelines for tropical and subtropical crops.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

# ---------------------------------------------------------------------------
# Severity Classification
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Severity classification for soil conditions."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


# ---------------------------------------------------------------------------
# Nutrient Thresholds (ppm or mg/kg)
# ---------------------------------------------------------------------------

@dataclass
class NutrientThreshold:
    """Threshold ranges for a single nutrient."""
    name: str
    unit: str
    critical_low: float      # Below this = Critical
    low: float               # Below this = High
    moderate_low: float      # Below this = Moderate
    optimal_low: float       # Start of optimal range
    optimal_high: float      # End of optimal range
    high: float              # Above optimal = Moderate
    critical_high: float     # Above this = Critical


# Nitrogen (N) in ppm or mg/kg
NITROGEN = NutrientThreshold(
    name="Nitrogen",
    unit="ppm",
    critical_low=5.0,
    low=10.0,
    moderate_low=15.0,
    optimal_low=20.0,
    optimal_high=40.0,
    high=60.0,
    critical_high=80.0,
)

# Phosphorus (P) in ppm
PHOSPHORUS = NutrientThreshold(
    name="Phosphorus",
    unit="ppm",
    critical_low=2.0,
    low=5.0,
    moderate_low=8.0,
    optimal_low=10.0,
    optimal_high=30.0,
    high=50.0,
    critical_high=70.0,
)

# Potassium (K) in ppm
POTASSIUM = NutrientThreshold(
    name="Potassium",
    unit="ppm",
    critical_low=30.0,
    low=50.0,
    moderate_low=80.0,
    optimal_low=100.0,
    optimal_high=200.0,
    high=300.0,
    critical_high=400.0,
)


# ---------------------------------------------------------------------------
# pH Thresholds
# ---------------------------------------------------------------------------

@dataclass
class PHThreshold:
    """pH range classification for soil types."""
    soil_type: str
    critical_low: float      # Extremely acidic
    low: float               # Acidic
    moderate_low: float      # Moderately acidic
    optimal_low: float       # Optimal range start
    optimal_high: float      # Optimal range end
    moderate_high: float     # Moderately alkaline
    high: float              # Alkaline
    critical_high: float     # Extremely alkaline


# Standard pH thresholds (for general tropical/subtropical crops)
PH_GENERAL = PHThreshold(
    soil_type="General Crops",
    critical_low=4.0,
    low=4.5,
    moderate_low=5.0,
    optimal_low=6.0,
    optimal_high=7.0,
    moderate_high=7.5,
    high=8.0,
    critical_high=8.5,
)

# Acidic-loving crops (e.g., blueberries, azaleas)
PH_ACIDIC_CROPS = PHThreshold(
    soil_type="Acidic-Loving Crops",
    critical_low=4.0,
    low=4.2,
    moderate_low=4.5,
    optimal_low=4.8,
    optimal_high=5.5,
    moderate_high=6.0,
    high=6.5,
    critical_high=7.5,
)


# ---------------------------------------------------------------------------
# Moisture Thresholds (% volumetric water content)
# ---------------------------------------------------------------------------

@dataclass
class MoistureThreshold:
    """Soil moisture content ranges (volumetric %)."""
    soil_type: str
    critical_dry: float      # Severe drought stress
    dry: float               # Drought stress
    moderate_dry: float      # Suboptimal
    optimal_low: float       # Optimal range start
    optimal_high: float      # Optimal range end
    wet: float               # Waterlogging risk
    critical_wet: float      # Severe waterlogging


# Sandy loam moisture thresholds
MOISTURE_SANDY = MoistureThreshold(
    soil_type="Sandy Loam",
    critical_dry=5.0,
    dry=8.0,
    moderate_dry=12.0,
    optimal_low=15.0,
    optimal_high=22.0,
    wet=28.0,
    critical_wet=35.0,
)

# Clay loam moisture thresholds
MOISTURE_CLAY = MoistureThreshold(
    soil_type="Clay Loam",
    critical_dry=8.0,
    dry=12.0,
    moderate_dry=16.0,
    optimal_low=20.0,
    optimal_high=30.0,
    wet=38.0,
    critical_wet=45.0,
)


# ---------------------------------------------------------------------------
# Temperature Thresholds (°C)
# ---------------------------------------------------------------------------

@dataclass
class TemperatureThreshold:
    """Soil temperature ranges for crop growth."""
    crop: str
    critical_low: float      # Growth severely inhibited
    low: float               # Growth inhibited
    moderate_low: float      # Suboptimal growth
    optimal_low: float       # Optimal range start
    optimal_high: float      # Optimal range end
    moderate_high: float     # Suboptimal growth
    high: float              # Growth inhibited
    critical_high: float     # Growth severely inhibited


# Temperature thresholds for general warm-season crops
TEMPERATURE_WARM_SEASON = TemperatureThreshold(
    crop="Warm-Season Crops (maize, rice)",
    critical_low=10.0,
    low=12.0,
    moderate_low=15.0,
    optimal_low=18.0,
    optimal_high=30.0,
    moderate_high=32.0,
    high=35.0,
    critical_high=40.0,
)

# Temperature thresholds for cool-season crops
TEMPERATURE_COOL_SEASON = TemperatureThreshold(
    crop="Cool-Season Crops (wheat, legumes)",
    critical_low=0.0,
    low=3.0,
    moderate_low=8.0,
    optimal_low=12.0,
    optimal_high=25.0,
    moderate_high=28.0,
    high=32.0,
    critical_high=35.0,
)


# ---------------------------------------------------------------------------
# Salinity Thresholds (dS/m — deciSiemens per meter)
# ---------------------------------------------------------------------------

@dataclass
class SalinityThreshold:
    """Electrical conductivity (EC) ranges indicating salt stress."""
    name: str
    optimal_high: float      # Safe upper limit
    moderate_high: float     # Mild salt stress
    high: float              # Significant salt stress
    critical_high: float     # Severe salt stress


# General salinity thresholds
SALINITY_GENERAL = SalinityThreshold(
    name="General Crops",
    optimal_high=1.0,
    moderate_high=1.5,
    high=2.5,
    critical_high=4.0,
)

# Salt-sensitive crops
SALINITY_SENSITIVE = SalinityThreshold(
    name="Salt-Sensitive Crops",
    optimal_high=0.7,
    moderate_high=1.0,
    high=1.5,
    critical_high=2.5,
)

# Salt-tolerant crops
SALINITY_TOLERANT = SalinityThreshold(
    name="Salt-Tolerant Crops",
    optimal_high=2.0,
    moderate_high=3.0,
    high=4.5,
    critical_high=6.0,
)


# ---------------------------------------------------------------------------
# Organic Matter Thresholds (%)
# ---------------------------------------------------------------------------

@dataclass
class OrganicMatterThreshold:
    """Soil organic matter percentage ranges."""
    soil_type: str
    critical_low: float      # Very poor soil structure
    low: float               # Poor soil structure
    moderate_low: float      # Suboptimal
    optimal_low: float       # Adequate start
    optimal_high: float      # Ideal
    high: float              # Excess (rare)


ORGANIC_MATTER = OrganicMatterThreshold(
    soil_type="General",
    critical_low=0.5,
    low=1.0,
    moderate_low=1.5,
    optimal_low=2.0,
    optimal_high=5.0,
    high=8.0,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def classify_nutrient_severity(value: float, threshold: NutrientThreshold) -> Severity:
    """
    Classify nutrient concentration as Low, Moderate, High, or Critical.
    
    Args:
        value: Measured nutrient concentration.
        threshold: NutrientThreshold object.
    
    Returns:
        Severity classification.
    """
    if value <= threshold.critical_low:
        return Severity.CRITICAL
    elif value <= threshold.low:
        return Severity.HIGH
    elif value <= threshold.moderate_low:
        return Severity.MODERATE
    elif value <= threshold.optimal_low:
        return Severity.MODERATE
    elif value <= threshold.optimal_high:
        return Severity.LOW  # Optimal
    elif value <= threshold.high:
        return Severity.MODERATE
    elif value <= threshold.critical_high:
        return Severity.HIGH
    else:
        return Severity.CRITICAL


def classify_ph_severity(value: float, threshold: PHThreshold) -> Severity:
    """Classify soil pH as Low, Moderate, High, or Critical."""
    if value <= threshold.critical_low:
        return Severity.CRITICAL
    elif value <= threshold.low:
        return Severity.HIGH
    elif value <= threshold.moderate_low:
        return Severity.MODERATE
    elif value <= threshold.optimal_low:
        return Severity.MODERATE
    elif value <= threshold.optimal_high:
        return Severity.LOW  # Optimal
    elif value <= threshold.moderate_high:
        return Severity.MODERATE
    elif value <= threshold.high:
        return Severity.HIGH
    else:
        return Severity.CRITICAL


def classify_moisture_severity(value: float, threshold: MoistureThreshold) -> Severity:
    """Classify soil moisture as Low, Moderate, High, or Critical."""
    if value <= threshold.critical_dry:
        return Severity.CRITICAL
    elif value <= threshold.dry:
        return Severity.HIGH
    elif value <= threshold.moderate_dry:
        return Severity.MODERATE
    elif value <= threshold.optimal_low:
        return Severity.MODERATE
    elif value <= threshold.optimal_high:
        return Severity.LOW  # Optimal
    elif value <= threshold.wet:
        return Severity.MODERATE
    elif value <= threshold.critical_wet:
        return Severity.HIGH
    else:
        return Severity.CRITICAL


def get_optimal_range(threshold: NutrientThreshold) -> tuple[float, float]:
    """Return the optimal range for a nutrient."""
    return (threshold.optimal_low, threshold.optimal_high)
