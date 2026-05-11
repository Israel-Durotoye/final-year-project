"""
mock_lora.py — Soil Doctor: Mock LoRa Telemetry Generator

Responsibility:
    Simulates the telemetry stream from real in-field LoRa sensor nodes.
    Produces realistic, semi-randomized payloads that intentionally
    drift in and out of optimal ranges to produce both OK and flagged
    evaluations — giving the prescriptive engine something meaningful to work on.

Three generation modes:
    1. random_nominal()  — All parameters near-optimal (baseline healthy field)
    2. random_stressed() — Random parameters pushed into WARNING or CRITICAL
    3. scenario()        — Named agronomic scenarios (acid_soil, drought, etc.)
                           for deterministic regression testing

All payloads are plain Python dicts that match the parameter keys in
optimal_thresholds.json. The evaluator consumes them directly.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Scenario catalogue — deterministic, named test cases
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, dict[str, Any]] = {
    "acid_soil": {
        "_description": "Severely acidic soil — aluminium toxicity risk. N, P, K all deficient.",
        "soil_ph": 4.6,          # CRITICAL low
        "soil_moisture": 52.0,
        "soil_temperature": 26.0,
        "nitrogen_ppm": 17.0,    # WARNING low
        "phosphorus_ppm": 11.0,  # WARNING low
        "potassium_ppm": 95.0,   # WARNING low
        "electrical_conductivity": 0.7,
        "organic_matter_percent": 1.8,  # WARNING low
        "ambient_humidity": 68.0,
        "ambient_temperature": 28.0,
    },
    "drought_stress": {
        "_description": "Severe water deficit with heat stress — irrigation emergency.",
        "soil_ph": 6.4,
        "soil_moisture": 16.0,   # CRITICAL low
        "soil_temperature": 38.0, # WARNING high
        "nitrogen_ppm": 24.0,
        "phosphorus_ppm": 19.0,
        "potassium_ppm": 165.0,
        "electrical_conductivity": 1.2,
        "organic_matter_percent": 2.8,  # WARNING low
        "ambient_humidity": 22.0,  # WARNING low
        "ambient_temperature": 37.0,  # WARNING high
    },
    "waterlogged": {
        "_description": "Waterlogged field — anaerobic risk, nutrient leaching.",
        "soil_ph": 6.2,
        "soil_moisture": 88.0,   # CRITICAL high
        "soil_temperature": 22.0,
        "nitrogen_ppm": 8.5,     # CRITICAL low (leached out)
        "phosphorus_ppm": 14.0,  # WARNING low
        "potassium_ppm": 100.0,  # WARNING low (leached)
        "electrical_conductivity": 0.4,  # WARNING low
        "organic_matter_percent": 7.5,   # WARNING high
        "ambient_humidity": 92.0,  # WARNING high
        "ambient_temperature": 24.0,
    },
    "saline_soil": {
        "_description": "High-salinity soil — osmotic stress preventing water uptake.",
        "soil_ph": 7.8,           # WARNING high
        "soil_moisture": 55.0,
        "soil_temperature": 28.0,
        "nitrogen_ppm": 50.0,     # WARNING high
        "phosphorus_ppm": 40.0,   # WARNING high
        "potassium_ppm": 420.0,   # CRITICAL high
        "electrical_conductivity": 3.5,  # CRITICAL high
        "organic_matter_percent": 2.0,   # WARNING low
        "ambient_humidity": 55.0,
        "ambient_temperature": 30.0,
    },
    "fertile_optimal": {
        "_description": "Near-perfect field conditions — FHS should score EXCELLENT.",
        "soil_ph": 6.4,
        "soil_moisture": 58.0,
        "soil_temperature": 26.0,
        "nitrogen_ppm": 32.0,
        "phosphorus_ppm": 22.0,
        "potassium_ppm": 195.0,
        "electrical_conductivity": 1.2,
        "organic_matter_percent": 4.2,
        "ambient_humidity": 62.0,
        "ambient_temperature": 28.0,
    },
    "nutrient_poor": {
        "_description": "Depleted soil — widespread macro and micronutrient deficiency.",
        "soil_ph": 5.6,           # WARNING low
        "soil_moisture": 45.0,
        "soil_temperature": 25.0,
        "nitrogen_ppm": 9.0,      # CRITICAL low
        "phosphorus_ppm": 5.0,    # CRITICAL low
        "potassium_ppm": 55.0,    # CRITICAL low
        "electrical_conductivity": 0.3,  # WARNING low
        "organic_matter_percent": 0.9,   # CRITICAL low
        "ambient_humidity": 57.0,
        "ambient_temperature": 27.0,
    },
}


# ---------------------------------------------------------------------------
# Internal parameter variation config
# ---------------------------------------------------------------------------

# Format: param_key -> (center, half_range)
# Used by random_nominal() to generate realistic in-band readings.
_NOMINAL_PARAMS: dict[str, tuple[float, float]] = {
    "soil_ph":                 (6.4,   0.3),
    "soil_moisture":           (57.0,  6.0),
    "soil_temperature":        (25.0,  3.0),
    "nitrogen_ppm":            (30.0,  5.0),
    "phosphorus_ppm":          (21.0,  4.0),
    "potassium_ppm":           (195.0, 25.0),
    "electrical_conductivity": (1.15,  0.20),
    "organic_matter_percent":  (4.0,   0.5),
    "ambient_humidity":        (62.0,  8.0),
    "ambient_temperature":     (27.5,  3.0),
}

# For stressed generation: per-param deviation multiplier beyond the optimal boundary
_STRESS_MAGNITUDE: dict[str, dict[str, float]] = {
    "soil_ph":                 {"low_push": 1.0, "high_push": 1.5},
    "soil_moisture":           {"low_push": 25.0, "high_push": 20.0},
    "soil_temperature":        {"low_push": 8.0, "high_push": 10.0},
    "nitrogen_ppm":            {"low_push": 12.0, "high_push": 18.0},
    "phosphorus_ppm":          {"low_push": 10.0, "high_push": 15.0},
    "potassium_ppm":           {"low_push": 70.0, "high_push": 160.0},
    "electrical_conductivity": {"low_push": 0.3, "high_push": 1.8},
    "organic_matter_percent":  {"low_push": 2.0, "high_push": 4.0},
    "ambient_humidity":        {"low_push": 25.0, "high_push": 18.0},
    "ambient_temperature":     {"low_push": 14.0, "high_push": 8.0},
}

# Optimal boundaries (mirrors the JSON — duplicated here to avoid file I/O in this module)
_OPT_BOUNDS: dict[str, tuple[float, float]] = {
    "soil_ph":                 (5.8, 7.0),
    "soil_moisture":           (40.0, 70.0),
    "soil_temperature":        (18.0, 32.0),
    "nitrogen_ppm":            (20.0, 45.0),
    "phosphorus_ppm":          (15.0, 30.0),
    "potassium_ppm":           (120.0, 280.0),
    "electrical_conductivity": (0.5, 1.8),
    "organic_matter_percent":  (3.0, 6.0),
    "ambient_humidity":        (45.0, 80.0),
    "ambient_temperature":     (22.0, 33.0),
}


# ---------------------------------------------------------------------------
# MockLoRaNode class
# ---------------------------------------------------------------------------

class MockLoRaNode:
    """
    Simulates a single LoRa sensor node transmitting periodic telemetry.

    Args:
        node_id   : Unique identifier for this node (e.g. "NODE_01").
        seed      : Optional random seed for reproducible output.
    """

    def __init__(self, node_id: str = "NODE_01", seed: int | None = None) -> None:
        self.node_id = node_id
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public generators
    # ------------------------------------------------------------------

    def random_nominal(self, noise_factor: float = 0.8) -> dict[str, Any]:
        """
        Generate a telemetry payload with all parameters near-optimal.

        Args:
            noise_factor : 0.0 = no noise (pure center values),
                           1.0 = full half-range noise.

        Returns:
            Dict with all parameter readings + node metadata.
        """
        payload: dict[str, Any] = {}
        for param, (center, half_range) in _NOMINAL_PARAMS.items():
            noise = self._rng.uniform(-1.0, 1.0) * half_range * noise_factor
            value = round(center + noise, 2)
            payload[param] = value
        return self._attach_metadata(payload, mode="nominal")

    def random_stressed(
        self,
        stress_count: int = 3,
        allow_critical: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a telemetry payload with a random subset of parameters
        pushed outside their optimal range.

        Args:
            stress_count  : Number of parameters to stress (1–10).
            allow_critical: If True, some stresses may reach CRITICAL level.

        Returns:
            Dict with telemetry readings + node metadata.
        """
        # Start from a nominal base
        payload = self.random_nominal()
        del payload["node_id"], payload["timestamp"], payload["mode"]

        params_to_stress = self._rng.sample(list(_OPT_BOUNDS.keys()), k=min(stress_count, len(_OPT_BOUNDS)))

        for param in params_to_stress:
            opt_min, opt_max = _OPT_BOUNDS[param]
            mag = _STRESS_MAGNITUDE[param]
            direction = self._rng.choice(["low", "high"])

            if direction == "low":
                push = mag["low_push"]
                if allow_critical and self._rng.random() > 0.5:
                    push *= self._rng.uniform(1.2, 2.0)   # push into CRITICAL
                value = opt_min - push
            else:
                push = mag["high_push"]
                if allow_critical and self._rng.random() > 0.5:
                    push *= self._rng.uniform(1.2, 2.0)
                value = opt_max + push

            # Clamp to physically plausible range
            value = max(0.0, round(value, 2))
            payload[param] = value

        return self._attach_metadata(payload, mode="stressed")

    def scenario(self, name: str) -> dict[str, Any]:
        """
        Return a named deterministic scenario payload.

        Available scenarios:
            acid_soil, drought_stress, waterlogged, saline_soil,
            fertile_optimal, nutrient_poor

        Raises:
            KeyError if the scenario name is not recognised.
        """
        if name not in _SCENARIOS:
            available = list(_SCENARIOS.keys())
            raise KeyError(
                f"Unknown scenario '{name}'. Available: {available}"
            )
        payload = {k: v for k, v in _SCENARIOS[name].items() if not k.startswith("_")}
        return self._attach_metadata(payload, mode=f"scenario:{name}")

    def stream(
        self,
        count: int = 5,
        interval_seconds: float = 0.0,
        mode: str = "mixed",
    ) -> list[dict[str, Any]]:
        """
        Generate a list of successive telemetry payloads, simulating a data stream.

        Args:
            count            : Number of payloads to generate.
            interval_seconds : Sleep between yields (0 = no sleep, useful for UI).
            mode             : 'nominal' | 'stressed' | 'mixed'

        Returns:
            List of payload dicts.
        """
        payloads = []
        for i in range(count):
            if mode == "nominal":
                p = self.random_nominal()
            elif mode == "stressed":
                p = self.random_stressed()
            else:  # mixed — 60% nominal, 40% stressed
                p = self.random_nominal() if self._rng.random() < 0.6 else self.random_stressed()
            payloads.append(p)
            if interval_seconds > 0 and i < count - 1:
                time.sleep(interval_seconds)
        return payloads

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _attach_metadata(self, payload: dict[str, Any], mode: str) -> dict[str, Any]:
        """Attach node identification and timing metadata to a payload."""
        return {
            "node_id": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            **payload,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_scenario_names() -> list[str]:
    """Return the list of all available named scenario keys."""
    return list(_SCENARIOS.keys())


def get_scenario_description(name: str) -> str:
    """Return the human-readable description of a named scenario."""
    if name not in _SCENARIOS:
        raise KeyError(f"Scenario '{name}' not found.")
    return _SCENARIOS[name].get("_description", "No description available.")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    node = MockLoRaNode(node_id="NODE_01", seed=42)

    print("=" * 60)
    print("NOMINAL READING")
    print("=" * 60)
    import json as _json
    print(_json.dumps(node.random_nominal(), indent=2))

    print("\n" + "=" * 60)
    print("STRESSED READING (3 parameters flagged)")
    print("=" * 60)
    print(_json.dumps(node.random_stressed(stress_count=3), indent=2))

    print("\n" + "=" * 60)
    print("SCENARIO: acid_soil")
    print("=" * 60)
    print(_json.dumps(node.scenario("acid_soil"), indent=2))
