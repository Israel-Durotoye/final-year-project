"""
test_evaluator.py — Soil Doctor: Evaluator Unit Tests

Tests every mathematical contract in evaluator.py:
  - Status classification (OK / WARNING / CRITICAL) for each direction
  - Deviation computation (absolute and relative)
  - Parameter score continuity and boundary values
  - Composite Field Health Score correctness
  - LLM alert block structure
  - Named scenario evaluation (end-to-end regression)
  - Edge cases: missing data, non-numeric data, single-parameter payloads

Run with:
    cd soil_doctor
    python -m pytest tests/test_evaluator.py -v
"""

import sys
import os
import math
import pytest

# Resolve import path for the backend module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.prescriptive.evaluator import (
    ThresholdEvaluator,
    ParameterStatus,
    DeviationDirection,
    EvaluationReport,
    ParameterFlag,
)
from backend.ml.mock_lora import MockLoRaNode, get_scenario_names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evaluator() -> ThresholdEvaluator:
    return ThresholdEvaluator()


@pytest.fixture(scope="module")
def node() -> MockLoRaNode:
    return MockLoRaNode(node_id="TEST_NODE", seed=0)


def strip_meta(payload: dict) -> dict:
    """Remove LoRa metadata keys before passing to evaluator."""
    return {k: v for k, v in payload.items() if k not in ("node_id", "timestamp", "mode")}


# ---------------------------------------------------------------------------
# 1. Classification logic
# ---------------------------------------------------------------------------

class TestClassify:
    """Unit tests for ThresholdEvaluator._classify()"""

    def test_ok_at_optimal_midpoint(self, evaluator):
        # Midpoint of pH optimal range (5.8–7.0) = 6.4
        status, direction = evaluator._classify(6.4, 5.8, 7.0, 5.0, 8.0)
        assert status == ParameterStatus.OK
        assert direction == DeviationDirection.NONE

    def test_ok_at_optimal_lower_boundary(self, evaluator):
        status, direction = evaluator._classify(5.8, 5.8, 7.0, 5.0, 8.0)
        assert status == ParameterStatus.OK

    def test_ok_at_optimal_upper_boundary(self, evaluator):
        status, direction = evaluator._classify(7.0, 5.8, 7.0, 5.0, 8.0)
        assert status == ParameterStatus.OK

    def test_warning_low(self, evaluator):
        # pH 5.4 is below optimal (5.8) but above critical (5.0)
        status, direction = evaluator._classify(5.4, 5.8, 7.0, 5.0, 8.0)
        assert status == ParameterStatus.WARNING
        assert direction == DeviationDirection.LOW

    def test_warning_high(self, evaluator):
        # pH 7.5 is above optimal (7.0) but below critical (8.0)
        status, direction = evaluator._classify(7.5, 5.8, 7.0, 5.0, 8.0)
        assert status == ParameterStatus.WARNING
        assert direction == DeviationDirection.HIGH

    def test_critical_low(self, evaluator):
        # pH 4.6 is below critical minimum (5.0)
        status, direction = evaluator._classify(4.6, 5.8, 7.0, 5.0, 8.0)
        assert status == ParameterStatus.CRITICAL
        assert direction == DeviationDirection.LOW

    def test_critical_high(self, evaluator):
        # pH 8.5 is above critical maximum (8.0)
        status, direction = evaluator._classify(8.5, 5.8, 7.0, 5.0, 8.0)
        assert status == ParameterStatus.CRITICAL
        assert direction == DeviationDirection.HIGH

    def test_at_critical_boundary_is_critical(self, evaluator):
        # Exactly at critical_min should be CRITICAL, not WARNING
        status, _ = evaluator._classify(5.0, 5.8, 7.0, 5.0, 8.0)
        assert status == ParameterStatus.CRITICAL


# ---------------------------------------------------------------------------
# 2. Deviation computation
# ---------------------------------------------------------------------------

class TestDeviation:

    def test_no_deviation_inside_range(self, evaluator):
        abs_dev, rel_dev = evaluator._compute_deviation(6.4, 5.8, 7.0)
        assert abs_dev == 0.0
        assert rel_dev == 0.0

    def test_absolute_deviation_low(self, evaluator):
        # observed 5.4, opt_min 5.8 → abs_dev = 0.4
        abs_dev, rel_dev = evaluator._compute_deviation(5.4, 5.8, 7.0)
        assert abs_dev == pytest.approx(0.4, abs=1e-6)

    def test_absolute_deviation_high(self, evaluator):
        # observed 7.5, opt_max 7.0 → abs_dev = 0.5
        abs_dev, rel_dev = evaluator._compute_deviation(7.5, 5.8, 7.0)
        assert abs_dev == pytest.approx(0.5, abs=1e-6)

    def test_relative_deviation_calculation(self, evaluator):
        # opt range = 7.0 - 5.8 = 1.2; abs_dev = 0.4; rel = 0.4/1.2*100 ≈ 33.3%
        abs_dev, rel_dev = evaluator._compute_deviation(5.4, 5.8, 7.0)
        assert rel_dev == pytest.approx(33.33, abs=0.1)

    def test_deviation_at_optimal_boundary(self, evaluator):
        # Exactly at boundary → zero deviation
        abs_dev, rel_dev = evaluator._compute_deviation(5.8, 5.8, 7.0)
        assert abs_dev == 0.0


# ---------------------------------------------------------------------------
# 3. Parameter score continuity
# ---------------------------------------------------------------------------

class TestParameterScore:

    def test_score_at_optimal_midpoint_is_100(self, evaluator):
        # Midpoint of [5.8, 7.0] = 6.4
        score = evaluator._compute_parameter_score(6.4, 5.8, 7.0, 5.0, 8.0)
        assert score == pytest.approx(100.0, abs=0.1)

    def test_score_at_optimal_boundary_is_70(self, evaluator):
        score_low = evaluator._compute_parameter_score(5.8, 5.8, 7.0, 5.0, 8.0)
        score_high = evaluator._compute_parameter_score(7.0, 5.8, 7.0, 5.0, 8.0)
        assert score_low == pytest.approx(70.0, abs=0.5)
        assert score_high == pytest.approx(70.0, abs=0.5)

    def test_score_at_critical_boundary_is_30(self, evaluator):
        score_low = evaluator._compute_parameter_score(5.0, 5.8, 7.0, 5.0, 8.0)
        score_high = evaluator._compute_parameter_score(8.0, 5.8, 7.0, 5.0, 8.0)
        assert score_low == pytest.approx(30.0, abs=0.5)
        assert score_high == pytest.approx(30.0, abs=0.5)

    def test_score_beyond_critical_approaches_zero(self, evaluator):
        score = evaluator._compute_parameter_score(1.0, 5.8, 7.0, 5.0, 8.0)
        assert score < 10.0

    def test_score_is_monotonically_decreasing_away_from_midpoint(self, evaluator):
        midpoint = 6.4
        prev_score = evaluator._compute_parameter_score(midpoint, 5.8, 7.0, 5.0, 8.0)
        for ph in [6.0, 5.8, 5.4, 5.0, 4.5, 3.0]:
            score = evaluator._compute_parameter_score(ph, 5.8, 7.0, 5.0, 8.0)
            assert score <= prev_score, f"Score should decrease: {ph} gave {score} > {prev_score}"
            prev_score = score

    def test_score_never_negative(self, evaluator):
        for extreme in [-100.0, 0.0, 50.0, 100.0]:
            score = evaluator._compute_parameter_score(extreme, 5.8, 7.0, 5.0, 8.0)
            assert score >= 0.0


# ---------------------------------------------------------------------------
# 4. Full evaluation — scenario-based
# ---------------------------------------------------------------------------

class TestEvaluateScenarios:

    def test_fertile_optimal_scores_high(self, evaluator, node):
        payload = strip_meta(node.scenario("fertile_optimal"))
        report = evaluator.evaluate(payload)
        assert report.field_health_score >= 80.0
        assert len(report.critical_parameters) == 0
        assert not report.requires_intervention

    def test_acid_soil_has_critical_ph(self, evaluator, node):
        payload = strip_meta(node.scenario("acid_soil"))
        report = evaluator.evaluate(payload)
        assert report.requires_intervention
        # pH should be CRITICAL
        ph_flag = next((f for f in report.flags if f.param_key == "soil_ph"), None)
        assert ph_flag is not None
        assert ph_flag.status == ParameterStatus.CRITICAL
        assert ph_flag.direction == DeviationDirection.LOW

    def test_drought_stress_has_critical_moisture(self, evaluator, node):
        payload = strip_meta(node.scenario("drought_stress"))
        report = evaluator.evaluate(payload)
        moisture_flag = next((f for f in report.flags if f.param_key == "soil_moisture"), None)
        assert moisture_flag is not None
        assert moisture_flag.status == ParameterStatus.CRITICAL
        assert moisture_flag.direction == DeviationDirection.LOW

    def test_nutrient_poor_has_critical_npk(self, evaluator, node):
        payload = strip_meta(node.scenario("nutrient_poor"))
        report = evaluator.evaluate(payload)
        critical_keys = {f.param_key for f in report.flags if f.status == ParameterStatus.CRITICAL}
        assert "nitrogen_ppm" in critical_keys
        assert "phosphorus_ppm" in critical_keys
        assert "potassium_ppm" in critical_keys

    def test_saline_soil_has_critical_ec(self, evaluator, node):
        payload = strip_meta(node.scenario("saline_soil"))
        report = evaluator.evaluate(payload)
        ec_flag = next((f for f in report.flags if f.param_key == "electrical_conductivity"), None)
        assert ec_flag is not None
        assert ec_flag.status == ParameterStatus.CRITICAL
        assert ec_flag.direction == DeviationDirection.HIGH

    def test_field_health_score_is_bounded(self, evaluator, node):
        for scenario in get_scenario_names():
            payload = strip_meta(node.scenario(scenario))
            report = evaluator.evaluate(payload)
            assert 0.0 <= report.field_health_score <= 100.0, (
                f"FHS out of range for scenario '{scenario}': {report.field_health_score}"
            )

    def test_health_band_is_always_set(self, evaluator, node):
        valid_bands = {"EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL"}
        for scenario in get_scenario_names():
            payload = strip_meta(node.scenario(scenario))
            report = evaluator.evaluate(payload)
            assert report.health_band in valid_bands


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_missing_parameter_is_recorded(self, evaluator):
        # Provide all params except nitrogen_ppm
        payload = strip_meta(MockLoRaNode(seed=1).scenario("fertile_optimal"))
        del payload["nitrogen_ppm"]
        report = evaluator.evaluate(payload)
        assert "Available Nitrogen (N)" in report.missing_parameters

    def test_none_value_treated_as_missing(self, evaluator):
        payload = strip_meta(MockLoRaNode(seed=1).scenario("fertile_optimal"))
        payload["soil_ph"] = None
        report = evaluator.evaluate(payload)
        assert "Soil pH" in report.missing_parameters

    def test_non_numeric_value_treated_as_missing(self, evaluator):
        payload = strip_meta(MockLoRaNode(seed=1).scenario("fertile_optimal"))
        payload["soil_temperature"] = "ERROR"
        report = evaluator.evaluate(payload)
        assert "Soil Temperature" in report.missing_parameters

    def test_empty_payload_all_missing(self, evaluator):
        report = evaluator.evaluate({})
        assert len(report.missing_parameters) > 0
        assert report.field_health_score == 0.0

    def test_unknown_crop_raises_key_error(self, evaluator):
        with pytest.raises(KeyError):
            evaluator.evaluate({"soil_ph": 6.5}, crop="cassava_tropical")

    def test_extra_keys_in_payload_are_ignored(self, evaluator):
        payload = strip_meta(MockLoRaNode(seed=1).scenario("fertile_optimal"))
        payload["unknown_sensor_xyz"] = 999.0
        # Should not raise
        report = evaluator.evaluate(payload)
        assert isinstance(report, EvaluationReport)


# ---------------------------------------------------------------------------
# 6. LLM alert block structure
# ---------------------------------------------------------------------------

class TestLLMAlertBlock:

    def test_alert_block_contains_fhs(self, evaluator, node):
        payload = strip_meta(node.scenario("acid_soil"))
        report = evaluator.evaluate(payload)
        assert "Field Health" in report.llm_alert_block
        assert str(int(report.field_health_score)) in report.llm_alert_block

    def test_alert_block_contains_critical_header(self, evaluator, node):
        payload = strip_meta(node.scenario("acid_soil"))
        report = evaluator.evaluate(payload)
        assert "CRITICAL FLAGS" in report.llm_alert_block

    def test_alert_block_contains_nominal_section_for_optimal(self, evaluator, node):
        payload = strip_meta(node.scenario("fertile_optimal"))
        report = evaluator.evaluate(payload)
        assert "NOMINAL PARAMETERS" in report.llm_alert_block

    def test_intervention_hint_present_in_alert_block_for_flagged(self, evaluator, node):
        payload = strip_meta(node.scenario("drought_stress"))
        report = evaluator.evaluate(payload)
        # The hint for low soil_moisture should mention irrigation
        assert "irrigation" in report.llm_alert_block.lower() or "irrigat" in report.llm_alert_block.lower()

    def test_alert_block_starts_and_ends_correctly(self, evaluator, node):
        payload = strip_meta(node.scenario("fertile_optimal"))
        report = evaluator.evaluate(payload)
        assert report.llm_alert_block.strip().startswith("=== FIELD EVALUATION REPORT ===")
        assert report.llm_alert_block.strip().endswith("=== END EVALUATION REPORT ===")


# ---------------------------------------------------------------------------
# 7. as_summary_dict contract
# ---------------------------------------------------------------------------

class TestSummaryDict:

    def test_summary_dict_has_required_keys(self, evaluator, node):
        payload = strip_meta(node.scenario("acid_soil"))
        report = evaluator.evaluate(payload)
        summary = report.as_summary_dict()
        required_keys = {
            "crop_profile", "field_health_score", "health_band",
            "requires_intervention", "critical_count", "warning_count",
        }
        assert required_keys.issubset(summary.keys())

    def test_summary_counts_match_lists(self, evaluator, node):
        payload = strip_meta(node.scenario("nutrient_poor"))
        report = evaluator.evaluate(payload)
        summary = report.as_summary_dict()
        assert summary["critical_count"] == len(report.critical_parameters)
        assert summary["warning_count"] == len(report.warning_parameters)
