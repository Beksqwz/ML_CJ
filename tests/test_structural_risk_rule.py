import unittest

from recommendations.engine import recommend


def _run(values):
    return recommend(
        probability=0.01,
        shap_values={},
        feature_values=values,
        model_horizon="1h",
        final_model_version="test",
    )


class StructuralRiskRuleTests(unittest.TestCase):
    def test_structural_risk_flag_without_history(self):
        result = _run(
            {"segment_accidents_total_prior": 0, "speed_infra_mismatch": True}
        )
        self.assertIn(
            "structural_risk_flag", {x["rule"] for x in result["recommendations"]}
        )

    def test_structural_risk_flag_requires_a_factor(self):
        result = _run(
            {"segment_accidents_total_prior": 0, "speed_infra_mismatch": False}
        )
        self.assertNotIn(
            "structural_risk_flag", {x["rule"] for x in result["recommendations"]}
        )
