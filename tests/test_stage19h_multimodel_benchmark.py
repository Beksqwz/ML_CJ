import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage19HFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.reports = ROOT / "reports" / "stage19h"
        self.audit = ROOT / "data" / "audit" / "stage19h"

    def test_required_artifacts_and_checkpoint_are_consistent(self):
        for name in (
            "validation_metrics.json",
            "robustness_summary.json",
            "winner_decision.json",
            "summary.json",
        ):
            self.assertTrue((self.reports / name).is_file())
        checkpoint = json.loads((self.audit / "evaluation_checkpoint.json").read_text())
        self.assertEqual(checkpoint["status"], "completed")
        self.assertTrue(checkpoint["reconstructed_from_completed_run"])
        self.assertEqual(len(checkpoint["completed_windows"]), 3)

    def test_frozen_contract_and_winner_rule(self):
        validation = json.loads((self.reports / "validation_metrics.json").read_text())
        decision = json.loads((self.reports / "winner_decision.json").read_text())
        self.assertEqual(validation["feature_count"], 77)
        self.assertFalse(decision["replacement_approved"])
        self.assertEqual(decision["best_production_model"], "stage7b_frozen_reference")

    def test_baseline_and_warnings_are_preserved(self):
        metrics = json.loads(
            (self.reports / "seasonal_natural_metrics.json").read_text()
        )
        efficiency = json.loads(
            (self.reports / "efficiency_comparison.json").read_text()
        )
        self.assertIn("historical_baseline", metrics["autumn"])
        self.assertTrue(efficiency["logistic_regression"]["warnings"])
        self.assertTrue(efficiency["extra_trees"]["warnings"])


if __name__ == "__main__":
    unittest.main()
