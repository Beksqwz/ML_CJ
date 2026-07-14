import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Stage19IWeightSearchTests(unittest.TestCase):
    def test_frozen_weighted_config(self):
        p = ROOT / "models" / "stage19i_simple" / "weighted_ensemble_config.json"
        d = json.loads(p.read_text())
        self.assertEqual(d["status"], "frozen")
        self.assertAlmostEqual(sum(d["weights"].values()), 1.0)
        self.assertTrue(all(v >= 0 for v in d["weights"].values()))
        self.assertEqual(d["weights"]["score_logistic_regression"], 0.0)
        self.assertNotIn("seasonal", json.dumps(d).lower())


if __name__ == "__main__":
    unittest.main()
