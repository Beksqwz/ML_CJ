"""Small invariant tests for reusable Stage 19D metric helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.run_stage19d_evaluation_audit import pr_auc, roc_auc


class Stage19DAuditTests(unittest.TestCase):
    def test_perfect_ranking_metrics_are_one(self) -> None:
        y = np.array([1, 1, 0, 0], dtype=np.int8)
        score = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertEqual(pr_auc(y, score), 1.0)
        self.assertEqual(roc_auc(y, score), 1.0)


if __name__ == "__main__":
    unittest.main()
