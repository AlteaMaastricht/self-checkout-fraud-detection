import unittest

import numpy as np
import pandas as pd

from tests import _path  # noqa: F401
from checkout_anomaly.evaluation import (evaluate_synthetic, feature_perturbation_check,
                                         precision_recall_at_k, ranking_metrics, summarise)


class MetricTests(unittest.TestCase):
    def test_precision_recall_at_k(self):
        y = np.array([1, 0, 1, 0, 0], dtype=bool)
        s = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
        self.assertEqual(precision_recall_at_k(y, s, 2), (0.5, 0.5))
        self.assertEqual(precision_recall_at_k(y, s, 5), (0.4, 1.0))

    def test_ranking_metrics_perfect(self):
        m = ranking_metrics(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9]), ks=(1,))
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertEqual(m["precision_at_1"], 1.0)


class SyntheticEvaluationTests(unittest.TestCase):
    """Small run; the measured numbers are printed, not tuned. See results/ for the full demo."""

    @classmethod
    def setUpClass(cls):
        cls.r = evaluate_synthetic(seed=11, n_train=800, n_test=300, test_fraud_rate=0.15, k=50)

    def test_structure_and_protocol(self):
        r = self.r
        self.assertEqual(r["threshold_fitted_on"], "train")
        self.assertEqual(set(r["methods"]), {"isolation_forest", "kmeans_distance", "ensemble"})
        for m in r["methods"].values():
            for key in ("roc_auc", "average_precision", "precision_at_50", "recall_at_50"):
                self.assertTrue(0.0 <= m[key] <= 1.0)
        print("\n[measured, seed 11, 800 train / 300 test] ensemble:",
              {k: round(v, 3) for k, v in self.r["methods"]["ensemble"].items() if not isinstance(v, dict)},
              {k: round(v, 3) for k, v in self.r["methods"]["ensemble"]["auc_by_fraud_type"].items()},
              "at train threshold:", {k: round(v, 3) for k, v in self.r["at_train_threshold"].items()})

    def test_ensemble_ranks_injected_fraud_above_normal(self):
        # The only claim tested: better than random ranking on this generator.
        self.assertGreater(self.r["methods"]["ensemble"]["roc_auc"], 0.5)

    def test_summarise(self):
        df = summarise([self.r])
        self.assertIn(("roc_auc", "mean"), df.columns)

    def test_perturbation_check_runs(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(400, 5)) + 10)
        out = feature_perturbation_check(X, seed=0)
        self.assertTrue(0 <= out["precision"] <= 1 and 0 <= out["recall"] <= 1)


if __name__ == "__main__":
    unittest.main()
