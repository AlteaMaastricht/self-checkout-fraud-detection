import math
import unittest

import numpy as np

from tests import _path  # noqa: F401
from checkout_anomaly.features import CorrelationFilter, FeatureExtractor, gini
from checkout_anomaly.parsing import Visit


def visit(vid, items):
    return Visit(visit_id=vid, items=[(int(d), float(t), float(p)) for d, t, p in items])


TRAIN = [
    visit("a", [(1, 10, 2.0), (1, 20, 4.0), (3, 70, 1.0)]),
    visit("b", [(1, 5, 2.0), (3, 15, 3.0)]),
    visit("c", [(2, 8, 10.0), (2, 9, 12.0), (1, 30, 3.0), (3, 2, 2.0)]),
]


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureExtractor(n_departments=3, n_high_value=1).fit(TRAIN)

    def test_basic_aggregations_on_hand_made_visit(self):
        f = self.fe.visit_features(TRAIN[0])
        self.assertEqual(f["total_items"], 3)
        self.assertAlmostEqual(f["total_spent"], 7.0)
        self.assertAlmostEqual(f["visit_duration"], 100.0)
        self.assertAlmostEqual(f["max_time_delta"], 70.0)
        self.assertEqual(f["very_long_scans"], 1)  # 70 s > 60 s
        self.assertEqual(f["very_short_scans"], 0)
        self.assertEqual(f["unique_departments"], 2)
        self.assertEqual(f["dept_switches"], 1)
        self.assertEqual(f["max_dept_distance"], 2)
        self.assertEqual(f["dept_1_count"], 2)
        self.assertAlmostEqual(f["dept_1_spent"], 6.0)
        self.assertAlmostEqual(f["dept_1_pct"], 2 / 3)
        expected_entropy = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
        self.assertAlmostEqual(f["dept_entropy"], expected_entropy)
        self.assertAlmostEqual(f["std_time_delta"], np.std([10, 20, 70], ddof=1))

    def test_population_statistics_learned_from_training_only(self):
        # department 2 has the highest mean price in TRAIN -> learned as high-value
        self.assertEqual(self.fe.high_value_, [2])
        self.assertAlmostEqual(self.fe.dept_avg_price_[1], (2 + 4 + 2 + 3) / 4)
        before = self.fe.dept_avg_price_.copy()
        test = [visit("t", [(1, 10, 100.0), (1, 10, 100.0)])]
        f = self.fe.transform(test).iloc[0]
        np.testing.assert_array_equal(before, self.fe.dept_avg_price_)  # transform does not refit
        self.assertAlmostEqual(f["dept_relative_price_max"], 100.0 / 2.75)

    def test_markov_sequence_nll(self):
        # transitions in TRAIN from dept 1: 1->1, 1->3, 1->3, 1->3 => P(1->3) = 3/4
        f = self.fe.visit_features(visit("s", [(1, 1, 1.0), (3, 1, 1.0)]))
        self.assertAlmostEqual(f["sequence_nll"], -math.log(0.75))
        unseen = self.fe.visit_features(visit("u", [(3, 1, 1.0), (2, 1, 1.0)]))
        self.assertAlmostEqual(unseen["sequence_nll"], -math.log(1e-10))

    def test_single_item_visit(self):
        f = self.fe.visit_features(visit("one", [(2, 15, 3.0)]))
        self.assertEqual(f["total_items"], 1)
        self.assertEqual(f["std_time_delta"], 0.0)
        self.assertEqual(f["sequence_nll"], 0.0)
        self.assertTrue(all(np.isfinite(v) for v in f.values()))

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            FeatureExtractor().transform(TRAIN)

    def test_gini(self):
        self.assertEqual(gini(np.array([5.0, 5.0, 5.0])), 0.0)
        self.assertGreater(gini(np.array([0.0, 0.0, 10.0])), 0.5)

    def test_correlation_filter_fitted_on_train(self):
        import pandas as pd
        X = pd.DataFrame({"a": [1, 2, 3, 4.0], "b": [2, 4, 6, 8.0], "c": [1, 0, 1, 0.0], "k": [1, 1, 1, 1.0]})
        cf = CorrelationFilter(0.95).fit(X)
        self.assertEqual(cf.columns_, ["a", "c"])
        Y = pd.DataFrame({"a": [9.0], "c": [1.0]})  # missing columns are filled, order preserved
        self.assertEqual(list(cf.transform(Y).columns), ["a", "c"])


if __name__ == "__main__":
    unittest.main()
