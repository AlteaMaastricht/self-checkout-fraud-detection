import unittest

import numpy as np
import pandas as pd

from tests import _path  # noqa: F401
from checkout_anomaly.ensemble import AnomalyEnsemble, CheckoutAnomalyPipeline, combine_normalised_scores
from checkout_anomaly.models import BaseDetector
from checkout_anomaly.synthetic import VisitGenerator


class ColumnDetector(BaseDetector):
    """Stub: the anomaly score is one raw input column (no learning)."""

    def __init__(self, name, column):
        self.name = name
        self.column = column

    def fit(self, X):
        self.scaler_ = None
        return self

    def score(self, X):
        return np.asarray(X[self.column], dtype=float)


def original_buggy_ensemble(normalized_scores):
    """Replica of the control flow of the course code (returns inside the loop)."""
    for method, scores in normalized_scores.items():
        if normalized_scores:
            first = list(normalized_scores.values())[0]
            ens = np.zeros_like(first)
            for s in normalized_scores.values():
                ens += s
            ens /= len(normalized_scores)
            return ens
    # NB: in the original, the dict was still being filled while this ran


class EnsembleTests(unittest.TestCase):
    def test_combine_uses_all_methods(self):
        s = {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 1.0]), "c": np.array([0.5, 0.0])}
        np.testing.assert_allclose(combine_normalised_scores(s), [0.5, 2 / 3])
        np.testing.assert_allclose(combine_normalised_scores(s, {"a": 1, "b": 0, "c": 1}), [0.25, 0.5])

    def test_regression_ensemble_not_first_method_only(self):
        # The bug: normalised scores were filled one method at a time and the function
        # returned in the first iteration -> "ensemble" == first method only.
        partial = {}
        buggy = None
        for name, s in {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 0.0])}.items():
            partial[name] = s
            buggy = original_buggy_ensemble(partial)
            break
        np.testing.assert_allclose(buggy, [0.0, 1.0])  # what the original returned
        fixed = combine_normalised_scores({"a": np.array([0.0, 1.0]), "b": np.array([1.0, 0.0])})
        np.testing.assert_allclose(fixed, [0.5, 0.5])

    def test_ensemble_combines_every_detector(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["x", "y", "z"])
        dets = [ColumnDetector("dx", "x"), ColumnDetector("dy", "y"), ColumnDetector("dz", "z")]
        ens = AnomalyEnsemble(dets, contamination=0.1).fit(X)
        comps = ens.score_components(X)
        self.assertEqual(list(comps.columns), ["dx", "dy", "dz"])
        np.testing.assert_allclose(ens.score(X), comps.mean(axis=1).to_numpy())
        self.assertFalse(np.allclose(ens.score(X), comps["dx"].to_numpy()))

    def test_threshold_fitted_on_train_only(self):
        rng = np.random.default_rng(1)
        train = pd.DataFrame({"x": rng.normal(size=1000)})
        ens = AnomalyEnsemble([ColumnDetector("dx", "x")], contamination=0.15).fit(train)
        thr = ens.threshold_
        self.assertAlmostEqual(ens.predict(train).mean(), 0.15, delta=0.01)
        # a test file full of ordinary values -> almost nothing flagged (original: always 15 %)
        calm = pd.DataFrame({"x": rng.normal(size=500) * 0.1})
        self.assertLess(ens.predict(calm).mean(), 0.01)
        # a test file of extreme values -> (almost) everything flagged
        wild = pd.DataFrame({"x": rng.normal(size=500) + 5})
        self.assertGreater(ens.predict(wild).mean(), 0.99)
        self.assertEqual(ens.threshold_, thr)  # scoring never refits the threshold
        # min-max constants come from training scores, so test scores may exceed 1
        self.assertGreater(ens.score_components(wild)["dx"].max(), 1.0)

    def test_rank_normalisation(self):
        train = pd.DataFrame({"x": np.arange(100, dtype=float)})
        ens = AnomalyEnsemble([ColumnDetector("dx", "x")], normalisation="rank").fit(train)
        np.testing.assert_allclose(ens.score(pd.DataFrame({"x": [-1.0, 49.0, 1000.0]})), [0.0, 0.5, 1.0])

    def test_pipeline_end_to_end(self):
        gen = VisitGenerator(seed=3)
        train = gen.generate(300, fraud_rate=0.0)
        test = gen.generate(60, fraud_rate=0.2, id_offset=300)
        pipe = CheckoutAnomalyPipeline(random_state=0).fit(train.visits)
        out = pipe.score(test.visits)
        self.assertEqual(len(out), 60)
        self.assertIn("isolation_forest", out.columns)
        self.assertIn("kmeans_distance", out.columns)
        self.assertEqual(len(pipe.top_k(test.visits, k=10)), 10)
        self.assertEqual(list(pipe.features(test.visits).columns), pipe.filter.columns_)


if __name__ == "__main__":
    unittest.main()
