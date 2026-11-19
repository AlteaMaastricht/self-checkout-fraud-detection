import importlib.util
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from tests import _path  # noqa: F401
from checkout_anomaly.models import DETECTORS, KerasAutoencoderDetector, make_detector

HAS_TF = importlib.util.find_spec("tensorflow") is not None


class DetectorSanityTests(unittest.TestCase):
    """Every detector must rank gross outliers above inliers (a sanity check, not a benchmark)."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.train = rng.normal(size=(600, 6))
        inliers = rng.normal(size=(200, 6))
        outliers = rng.normal(size=(20, 6)) + rng.choice([-6, 6], size=(20, 6))
        cls.test = np.vstack([inliers, outliers])
        cls.y = np.r_[np.zeros(200), np.ones(20)]

    def test_sklearn_detectors(self):
        for name in ("isolation_forest", "kmeans_distance", "mlp_autoencoder", "lof", "one_class_svm", "gmm"):
            with self.subTest(detector=name):
                det = make_detector(name).fit(self.train)
                s = det.score(self.test)
                self.assertEqual(s.shape, (220,))
                self.assertGreater(roc_auc_score(self.y, s), 0.95)

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            make_detector("isolation_forest").score(self.test)

    def test_unknown_detector(self):
        with self.assertRaises(ValueError):
            make_detector("nope")

    @unittest.skipIf(HAS_TF, "TensorFlow installed; import-guard path not reachable")
    def test_keras_import_guard(self):
        with self.assertRaises(ImportError):
            KerasAutoencoderDetector(epochs=1).fit(self.train)

    @unittest.skipUnless(HAS_TF, "NOT EXECUTED - needs tensorflow (optional dependency)")
    def test_keras_autoencoder(self):  # pragma: no cover
        det = KerasAutoencoderDetector(epochs=5).fit(self.train)
        self.assertGreater(roc_auc_score(self.y, det.score(self.test)), 0.95)

    def test_registry(self):
        self.assertIn("keras_autoencoder", DETECTORS)


if __name__ == "__main__":
    unittest.main()
