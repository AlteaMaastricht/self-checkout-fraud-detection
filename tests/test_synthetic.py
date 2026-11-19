import unittest

import numpy as np

from tests import _path  # noqa: F401
from checkout_anomaly.parsing import format_visit, parse_visit_line
from checkout_anomaly.synthetic import FRAUD_TYPES, GeneratorConfig, VisitGenerator


class GeneratorTests(unittest.TestCase):
    def test_deterministic_for_seed(self):
        a = VisitGenerator(seed=5).generate(50, fraud_rate=0.2)
        b = VisitGenerator(seed=5).generate(50, fraud_rate=0.2)
        self.assertEqual([v.items for v in a.visits], [v.items for v in b.visits])
        self.assertEqual(a.fraud_type, b.fraud_type)

    def test_labels_and_schema(self):
        ds = VisitGenerator(seed=1).generate(200, fraud_rate=0.1)
        self.assertEqual(int(ds.is_fraud.sum()), 20)
        self.assertTrue(set(ds.fraud_type) <= set(FRAUD_TYPES) | {"normal"})
        for v, y, t in zip(ds.visits, ds.is_fraud, ds.fraud_type):
            self.assertEqual(bool(y), t != "normal")
            self.assertGreaterEqual(len(v), 1)
            for d, s, p in v.items:
                self.assertTrue(1 <= d <= 18)
                self.assertGreaterEqual(s, 0)
                self.assertGreater(p, 0)
            back = parse_visit_line(format_visit(v), default_id=v.visit_id)
            self.assertEqual(len(back.items), len(v.items))

    def test_single_pattern_effects(self):
        # skipped scans remove items; pauses add long gaps; under-pricing lowers prices
        for kind in FRAUD_TYPES:
            cfg = GeneratorConfig(fraud_mix={kind: 1.0})
            clean = VisitGenerator(cfg, seed=9).generate(300, fraud_rate=0.0)
            dirty = VisitGenerator(cfg, seed=9).generate(300, fraud_rate=1.0)
            n_c = np.mean([len(v) for v in clean.visits])
            n_d = np.mean([len(v) for v in dirty.visits])
            max_gap_c = np.mean([max(t for _, t, _ in v.items) for v in clean.visits])
            max_gap_d = np.mean([max(t for _, t, _ in v.items) for v in dirty.visits])
            with self.subTest(kind=kind):
                if kind == "skipped_scan":
                    self.assertLess(n_d, n_c)
                if kind == "scan_pause":
                    self.assertGreater(max_gap_d, max_gap_c + 100)

    def test_world_seed_separates_store_from_visits(self):
        a = VisitGenerator(seed=0, world_seed=7)
        b = VisitGenerator(seed=1, world_seed=7)
        np.testing.assert_array_equal(a.price_median, b.price_median)
        self.assertNotEqual([v.items for v in a.generate(20).visits], [v.items for v in b.generate(20).visits])

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            VisitGenerator().generate(10, fraud_rate=1.5)


if __name__ == "__main__":
    unittest.main()
