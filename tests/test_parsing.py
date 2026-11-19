import tempfile
import unittest
from pathlib import Path

from tests import _path  # noqa: F401
from checkout_anomaly.parsing import ParseError, iter_visits, parse_visit_line, read_visits, write_visits


class ParsingTests(unittest.TestCase):
    def test_plain_line(self):
        v = parse_visit_line("15 98 4.34, 6 132 1.02, 6 71 1.75\r\n", default_id="7")
        self.assertEqual(v.visit_id, "7")
        self.assertEqual(v.items, [(15, 98.0, 4.34), (6, 132.0, 1.02), (6, 71.0, 1.75)])

    def test_transaction_prefix_keeps_first_item(self):
        # Regression: the original parsers dropped the first tuple of prefixed lines.
        v = parse_visit_line("Transaction 11052:3 40 2.10, 4 12 0.99", default_id="0")
        self.assertEqual(v.visit_id, "11052")
        self.assertEqual(len(v.items), 2)
        self.assertEqual(v.items[0], (3, 40.0, 2.10))

    def test_malformed_tuple_skipped_or_strict(self):
        line = "1 10 2.0, 2 x 1.0, 3 5"
        self.assertEqual(len(parse_visit_line(line, "0").items), 1)
        with self.assertRaises(ParseError):
            parse_visit_line(line, "0", strict=True)

    def test_blank_and_empty_lines_skipped(self):
        visits = list(iter_visits(["1 2 3.0\n", "\n", " , \n", "2 3 4.0\n"]))
        self.assertEqual([v.visit_id for v in visits], ["0", "3"])

    def test_roundtrip(self):
        visits = list(iter_visits(["1 10 2.50, 2 20 1.25", "3 30 9.99"]))
        with tempfile.TemporaryDirectory() as d:
            for with_id in (False, True):
                path = Path(d) / f"v{with_id}.csv"
                write_visits(visits, path, with_id=with_id)
                back = read_visits(path)
                self.assertEqual([v.items for v in back], [v.items for v in visits])


if __name__ == "__main__":
    unittest.main()
