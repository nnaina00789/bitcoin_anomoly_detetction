"""Core unit and artifact integrity tests.

Run with:  python -m unittest discover tests -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import n_ties, precision_at_k, wilson_ci
from models import blend, to_percentile


class TestMetrics(unittest.TestCase):
    def test_precision_at_k_handles_all_ties_fairly(self):
        # 10 items, all tied at score 0.5, 3 are illicit -> expected P@k = 0.3
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        s = np.full(10, 0.5)
        self.assertAlmostEqual(precision_at_k(y, s, 5), 0.3)

    def test_precision_at_k_partial_tie_at_cutoff(self):
        # Top 2 strictly above tie (both 1), next 4 tied (2 illicit, 2 licit), k=4 -> (2 + 2*0.5)/4 = 0.75
        y = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=float)
        s = np.array([0.9, 0.8, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1])
        self.assertAlmostEqual(precision_at_k(y, s, 4), 0.75)

    def test_wilson_ci_bounds(self):
        lo, hi = wilson_ci(8, 19)
        self.assertTrue(0.0 <= lo < 8 / 19 < hi <= 1.0)
        lo0, hi0 = wilson_ci(0, 0)
        self.assertTrue(np.isnan(lo0) and np.isnan(hi0))

    def test_n_ties(self):
        self.assertEqual(n_ties([1, 1, 2, 3]), 3)


class TestModels(unittest.TestCase):
    def test_to_percentile_range_and_order(self):
        ref = np.array([10.0, 20.0, 30.0, 40.0])
        pct = to_percentile(np.array([5.0, 20.0, 50.0]), ref)
        self.assertTrue(np.all((pct >= 0.0) & (pct <= 1.0)))
        self.assertTrue(pct[0] < pct[1] < pct[2])

    def test_blend_endpoints(self):
        a = np.array([0.2, 0.8])
        b = np.array([0.9, 0.1])
        np.testing.assert_allclose(blend(a, b, 1.0), a)
        np.testing.assert_allclose(blend(a, b, 0.0), b)


class TestArtifacts(unittest.TestCase):
    def test_pipeline_outputs_exist_and_consistent(self):
        res = ROOT / "results"
        for name in ["scores.csv", "metrics.csv", "summary.json", "chains.csv",
                     "flow_edges.csv", "per_step.csv", "verification_summary.csv"]:
            self.assertTrue((res / name).exists(), f"Missing {name}")
        summary = json.loads((res / "summary.json").read_text())
        metrics = pd.read_csv(res / "metrics.csv")
        self.assertAlmostEqual(summary["PR-AUC_final"], float(metrics.iloc[-1]["PR-AUC"]))
        self.assertGreater(int(metrics.iloc[-1]["distinct_scores"]), 1000)


if __name__ == "__main__":
    unittest.main()

