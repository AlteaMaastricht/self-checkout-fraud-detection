"""Visit-level feature engineering.

Ported from the author's course scripts (see ``docs/PROVENANCE.md`` for the
source function of every group):

1. basic aggregations         - item count, spend, duration, time/price quantiles
2. scan-rhythm statistics     - consistency, price/time correlation, CVs
3. department behaviour       - counts, spend and shares per department, switches,
                                code distance between consecutive departments
4. high-value departments     - items and spend in the most expensive departments
5. entropy                    - Shannon entropy (base 2) of departments visited
6. derived ratios             - price range/IQR, spend per second, bursts/pauses
7. temporal phases            - acceleration, Gini of gaps, early/middle/late ratios,
                                quarter volatility
8. price patterns             - Gini, Q1/Q3 shares, run lengths, price-position trend
9. sequence behaviour         - revisits, dwell, ascending/descending trends,
                                price relative to the department average
10. Markov likelihood         - mean negative log-probability of the department
                                transitions under a transition matrix

Everything that depends on the population (department average prices, the
transition matrix, the high-value departments, the correlated-column filter) is
**learned in ``fit`` on training visits only** and re-used in ``transform``.
The original scripts recomputed these on whatever file they were scoring.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .parsing import Visit

SHORT_SCAN_S = 5.0  # "very short" gap
LONG_SCAN_S = 60.0  # "very long" gap / pause
BURST_S = 3.0  # burst gap
DEPT_JUMP = 5  # code distance counted as a jump across the store
PRICE_RUN_TOL = 0.15  # relative tolerance for a "run" of similar prices
TRANSITION_FLOOR = 1e-10


def gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values (0 = equal)."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = arr.size
    if n <= 1 or arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def entropy_base2(depts: np.ndarray) -> float:
    _, counts = np.unique(depts, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def price_runs(prices: np.ndarray, tol: float = PRICE_RUN_TOL) -> tuple[int, float, float]:
    """Max run length, mean run length and run CV of consecutive similar prices."""
    if prices.size < 2:
        return 1, 1.0, 0.0
    runs, cur = [], 1
    for prev, nxt in zip(prices[:-1], prices[1:]):
        if abs(nxt - prev) / max(prev, 1e-6) <= tol:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    r = np.asarray(runs, dtype=float)
    return int(r.max()), float(r.mean()), float(r.std() / r.mean()) if r.mean() > 0 else 0.0


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() <= 1e-6 or b.std() <= 1e-6:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


class FeatureExtractor:
    """Fit population statistics on training visits, then compute ~100 features per visit."""

    def __init__(
        self,
        n_departments: int = 18,
        high_value_departments: Sequence[int] | None = None,
        n_high_value: int = 3,
    ):
        self.n_departments = n_departments
        self.high_value_departments = high_value_departments
        self.n_high_value = n_high_value

    # ------------------------------------------------------------------ fit
    def fit(self, visits: Iterable[Visit]) -> "FeatureExtractor":
        visits = list(visits)
        if not visits:
            raise ValueError("cannot fit on zero visits")
        n = self.n_departments
        sums = np.zeros(n + 1)
        counts = np.zeros(n + 1)
        trans = np.zeros((n + 1, n + 1))
        for v in visits:
            d = np.array([x[0] for x in v.items], dtype=int)
            p = np.array([x[2] for x in v.items], dtype=float)
            d = np.clip(d, 0, n)
            np.add.at(sums, d, p)
            np.add.at(counts, d, 1)
            if d.size > 1:
                np.add.at(trans, (d[:-1], d[1:]), 1)
        self.dept_avg_price_ = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        rows = trans.sum(axis=1, keepdims=True)
        self.transition_matrix_ = np.divide(trans, rows, out=np.zeros_like(trans), where=rows > 0)
        if self.high_value_departments is not None:
            self.high_value_ = sorted(int(x) for x in self.high_value_departments)
        else:  # the k departments with the highest mean training price
            order = np.argsort(self.dept_avg_price_[1:])[::-1] + 1
            self.high_value_ = sorted(int(x) for x in order[: self.n_high_value])
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "transition_matrix_"):
            raise RuntimeError("FeatureExtractor is not fitted; call fit() on training visits first")

    # ------------------------------------------------------------------ transform
    def transform(self, visits: Iterable[Visit]) -> pd.DataFrame:
        self._check_fitted()
        visits = list(visits)
        rows = [self.visit_features(v) for v in visits]
        return pd.DataFrame(rows, index=pd.Index([v.visit_id for v in visits], name="visit_id"))

    def fit_transform(self, visits: Iterable[Visit]) -> pd.DataFrame:
        visits = list(visits)
        return self.fit(visits).transform(visits)

    def visit_features(self, visit: Visit) -> dict[str, float]:
        self._check_fitted()
        if not visit.items:
            raise ValueError(f"visit {visit.visit_id} has no items")
        d = np.array([x[0] for x in visit.items], dtype=int)
        t = np.array([x[1] for x in visit.items], dtype=float)
        p = np.array([x[2] for x in visit.items], dtype=float)
        n_items = d.size
        f: dict[str, float] = {}

        # 1. basic aggregations
        f["total_items"] = n_items
        f["total_spent"] = p.sum()
        f["visit_duration"] = t.sum()
        f["avg_time_delta"] = t.mean()
        f["median_time_delta"] = float(np.median(t))
        f["std_time_delta"] = t.std(ddof=1) if n_items > 1 else 0.0
        f["min_time_delta"] = t.min()
        f["max_time_delta"] = t.max()
        f["p95_time_delta"] = float(np.percentile(t, 95))
        f["avg_price"] = p.mean()
        f["median_price"] = float(np.median(p))
        f["std_price"] = p.std(ddof=1) if n_items > 1 else 0.0
        f["min_price"] = p.min()
        f["max_price"] = p.max()
        p25, p75 = np.percentile(p, [25, 75])
        f["p25_price"], f["p75_price"] = float(p25), float(p75)
        f["unique_departments"] = int(np.unique(d).size)
        f["very_short_scans"] = int((t < SHORT_SCAN_S).sum())
        f["very_long_scans"] = int((t > LONG_SCAN_S).sum())

        # 2. scan rhythm / transitions
        if n_items > 1:
            f["time_consistency"] = float(np.abs(np.diff(t)).mean())
            f["price_time_corr"] = _corr(p, t)
            dist = np.abs(np.diff(d))
            f["dept_switches"] = int((np.diff(d) != 0).sum())
            f["avg_dept_distance"] = float(dist.mean())
            f["max_dept_distance"] = int(dist.max())
            f["dept_jumping"] = int((dist > DEPT_JUMP).sum())
        else:
            for k in ("time_consistency", "price_time_corr", "dept_switches",
                      "avg_dept_distance", "max_dept_distance", "dept_jumping"):
                f[k] = 0.0
        f["time_delta_cv"] = _safe_div(f["std_time_delta"], f["avg_time_delta"])
        f["price_cv"] = _safe_div(f["std_price"], f["avg_price"])

        # 3. department counts / spend / shares
        n = self.n_departments
        dc = np.clip(d, 0, n)
        cnt = np.bincount(dc, minlength=n + 1)
        spent = np.bincount(dc, weights=p, minlength=n + 1)
        for k in range(1, n + 1):
            f[f"dept_{k}_count"] = int(cnt[k])
            f[f"dept_{k}_spent"] = float(spent[k])
            f[f"dept_{k}_pct"] = _safe_div(cnt[k], n_items)
            f[f"dept_{k}_spent_pct"] = _safe_div(spent[k], f["total_spent"])

        # 4. high-value departments (learned in fit)
        hv = np.isin(d, self.high_value_)
        f["high_value_items"] = int(hv.sum())
        f["high_value_spent"] = float(p[hv].sum())
        f["high_value_pct"] = _safe_div(f["high_value_items"], n_items)
        f["high_value_spent_pct"] = _safe_div(f["high_value_spent"], f["total_spent"])

        # 5. entropy
        f["dept_entropy"] = entropy_base2(d)

        # 6. derived ratios
        f["price_range"] = f["max_price"] - f["min_price"]
        f["iqr_price"] = f["p75_price"] - f["p25_price"]
        f["price_time_ratio"] = _safe_div(f["total_spent"], f["visit_duration"])
        f["scan_time_ratio"] = _safe_div(f["max_time_delta"], f["min_time_delta"])
        f["dept_diversity_ratio"] = _safe_div(f["unique_departments"], n_items)
        f["dept_switches_rate"] = _safe_div(f["dept_switches"], n_items - 1)
        f["scan_bursts_pct"] = float((t < BURST_S).mean())
        f["scan_pauses_pct"] = float((t > LONG_SCAN_S).mean())
        f["items_per_dept"] = _safe_div(n_items, f["unique_departments"])

        # 7. temporal phases
        if n_items >= 3:
            acc = np.diff(t)
            f["scan_acceleration_mean"] = float(acc.mean())
            f["scan_acceleration_std"] = float(acc.std())
            f["time_delta_gini"] = gini(t)
            a, b = max(1, n_items // 3), max(2, 2 * n_items // 3)
            early_t, mid_t, late_t = t[:a].mean(), t[a:b].mean() if b > a else 0.0, t[b:].mean() if b < n_items else 0.0
            early_p, late_p = p[:a].mean(), p[b:].mean() if b < n_items else 0.0
            f["early_to_late_scan_ratio"] = _safe_div(early_t, late_t)
            f["middle_to_early_scan_ratio"] = _safe_div(mid_t, early_t)
            f["early_to_late_price_ratio"] = _safe_div(early_p, late_p)
            f["price_trend_coefficient"] = _safe_div(late_p - early_p, early_p)
            qs = [q for q in np.array_split(np.arange(n_items), 4) if q.size]
            if len(qs) == 4:
                qd = [np.unique(d[q]).size for q in qs]
                qp = np.array([p[q].mean() for q in qs])
                qt = np.array([t[q].mean() for q in qs])
                f["quarter_dept_diversity_ratio"] = _safe_div(max(qd), min(qd))
                f["quarter_price_volatility"] = _safe_div(qp.std(), qp.mean())
                f["quarter_time_volatility"] = _safe_div(qt.std(), qt.mean())
            else:
                f["quarter_dept_diversity_ratio"] = f["quarter_price_volatility"] = f["quarter_time_volatility"] = 0.0
        else:
            for k in ("scan_acceleration_mean", "scan_acceleration_std", "time_delta_gini",
                      "early_to_late_scan_ratio", "middle_to_early_scan_ratio",
                      "early_to_late_price_ratio", "price_trend_coefficient",
                      "quarter_dept_diversity_ratio", "quarter_price_volatility",
                      "quarter_time_volatility"):
                f[k] = 0.0

        # 8. price patterns
        if n_items >= 4:
            q1, q3 = p25, p75
            f["price_q3_q1_ratio"] = _safe_div(q3, q1)
            f["price_above_q3_pct"] = float((p > q3).mean())
            f["price_below_q1_pct"] = float((p < q1).mean())
        else:
            f["price_q3_q1_ratio"], f["price_above_q3_pct"], f["price_below_q1_pct"] = 1.0, 0.0, 0.0
        f["price_gini"] = gini(p)
        f["max_price_run_length"], f["avg_price_run_length"], f["price_run_consistency"] = price_runs(p)
        ranks = pd.Series(p).rank(pct=True).to_numpy()
        f["price_position_correlation"] = _corr(np.linspace(0, 1, n_items), ranks) if n_items > 1 else 0.0

        # 9. sequence behaviour
        if n_items >= 2:
            _, dept_counts = np.unique(d, return_counts=True)
            revisits = int((dept_counts - 1).sum())
            change = np.flatnonzero(np.diff(d) != 0)
            bounds = np.concatenate(([0], change + 1, [n_items]))
            dwell = np.diff(bounds)
            pdiff = np.diff(p)
            asc, desc = float((pdiff > 0).mean()), float((pdiff < 0).mean())
            f["dept_revisit_count"] = revisits
            f["dept_revisit_ratio"] = _safe_div(revisits, dept_counts.size)
            f["max_dept_dwell"] = int(dwell.max())
            f["avg_dept_dwell"] = float(dwell.mean())
            f["price_ascending_trend"] = asc
            f["price_descending_trend"] = desc
            f["price_ordering_bias"] = max(asc, desc)
            avg = self.dept_avg_price_[dc]
            rel = p[avg > 0] / avg[avg > 0]
            f["dept_relative_price_std"] = float(rel.std()) if rel.size else 0.0
            f["dept_relative_price_max"] = float(rel.max()) if rel.size else 0.0
        else:
            f.update(dept_revisit_count=0, dept_revisit_ratio=0.0, max_dept_dwell=1,
                     avg_dept_dwell=1.0, price_ascending_trend=0.0, price_descending_trend=0.0,
                     price_ordering_bias=0.0, dept_relative_price_std=0.0,
                     dept_relative_price_max=0.0)

        # 10. Markov transition likelihood (higher = less likely sequence)
        if n_items > 1:
            probs = self.transition_matrix_[dc[:-1], dc[1:]]
            f["sequence_nll"] = float(-np.log(np.maximum(probs, TRANSITION_FLOOR)).mean())
        else:
            f["sequence_nll"] = 0.0
        return {k: float(v) for k, v in f.items()}


class CorrelationFilter:
    """Drop one column of every pair with |Pearson r| > threshold, learned on training data."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame) -> "CorrelationFilter":
        num = X.select_dtypes(include=np.number)
        varying = num.loc[:, num.nunique() > 1]
        corr = varying.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop = {c for c in upper.columns if (upper[c] > self.threshold).any()}
        constant = set(num.columns) - set(varying.columns)
        self.columns_ = [c for c in num.columns if c not in drop and c not in constant]
        self.dropped_ = sorted(drop | constant)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "columns_"):
            raise RuntimeError("CorrelationFilter is not fitted")
        return X.reindex(columns=self.columns_, fill_value=0.0)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
