"""Synthetic self-checkout visit generator with labelled, injected fraud.

The course dataset cannot be redistributed, so this module produces visits in
the same schema (department code, seconds since previous scan, price). All
parameters below were chosen by hand to be plausible; they are **not** fitted
to the course data, and results on this generator say nothing about
performance on real shoppers.

Normal behaviour
----------------
* Departments ``1..n_departments`` are laid out along a walking route; a
  Markov walk either stays in the current department (``p_stay``) or moves to
  another department with probability proportional to its popularity times
  ``exp(-distance / layout_scale)``.
* Basket size is ``2 + NegativeBinomial`` (mean ``mean_items``).
* The gap between scans is log-normal; moving to another department adds a
  log-normal walking time that grows with the distance travelled.
* Prices are log-normal with a department-specific median.

Injected fraud patterns (configurable)
--------------------------------------
* ``skipped_scan`` - 1-3 items (biased towards expensive ones) are taken
  without scanning: they disappear from the log and their handling time is
  added to the next scan gap.
* ``under_priced`` - 1-3 items from the most expensive departments are
  recorded at 10-35 % of their price (e.g. a cheaper code entered at a
  weighing scale, or a swapped barcode within the department).
* ``scan_pause`` - 1-2 unusually long pauses (3-8 minutes) are inserted, as
  when goods are concealed between scans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .parsing import Visit

FRAUD_TYPES = ("skipped_scan", "under_priced", "scan_pause")


@dataclass
class GeneratorConfig:
    n_departments: int = 18
    mean_items: float = 18.0
    basket_dispersion: float = 3.0  # negative-binomial "n"; smaller = more spread
    p_stay: float = 0.45
    layout_scale: float = 3.0
    scan_gap_median: float = 12.0  # seconds between scans within a department
    scan_gap_sigma: float = 0.6
    walk_median_per_step: float = 8.0  # extra seconds per department step walked
    walk_sigma: float = 0.5
    price_median_range: tuple[float, float] = (0.8, 7.5)
    price_sigma: float = 0.45
    fraud_mix: dict[str, float] = field(
        default_factory=lambda: {t: 1.0 / len(FRAUD_TYPES) for t in FRAUD_TYPES}
    )
    high_value_share: float = 0.2  # share of departments treated as "expensive" for under_priced


@dataclass
class SyntheticDataset:
    visits: list[Visit]
    is_fraud: np.ndarray  # bool, one per visit
    fraud_type: list[str]  # "normal" or one of FRAUD_TYPES


class VisitGenerator:
    """Seeded generator; the store "world" (layout, prices) is fixed per seed."""

    def __init__(self, config: GeneratorConfig | None = None, seed: int = 0, world_seed: int | None = None):
        """``world_seed`` fixes the store (layout, prices; defaults to ``seed``);
        ``seed`` drives the visits. Use one world_seed and different seeds to
        generate train and test files of the same store in separate calls."""
        self.config = config or GeneratorConfig()
        self.seed = seed
        self.world_seed = seed if world_seed is None else world_seed
        world_rng = np.random.default_rng(self.world_seed)
        c = self.config
        n = c.n_departments
        self.popularity = world_rng.dirichlet(np.full(n, 2.0))
        lo, hi = c.price_median_range
        self.price_median = np.exp(world_rng.uniform(np.log(lo), np.log(hi), size=n))
        # per-department scan-gap multiplier (e.g. weighing produce takes longer)
        self.gap_factor = world_rng.uniform(0.7, 1.6, size=n)
        k = max(1, int(round(c.high_value_share * n)))
        self.high_value_depts = np.argsort(self.price_median)[-k:] + 1  # 1-based codes
        idx = np.arange(n)
        dist = np.abs(idx[:, None] - idx[None, :])
        move = self.popularity[None, :] * np.exp(-dist / c.layout_scale)
        np.fill_diagonal(move, 0.0)
        move /= move.sum(axis=1, keepdims=True)
        self.transition = c.p_stay * np.eye(n) + (1 - c.p_stay) * move
        # cumulative tables for fast inverse-CDF sampling
        self._pop_cdf = np.cumsum(self.popularity)
        self._trans_cdf = np.cumsum(self.transition, axis=1)
        self._rng = np.random.default_rng(seed + 1_000_003)

    # ------------------------------------------------------------------ normal
    def _normal_items(self, rng: np.random.Generator) -> list[tuple[int, float, float]]:
        c = self.config
        n = c.n_departments
        p = c.basket_dispersion / (c.basket_dispersion + max(c.mean_items - 2, 0.1))
        n_items = 2 + int(rng.negative_binomial(c.basket_dispersion, p))
        u = rng.random(n_items)
        z_gap, z_walk, z_price = rng.standard_normal((3, n_items))
        dept = min(int(np.searchsorted(self._pop_cdf, u[0])), n - 1)
        items = []
        for i in range(n_items):
            if i > 0:
                new_dept = min(int(np.searchsorted(self._trans_cdf[dept], u[i])), n - 1)
            else:
                new_dept = dept
            gap = np.exp(np.log(c.scan_gap_median * self.gap_factor[new_dept]) + c.scan_gap_sigma * z_gap[i])
            if i == 0 or new_dept != dept:
                steps = 1 if i == 0 else abs(new_dept - dept)
                gap += np.exp(np.log(c.walk_median_per_step * steps) + c.walk_sigma * z_walk[i])
            dept = new_dept
            price = np.exp(np.log(self.price_median[dept]) + c.price_sigma * z_price[i])
            items.append((dept + 1, float(round(gap)), float(round(max(price, 0.05), 2))))
        return items

    # ------------------------------------------------------------------ fraud
    def _inject(self, items, fraud: str, rng: np.random.Generator):
        items = list(items)
        if fraud == "skipped_scan":
            # baskets always have >= 2 items, so at least one item can be skipped
            k = int(rng.integers(1, 4))
            k = min(k, len(items) - 1)
            prices = np.array([p for _, _, p in items[:-1]])  # never skip the last scan
            drop = set(rng.choice(len(prices), size=k, replace=False, p=prices / prices.sum()).tolist())
            out, carry = [], 0.0
            for i, (d, t, p) in enumerate(items):
                if i in drop:
                    carry += t + rng.lognormal(np.log(10.0), 0.5)  # handling time of the unscanned item
                    continue
                out.append((d, float(round(t + carry)), p))
                carry = 0.0
            return out
        if fraud == "under_priced":
            hv = set(int(x) for x in self.high_value_depts)
            cand = [i for i, (d, _, _) in enumerate(items) if d in hv]
            if not cand:  # replace a random item's department with an expensive one first
                i = int(rng.integers(len(items)))
                d_new = int(rng.choice(sorted(hv)))
                price = rng.lognormal(np.log(self.price_median[d_new - 1]), self.config.price_sigma)
                items[i] = (d_new, items[i][1], float(round(price, 2)))
                cand = [i]
            k = min(int(rng.integers(1, 4)), len(cand))
            for i in rng.choice(cand, size=k, replace=False):
                d, t, p = items[int(i)]
                items[int(i)] = (d, t, float(round(max(p * rng.uniform(0.10, 0.35), 0.05), 2)))
            return items
        if fraud == "scan_pause":
            k = min(int(rng.integers(1, 3)), len(items))
            for i in rng.choice(len(items), size=k, replace=False):
                d, t, p = items[int(i)]
                items[int(i)] = (d, float(round(t + rng.uniform(180, 480))), p)
            return items
        raise ValueError(f"unknown fraud type {fraud!r}")

    # ------------------------------------------------------------------ api
    def generate(self, n_visits: int, fraud_rate: float = 0.0, id_offset: int = 0) -> SyntheticDataset:
        """Generate ``n_visits`` visits; a ``fraud_rate`` share receives one injected pattern."""
        if not 0.0 <= fraud_rate <= 1.0:
            raise ValueError("fraud_rate must be in [0, 1]")
        rng = self._rng
        mix = self.config.fraud_mix
        types = [t for t in mix if mix[t] > 0]
        probs = np.array([mix[t] for t in types], dtype=float)
        probs /= probs.sum() if probs.sum() > 0 else 1.0
        n_fraud = int(round(fraud_rate * n_visits))
        fraud_idx = set(rng.choice(n_visits, size=n_fraud, replace=False).tolist()) if n_fraud else set()
        visits, labels, kinds = [], np.zeros(n_visits, dtype=bool), []
        for i in range(n_visits):
            items = self._normal_items(rng)
            kind = "normal"
            if i in fraud_idx and types:
                kind = str(rng.choice(types, p=probs))
                items = self._inject(items, kind, rng)
                labels[i] = True
            visits.append(Visit(visit_id=str(id_offset + i), items=items))
            kinds.append(kind)
        return SyntheticDataset(visits=visits, is_fraud=labels, fraud_type=kinds)
