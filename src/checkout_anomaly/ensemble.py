"""Score-averaging ensemble and the end-to-end pipeline.

Two defects of the original course code are fixed here (details in
``docs/PROVENANCE.md``):

1. **Ensemble returned after the first method.** In the original
   ``ensemble_anomaly_detection`` the block that averages the normalised scores
   was indented inside the ``for`` loop that normalises them, so the function
   returned during the first iteration and the "ensemble" score was just the
   first detector's normalised score. :func:`combine_normalised_scores` averages
   over *all* methods; ``tests/test_ensemble.py`` contains a regression test.

2. **Thresholds and score normalisation were refitted on every scored file.**
   The originals min-max-scaled the scores of the test file and flagged its top
   15 % (85th percentile of the test file itself), so every file had exactly
   15 % flagged regardless of its content. Here the per-method normalisation
   constants and the decision threshold are learned from the **training**
   scores in :meth:`AnomalyEnsemble.fit` and only applied afterwards.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import CorrelationFilter, FeatureExtractor
from .models import BaseDetector, make_detector
from .parsing import Visit

DEFAULT_DETECTORS: tuple[str, ...] = ("isolation_forest", "kmeans_distance")


def combine_normalised_scores(scores: Mapping[str, np.ndarray],
                              weights: Mapping[str, float] | None = None) -> np.ndarray:
    """(Weighted) mean of already normalised per-method scores over **all** methods."""
    if not scores:
        raise ValueError("no scores to combine")
    names = list(scores)
    stacked = np.vstack([np.asarray(scores[n], dtype=float) for n in names])
    w = np.array([1.0 if weights is None else float(weights.get(n, 0.0)) for n in names])
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive value")
    return (w[:, None] * stacked).sum(axis=0) / w.sum()


class AnomalyEnsemble:
    """Fit several detectors on training features and average their normalised scores.

    ``normalisation``:
      * ``"minmax"`` - (s - min_train) / (max_train - min_train); test scores may
        fall outside [0, 1], which is intended (they are not refitted).
      * ``"rank"``   - empirical CDF of the training scores (robust to outliers).
    """

    def __init__(self, detectors: Sequence[str | BaseDetector] = DEFAULT_DETECTORS,
                 contamination: float = 0.15, normalisation: str = "minmax",
                 weights: Mapping[str, float] | None = None, random_state: int = 42):
        if not 0 < contamination < 1:
            raise ValueError("contamination must be in (0, 1)")
        if normalisation not in ("minmax", "rank"):
            raise ValueError("normalisation must be 'minmax' or 'rank'")
        self.detectors = detectors
        self.contamination = contamination
        self.normalisation = normalisation
        self.weights = weights
        self.random_state = random_state

    def _build(self) -> dict[str, BaseDetector]:
        built: dict[str, BaseDetector] = {}
        for d in self.detectors:
            if isinstance(d, str):
                kwargs = {"random_state": self.random_state} if d not in ("lof", "one_class_svm") else {}
                det = make_detector(d, **kwargs)
            else:
                det = d
            built[getattr(det, "name", type(det).__name__)] = det
        return built

    def fit(self, X: pd.DataFrame) -> "AnomalyEnsemble":
        self.detectors_ = self._build()
        self.train_scores_: dict[str, np.ndarray] = {}
        for name, det in self.detectors_.items():
            det.fit(X)
            self.train_scores_[name] = det.score(X)
        self.norm_ = {n: (float(s.min()), float(s.max())) for n, s in self.train_scores_.items()}
        self.sorted_train_ = {n: np.sort(s) for n, s in self.train_scores_.items()}
        train_ens = combine_normalised_scores(self._normalise(self.train_scores_), self.weights)
        self.threshold_ = float(np.quantile(train_ens, 1 - self.contamination))
        return self

    def _normalise(self, raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for name, s in raw.items():
            if self.normalisation == "minmax":
                lo, hi = self.norm_[name]
                out[name] = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
            else:
                ref = self.sorted_train_[name]
                out[name] = np.searchsorted(ref, s, side="right") / ref.size
        return out

    def score_components(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalised score of every detector (columns) for every visit (rows)."""
        if not hasattr(self, "threshold_"):
            raise RuntimeError("AnomalyEnsemble is not fitted")
        raw = {n: d.score(X) for n, d in self.detectors_.items()}
        idx = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(self._normalise(raw), index=idx)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        comps = self.score_components(X)
        return combine_normalised_scores({c: comps[c].to_numpy() for c in comps.columns}, self.weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean flag per visit, using the threshold learned on training data."""
        return self.score(X) > self.threshold_


class CheckoutAnomalyPipeline:
    """visits -> features (fit on train) -> correlation filter (fit on train) -> ensemble."""

    def __init__(self, detectors: Sequence[str | BaseDetector] = DEFAULT_DETECTORS,
                 contamination: float = 0.15, normalisation: str = "minmax",
                 n_departments: int = 18, drop_correlated: float | None = 0.95,
                 random_state: int = 42):
        self.extractor = FeatureExtractor(n_departments=n_departments)
        self.filter = CorrelationFilter(drop_correlated) if drop_correlated else None
        self.ensemble = AnomalyEnsemble(detectors, contamination=contamination,
                                        normalisation=normalisation, random_state=random_state)

    def _features(self, visits: Iterable[Visit], fit: bool) -> pd.DataFrame:
        visits = list(visits)
        if fit:
            self.extractor.fit(visits)
        X = self.extractor.transform(visits)
        if self.filter is not None:
            X = self.filter.fit_transform(X) if fit else self.filter.transform(X)
        return X

    def fit(self, visits: Iterable[Visit]) -> "CheckoutAnomalyPipeline":
        self.ensemble.fit(self._features(visits, fit=True))
        return self

    def features(self, visits: Iterable[Visit]) -> pd.DataFrame:
        return self._features(visits, fit=False)

    def score(self, visits: Iterable[Visit]) -> pd.DataFrame:
        """Per-visit normalised component scores, ensemble score and flag."""
        X = self._features(visits, fit=False)
        out = self.ensemble.score_components(X)
        out["ensemble_score"] = self.ensemble.score(X)
        out["flagged"] = out["ensemble_score"] > self.ensemble.threshold_
        return out

    def top_k(self, visits: Iterable[Visit], k: int = 150) -> pd.DataFrame:
        """The ``k`` highest-scoring visits (the course's blind-test submission format)."""
        return self.score(visits).sort_values("ensemble_score", ascending=False).head(k)
