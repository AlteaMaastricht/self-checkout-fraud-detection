"""Evaluation on synthetic data with injected (hence labelled) fraud.

The course data had **no labels**; the team could only submit 150 suspected
visits per instructor case file and receive a hit count. This module replaces
that with a fully synthetic, labelled protocol:

1. generate an *unlabelled* training set (it may contain a small share of
   fraud, as real history would),
2. fit features, normalisation and threshold on the training set only,
3. score a separate test set with a known fraud share, and
4. report ranking metrics (ROC AUC, average precision, precision/recall@k),
   metrics at the train-fitted threshold, and AUC per fraud type.

Results measure how well the detectors find **the patterns this generator
injects**. They are not an estimate of performance on real shoppers.

:func:`feature_perturbation_check` keeps the original robustness check of the
course code (multiplying random feature values of 5 % of rows), with a seed and
without leaking model scores into the features; it is a weak test because the
perturbed rows are extreme by construction.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from .ensemble import CheckoutAnomalyPipeline
from .synthetic import FRAUD_TYPES, GeneratorConfig, VisitGenerator


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=bool)
    k = int(min(k, y.size))
    if k <= 0:
        return 0.0, 0.0
    top = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:k]
    hits = int(y[top].sum())
    return hits / k, hits / max(int(y.sum()), 1)


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray, ks: Sequence[int] = (150,)) -> dict:
    y = np.asarray(y_true, dtype=bool)
    out = {"roc_auc": float(roc_auc_score(y, scores)),
           "average_precision": float(average_precision_score(y, scores)),
           "prevalence": float(y.mean())}
    for k in ks:
        p, r = precision_recall_at_k(y, scores, k)
        out[f"precision_at_{k}"] = p
        out[f"recall_at_{k}"] = r
    return out


def evaluate_synthetic(seed: int = 0, n_train: int = 10_000, n_test: int = 1_000,
                       train_fraud_rate: float = 0.02, test_fraud_rate: float = 0.10,
                       detectors: Sequence[str] = ("isolation_forest", "kmeans_distance"),
                       contamination: float = 0.15, k: int = 150,
                       config: GeneratorConfig | None = None) -> dict:
    """Run one synthetic train/test experiment and return a JSON-serialisable dict."""
    t0 = time.perf_counter()
    gen = VisitGenerator(config, seed=seed)
    train = gen.generate(n_train, fraud_rate=train_fraud_rate, id_offset=0)
    test = gen.generate(n_test, fraud_rate=test_fraud_rate, id_offset=n_train)

    pipe = CheckoutAnomalyPipeline(detectors=detectors, contamination=contamination, random_state=seed)
    pipe.fit(train.visits)
    scored = pipe.score(test.visits)
    y = test.is_fraud
    types = np.array(test.fraud_type)

    result: dict = {
        "seed": seed, "n_train": n_train, "n_test": n_test,
        "train_fraud_rate": train_fraud_rate, "test_fraud_rate": test_fraud_rate,
        "contamination": contamination, "detectors": list(detectors),
        "n_features_used": len(pipe.filter.columns_) if pipe.filter is not None else None,
        "threshold_fitted_on": "train",
        "methods": {},
    }
    columns = list(pipe.ensemble.detectors_) + ["ensemble_score"]
    for col in columns:
        s = scored[col].to_numpy()
        m = ranking_metrics(y, s, ks=(k, int(y.sum())))
        m["auc_by_fraud_type"] = {
            t: float(roc_auc_score(np.r_[np.zeros((types == "normal").sum()), np.ones((types == t).sum())],
                                   np.r_[s[types == "normal"], s[types == t]]))
            for t in FRAUD_TYPES if (types == t).any()
        }
        result["methods"]["ensemble" if col == "ensemble_score" else col] = m
    flags = scored["flagged"].to_numpy()
    tp = int((flags & y).sum())
    result["at_train_threshold"] = {
        "flag_rate": float(flags.mean()),
        "precision": tp / max(int(flags.sum()), 1),
        "recall": tp / max(int(y.sum()), 1),
    }
    result["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    return result


def summarise(runs: Sequence[dict]) -> pd.DataFrame:
    """Mean and standard deviation over seeds for the main metrics of every method."""
    rows = []
    for r in runs:
        for method, m in r["methods"].items():
            row = {"seed": r["seed"], "method": method}
            row.update({k: v for k, v in m.items() if not isinstance(v, dict)})
            row.update({f"auc_{t}": v for t, v in m["auc_by_fraud_type"].items()})
            rows.append(row)
    df = pd.DataFrame(rows).drop(columns=["seed"])
    return df.groupby("method", sort=False).agg(["mean", "std"])


def feature_perturbation_check(X: pd.DataFrame, seed: int = 0, frac: float = 0.05,
                               threshold_quantile: float = 0.86) -> dict:
    """Port of the course code's ``evaluate_model_robustness`` (seeded, no score leakage).

    5 % of rows are perturbed in three equal groups: one random feature x10;
    two features x5 and x0.2; Gaussian noise (sd 3) on all features. The Isolation
    Forest is fitted on the unperturbed matrix; the threshold is the 86th
    percentile of the unperturbed scores, as in the original.
    """
    rng = np.random.default_rng(seed)
    Xn = X.select_dtypes(include=np.number).to_numpy(dtype=float)
    iso = IsolationForest(random_state=seed).fit(Xn)
    base = -iso.decision_function(Xn)
    n_syn = int(Xn.shape[0] * frac)
    idx = rng.choice(Xn.shape[0], n_syn, replace=False)
    Xs = Xn.copy()
    a, b = n_syn // 3, 2 * n_syn // 3
    for i in idx[:a]:
        c = rng.integers(Xn.shape[1])
        Xs[i, c] *= 10
    for i in idx[a:b]:
        c = rng.choice(Xn.shape[1], 2, replace=False)
        Xs[i, c[0]] *= 5
        Xs[i, c[1]] *= 0.2
    for i in idx[b:]:
        Xs[i] += rng.normal(0, 3, Xn.shape[1])
    s = -iso.decision_function(Xs)
    y = np.zeros(Xn.shape[0], dtype=bool)
    y[idx] = True
    pred = s > np.quantile(base, threshold_quantile)
    tp = int((pred & y).sum())
    return {"precision": tp / max(int(pred.sum()), 1), "recall": tp / max(int(y.sum()), 1),
            "average_precision": float(average_precision_score(y, s))}
