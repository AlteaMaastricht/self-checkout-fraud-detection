"""Command-line interface.

    python -m checkout_anomaly.cli generate --n 10000 --fraud-rate 0.02 --seed 0 --world-seed 7 --out data/train.csv
    python -m checkout_anomaly.cli generate --n 1000 --fraud-rate 0.10 --seed 1 --world-seed 7 --id-offset 10000 \
        --with-id --labels data/test_labels.csv --out data/test.csv
    python -m checkout_anomaly.cli score --train data/train.csv --test data/test.csv --top 150 --out top150.csv
    python -m checkout_anomaly.cli evaluate --seeds 0 1 2 --out results/run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ensemble import CheckoutAnomalyPipeline
from .evaluation import evaluate_synthetic, summarise
from .parsing import read_visits, write_visits
from .synthetic import VisitGenerator


def _cmd_generate(a: argparse.Namespace) -> int:
    ds = VisitGenerator(seed=a.seed, world_seed=a.world_seed).generate(a.n, fraud_rate=a.fraud_rate, id_offset=a.id_offset)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_visits(ds.visits, a.out, with_id=a.with_id)
    if a.labels:
        with open(a.labels, "w", encoding="utf-8") as fh:
            fh.write("visit_id,is_fraud,fraud_type\n")
            for v, y, t in zip(ds.visits, ds.is_fraud, ds.fraud_type):
                fh.write(f"{v.visit_id},{int(y)},{t}\n")
    print(f"wrote {len(ds.visits)} synthetic visits ({int(ds.is_fraud.sum())} injected fraud) to {a.out}")
    return 0


def _cmd_score(a: argparse.Namespace) -> int:
    pipe = CheckoutAnomalyPipeline(detectors=a.detectors, contamination=a.contamination)
    pipe.fit(read_visits(a.train))
    top = pipe.top_k(read_visits(a.test), k=a.top)
    if a.out:
        top.to_csv(a.out)
        print(f"wrote top {len(top)} visits to {a.out}")
    else:
        top.to_csv(sys.stdout)
    return 0


def _cmd_evaluate(a: argparse.Namespace) -> int:
    runs = [evaluate_synthetic(seed=s, n_train=a.n_train, n_test=a.n_test,
                               train_fraud_rate=a.train_fraud_rate, test_fraud_rate=a.test_fraud_rate,
                               detectors=a.detectors, contamination=a.contamination, k=a.k)
            for s in a.seeds]
    print(summarise(runs).round(3).to_string())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"data": "SYNTHETIC (checkout_anomaly.synthetic)", "runs": runs}, fh, indent=2)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="checkout_anomaly")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write synthetic visits in the case log format")
    g.add_argument("--n", type=int, default=10_000)
    g.add_argument("--fraud-rate", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0, help="seed for the visits")
    g.add_argument("--world-seed", type=int, default=None,
                   help="seed for the store layout/prices (default: --seed); keep it fixed for train and test")
    g.add_argument("--id-offset", type=int, default=0)
    g.add_argument("--with-id", action="store_true", help="prefix lines with 'Transaction <id>:'")
    g.add_argument("--labels", help="optional CSV with the injected labels")
    g.add_argument("--out", required=True)
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("score", help="fit on a training log, rank visits of a test log")
    s.add_argument("--train", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--top", type=int, default=150)
    s.add_argument("--detectors", nargs="+", default=["isolation_forest", "kmeans_distance"])
    s.add_argument("--contamination", type=float, default=0.15)
    s.add_argument("--out")
    s.set_defaults(func=_cmd_score)

    e = sub.add_parser("evaluate", help="synthetic injected-fraud evaluation")
    e.add_argument("--seeds", type=int, nargs="+", default=[0])
    e.add_argument("--n-train", type=int, default=10_000)
    e.add_argument("--n-test", type=int, default=1_000)
    e.add_argument("--train-fraud-rate", type=float, default=0.02)
    e.add_argument("--test-fraud-rate", type=float, default=0.10)
    e.add_argument("--detectors", nargs="+", default=["isolation_forest", "kmeans_distance"])
    e.add_argument("--contamination", type=float, default=0.15)
    e.add_argument("--k", type=int, default=150)
    e.add_argument("--out")
    e.set_defaults(func=_cmd_evaluate)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
