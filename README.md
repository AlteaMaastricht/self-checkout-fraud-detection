# Self-checkout fraud detection: unsupervised anomaly scoring of scan sequences

Ranks self-checkout visits by how unusual their scan sequence is (departments,
scan timing, prices), using feature engineering and an Isolation Forest +
K-Means ensemble. Because the original course data can't be shared, it is
demonstrated end-to-end on a synthetic visit generator with labelled injected
fraud.

## 1. Overview

This repository contains my **individual contribution** to a six-person
course project (Maastricht University, eLab Business Case II, 2025): detect
likely fraud among self-checkout visits of a fictional supermarket, **without
any fraud labels**. Shared work and teammates' work are identified in
[`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) and are not claimed.

The repository contains a consolidated, tested rewrite of **my own part** of
the team's code:
- visit-level feature engineering
- Isolation Forest / K-Means / autoencoder detectors and their ensemble
- the synthetic evaluation idea

Two bugs from the course version are fixed and documented:
- the "ensemble" silently used only its first detector
- thresholds were refitted on every scored file, so 15 % of visits were always flagged

The course dataset is **not** included. A synthetic generator with the same
schema makes everything runnable offline.

## 2. Problem

Each visit is a sequence of scanned items `(department, seconds since previous scan, price)`.
Fraud at self-checkout takes several forms:
- items not scanned
- cheaper codes entered
- long pauses to conceal goods

All of these show up only indirectly in such a log. There are no labels, so the task is
**unsupervised ranking**: rank the most suspicious visits first.

In the course, the team submitted 150 visits per 1,000-visit instructor case and got back
only a hit count.

## 3. Approach

1. **Parse** the visit log, including the `Transaction N:` prefix of evaluation files.
2. **Features** (141 per visit with 18 departments; 107–115 remain after the train-fitted correlation filter in the demo runs). They come in ten groups:
   - basket aggregates and scan-time and price quantiles
   - scan rhythm
   - per-department counts, spend and shares
   - high-value departments
   - entropy
   - ratios
   - early, middle and late phase ratios
   - price Gini and run lengths
   - revisits and dwell
   - price relative to the department average
   - Markov transition negative log-likelihood

   Everything population-dependent is **fitted on training visits only**: department average prices, the transition matrix, the high-value departments and the correlated-column filter.
3. **Detectors**, each standardised on training data. Each one outputs a score where higher means more anomalous.
   - Default ensemble (as in the original lineage): Isolation Forest and K-Means distance to the nearest centroid after PCA.
   - Also available: an MLP autoencoder, LOF (novelty mode), One-Class SVM, a Gaussian mixture, and a Keras autoencoder (optional, needs TensorFlow).
4. **Ensemble.** Each detector's scores are normalised with **training** min/max (or the training empirical CDF), then averaged over **all** detectors. The flag threshold is the training quantile `1 − contamination`. It is never refitted on the file being scored.
5. **Evaluation on synthetic data.**
   - Training set: 5,000 visits, of which 2 % carry unlabelled injected fraud.
   - Test set: a separate 1,000 visits with 10 % labelled injected fraud. There are three injected patterns:
     - `skipped_scan`: 1–3 items, biased towards expensive ones, are not scanned. Their handling time is added to the next gap.
     - `under_priced`: 1–3 items from expensive departments are recorded at 10–35 % of their price.
     - `scan_pause`: 1–2 pauses of 3–8 minutes.
   - Metrics: ROC AUC, average precision, precision and recall at 150 (the course's submission size), AUC per fraud type, and precision and recall at the train-fitted threshold.

## 4. Architecture

```
src/checkout_anomaly/
  parsing.py     visit log reader/writer (Transaction-prefix fix)
  synthetic.py   VisitGenerator: Markov department walk, log-normal gaps/prices, injected fraud
  features.py    FeatureExtractor (fit on train) + CorrelationFilter
  models.py      IF, K-Means distance, MLP-AE, LOF, OCSVM, GMM, Keras-AE (optional)
  ensemble.py    AnomalyEnsemble (train-fitted normalisation/threshold), CheckoutAnomalyPipeline
  evaluation.py  synthetic injected-fraud protocol, ranking metrics, original perturbation check
  cli.py         generate | score | evaluate
examples/run_synthetic_demo.py   writes results/synthetic_demo_*.{json,md}
tests/                           unittest suite (35 tests)
docs/ARCHITECTURE.md             data flow diagram and design notes
docs/PROVENANCE.md               original function → staged module, every change and why
```

## 5. Technologies

Python ≥ 3.10 (tested on 3.14), NumPy, pandas, scikit-learn (IsolationForest,
KMeans, PCA, MLPRegressor, LocalOutlierFactor, OneClassSVM, GaussianMixture).
TensorFlow is optional; it is used only by `KerasAutoencoderDetector`, which was not tested.

## 6. Installation & running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e .
python3 -m unittest discover -s tests -t .   # ~25-65 s
PYTHONPATH=src python3 examples/run_synthetic_demo.py   # full demo, see results/ for wall time
```

The tests and demo were run with the system Python (numpy 2.4.4, pandas 3.1.0,
scikit-learn 1.8.0). The `pip install` steps above were not executed while preparing this repository.

## 7. Example usage

```bash
export PYTHONPATH=src
# same store (world seed 7), different visit seeds for train and test
python3 -m checkout_anomaly.cli generate --n 5000 --fraud-rate 0.02 --seed 0 --world-seed 7 --out data/train.csv
python3 -m checkout_anomaly.cli generate --n 1000 --fraud-rate 0.10 --seed 1 --world-seed 7 \
    --id-offset 5000 --with-id --labels data/test_labels.csv --out data/test.csv
python3 -m checkout_anomaly.cli score --train data/train.csv --test data/test.csv --top 150 --out top150.csv
python3 -m checkout_anomaly.cli evaluate --seeds 0 1 2 --n-train 5000 --out results/my_run.json
```

```python
from checkout_anomaly import CheckoutAnomalyPipeline, VisitGenerator

gen = VisitGenerator(seed=0)
train, test = gen.generate(5000, fraud_rate=0.02), gen.generate(1000, fraud_rate=0.10, id_offset=5000)
pipe = CheckoutAnomalyPipeline(detectors=["isolation_forest", "kmeans_distance"]).fit(train.visits)
scores = pipe.score(test.visits)      # per-detector normalised scores, ensemble_score, flagged
top150 = pipe.top_k(test.visits, 150)
```

## 8. Results

**Synthetic data only.** These numbers measure how well the detectors rank the
generator's own injected patterns. They are **not** an estimate of real-world
fraud detection.

Source: `results/synthetic_demo_summary.md` and `results/synthetic_demo_metrics.json`,
produced by `PYTHONPATH=src python3 examples/run_synthetic_demo.py`.

Protocol:
- 5,000 training visits, 2 % of them unlabelled fraud.
- 1,000 test visits, 10 % labelled fraud, so 100 positives.
- Threshold fitted on training scores.
- Default ensemble: seeds 0–4. All six scikit-learn detectors: seeds 0–2.

Default IF + K-Means ensemble (mean ± sd over 5 seeds):

| Metric | Isolation Forest | K-Means distance | Ensemble |
|---|---|---|---|
| ROC AUC | 0.653 ± 0.057 | 0.651 ± 0.054 | 0.656 ± 0.056 |
| Average precision (prevalence 0.10) | 0.252 | 0.272 | 0.278 |
| Precision@150 | 0.252 | 0.231 | 0.237 |
| Recall@150 | 0.378 | 0.346 | 0.356 |
| AUC, `scan_pause` | 0.925 | 0.898 | 0.927 |
| AUC, `skipped_scan` | 0.568 | 0.566 | 0.570 |
| AUC, `under_priced` | 0.487 | 0.505 | 0.490 |

The ensemble at the train-fitted threshold flags 16.2 % ± 1.0 % of test visits. Its precision is 0.230 and its recall is 0.376.

With all six scikit-learn detectors (3 seeds):
- Single-detector ROC AUC ranges from 0.642 to 0.676. The MLP autoencoder and the GMM are slightly best.
- The six-detector ensemble reaches 0.657.
- None of the detectors exceeds an AUC of 0.59 on `skipped_scan` or 0.54 on `under_priced`.

What this shows:
- Long scan pauses are easy to rank.
- Removing 1–3 items, or under-pricing 1–3 items, barely moves visit-level aggregates, so the
  detectors are close to random on those patterns. `under_priced` is even slightly below 0.5
  for some detectors.
- Precision@150 of about 0.24 at 10 % prevalence is a ~2.4× lift over random. That is in the same
  range as the 12–25 % blind-test hit rates the team reported on the real case (see Limitations),
  but the two numbers are **not comparable**.
- Combining detectors gives no meaningful gain over the best single detector.
- No parameters were tuned after seeing these results.

## 9. Limitations

### Fraud detection limitations

- **No ground truth.** The real data had no labels. The only external feedback in the course was a hit
  count for 150 submitted visits per instructor case file.
  - The team's report draft records hit rates of roughly **12–25 % of the 150 submitted visits**
    across the models tried (18/150 to 37/150).
  - These came from teammates' and team models, not from this code.
  - They are **reported in team documents and not reproducible here**: the case files and the
    instructor's answer key are not available.
- **Synthetic evaluation ≠ real performance.**
  - The generator's parameters are hand-chosen, not fitted to real data.
  - The detectors are evaluated on patterns that I designed myself.
  - Good scores on `scan_pause` show that long gaps are easy to rank, not that real concealment can be detected.
  - Poor scores on `skipped_scan` and `under_priced` are a real finding for this feature set and
    these detectors: subtle per-item fraud barely changes visit-level aggregates.
- **Anomalous ≠ fraudulent.** Large families, slow shoppers, people with disabilities or children,
  and unusual but legitimate baskets are all "anomalies".
  - Every flag has a cost: staff time, and the delay and embarrassment of an honest customer being checked.
  - A threshold must therefore be set from the cost of false positives, not from a fixed 15 %.
- **Privacy and ethics.** Scoring individual shoppers' behaviour is monitoring.
  - A real deployment would need:
    - a lawful basis and data-protection impact assessment (GDPR)
    - data minimisation and retention limits
    - transparency to customers
    - human review before any action
    - checks that flags do not fall disproportionately on particular groups
  - The model must not be used to accuse anyone automatically.

### Technical limitations

- The LSTM sequence embeddings from the course code were **not ported**. They need PyTorch, and the
  original retrained the encoder on each scored file, which made its features inconsistent (see
  `docs/PROVENANCE.md`).
- The rule layer of the original autoencoder was removed, because it implemented only 3 of its 25 rules.
- `KerasAutoencoderDetector` is implemented but **never executed** (TensorFlow is not installed).
- Feature extraction runs per visit in Python/NumPy, so it is slow for 100k visits (minutes). It is not vectorised.
- There is no per-visit explanation of flagged visits yet.
- The fraud-type mix and prevalence in the test set are assumptions.

### Future work

- vectorised features
- per-feature attribution for flagged visits
- PR curves and an ablation per feature group
- segmented models (by basket size or price level)
- cost-based threshold selection

## 10. Reproducibility

- **Data:** synthetic only. `VisitGenerator(seed, world_seed)` is deterministic; `results/` records
  the command, the seeds and the package versions.
- **Can be regenerated:** every number in `results/` and in the tests, by running the commands in §6.
- **Cannot be regenerated:** anything about the original course data, including the blind-test
  hit counts above and the original plots and pickled models. None of it is distributed.
- **Input format** for real data you are entitled to use: one visit per line, `dept seconds price`
  tuples separated by commas, with an optional `Transaction <id>:` prefix.

## 11. Context, attribution & licence

- **Course:** eLab Business Case II, Maastricht University (BSc Business Analytics, year 2), spring 2025.
  - The case and the supermarket dataset were provided for the course. The dataset is not
    redistributable and is **not included**: no real visit, case file, derived feature table or
    trained model is in this repository.
- **Team and contributions:** a project of six students; teammates are not named. Individual,
  shared and teammates' work are listed in [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md).
  - The team explored many approaches, including segmented Isolation Forests, One-Class SVM, DBSCAN,
    LSTM models, SHAP explanations and EDA. Those are **teammates' work and are not included**.
  - This repository contains only a rewrite of my own lineage of scripts: features, IF/K-Means/autoencoder
    ensemble, and synthetic evaluation. `docs/PROVENANCE.md` maps every function to its original.
- **AI assistance:**
  - **Original 2025 team code (author's lineage):** whether and how AI was used is unspecified.
  - **2026 rewrite:** the consolidation, bug fixes, regression tests, synthetic generator and
    documentation were done with Claude Code under my direction. I reviewed the changes.
- **Data privacy:**
  - The repository contains no personal data.
  - The synthetic generator does not sample from, and is not calibrated on, the course data.
  - If you apply this code to real shopper data, you are responsible for complying with data-protection law.
- **License:** none chosen yet. The code includes shared team work; see [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md).
