# Architecture

```
visit log (dept seconds price, ...)          synthetic.VisitGenerator
        │  parsing.read_visits                    │ (labelled injected fraud)
        ▼                                         ▼
   list[Visit] ──────────────────────────────────────────────
        │
        ▼  fit on TRAIN only                     transform TRAIN / TEST
 features.FeatureExtractor  ── dept avg prices, transition matrix, high-value depts
        │
        ▼
 features.CorrelationFilter ── kept columns (|r| ≤ 0.95, non-constant on TRAIN)
        │
        ▼
 models.*Detector (each: StandardScaler → model, fitted on TRAIN)
   isolation_forest │ kmeans_distance │ mlp_autoencoder │ lof │ one_class_svm │ gmm │ keras_autoencoder*
        │ raw scores (higher = more anomalous)
        ▼
 ensemble.AnomalyEnsemble
   normalise with TRAIN min/max (or TRAIN empirical CDF)
   mean over ALL detectors          (fixes the "returns after first method" bug)
   threshold = TRAIN quantile 1 - contamination   (never refitted on scored data)
        │
        ▼
 per-visit component scores, ensemble_score, flagged ; top_k(150)
        │
        ▼
 evaluation.evaluate_synthetic → ROC AUC, AP, precision/recall@k, AUC per fraud type
```

`*` optional; needs TensorFlow; not executed in this repository's tests.

Design notes:

- Every population-dependent quantity is learned on training data. The
  original course scripts recomputed thresholds, scalers, department averages
  and the transition matrix on each file they scored. That made the share of
  flagged visits a constant (15 %), whatever the data contained.
- Detectors share a two-method interface (`fit(X)`, `score(X)`), so the
  ensemble can take any subset or a custom object.
- Feature extraction is per visit in NumPy (about 12 ms/visit on the
  development machine, measured during development), so 100k visits would take
  roughly 20 minutes in one process. A vectorised pandas implementation would
  be the next step for real-scale data.
