# Provenance

This repository is a **consolidated rewrite**, not a copy. No original source
file is included verbatim. The originals were roughly 50 Google Colab exports
(many near-duplicates) written during a six-person course case (Maastricht
University, eLab Business Case II, 2025). They depend on a course dataset that
cannot be redistributed. This file maps every staged module to the original
functions it was ported from, and lists every behavioural change.

Line numbers refer to the original course folders, which are not published:
`Case 1/` and `Interview 2/`.

## Authorship scope

Only the author's own lineage was used:
`fraud_detection (1).py` → `fraud beta.py` → `test1.py` / `Interview 2/test1 (1).py` → `Interview 2/thursday (1).py`.
The following also count as the author's lineage:
- `333 clean.py`, which holds the synthetic evaluation.
- `autoecoder.py`, the TPU/Keras autoencoder. Its attribution is medium-confidence.
- the notes `Interview 2/Features and models.rtf`.

Attribution is based on the team's internal file-sharing record. There is no git history.

Teammates' code was **not** used. In particular, the segmented Isolation Forest
and K-Means scripts (`Clean_*.py`) are excluded, and so are the SVM, DBSCAN,
LSTM/SHAP and EDA work credited to other team members.

AI assistance: Whether and how AI was used in the original 2025 scripts is
unspecified; it is not inferred here from code style. This rewrite was prepared with
Claude Code under the author's direction.

## Module map

| Staged file | Ported from | What changed and why |
|---|---|---|
| `src/checkout_anomaly/parsing.py` | `_parse_visit` (`fraud_detection (1).py:166-197`), `parse_initial_data` (`test1.py:98-119`), `parse_visit` (`thursday (1).py:79-102`) | **Bug fix:** case files prefix lines with `Transaction N:`. The originals kept only 3-token tuples (`thursday (1).py:91`, `test1.py:111`), so the first item of every visit was silently dropped. The prefix is now stripped and the id kept. Added an optional strict mode and a writer for round-trips. |
| `src/checkout_anomaly/features.py` → groups 1–6 | `extract_features` + `calculate_group_stats` + `calculate_dept_entropy` (`fraud_detection (1).py:52-111, 200-369`); `_compute_visit_features` (`thursday (1).py:131-236`) for `time_delta_cv`, `price_cv` | Same definitions and thresholds (5 s, 60 s, 3 s, code distance > 5). Two exact duplicates were removed: `avg_price_per_item` (= `avg_price`) and `price_time_efficiency` (= `price_time_ratio`). The string feature `most_common_transition` was dropped because detectors need numeric input. Computed per visit in NumPy instead of pandas `groupby.apply`. |
| `features.py` → high-value departments | `HIGH_VALUE_DEPTS = [2, 7, 12]` (`fraud_detection (1).py:47`); `[2, 10, 12, 14]` in `thursday (1).py:45` | The two originals disagree, and both hard-code case-specific codes. Now the *k* most expensive departments by **training** mean price are learned, or an explicit list can be passed. |
| `features.py` → groups 7–9 | `calculate_temporal_features`, `calculate_price_pattern_features`, `calculate_gini`, `calculate_price_run_features`, `calculate_sequence_features` (`fraud beta.py:109-198, 435-531`) | Same formulas. **Changed:** department average prices were computed on the data being scored (`fraud beta.py:532`). They are now learned in `fit` on training visits only. Duplicates removed: `scan_rhythm_consistency` (= `time_delta_cv`), `scan_clustering_score` (= 1 − `dept_switches_rate`) and `path_efficiency` (= `dept_diversity_ratio`). |
| `features.py` → group 10 | `calculate_dept_transitions` + `analyze_sequence_patterns` (`fraud_detection (1).py:405-735`) | Same normalised negative log-likelihood with a 1e-10 floor. **Changed:** the transition matrix is fitted on training visits only, and the score is used as a feature instead of being thresholded at its own 99th percentile. |
| `features.py` → `CorrelationFilter` | `_handle_correlated_features` (`fraud_detection (1).py:371-402`) | Same \|r\| > 0.95 rule. It is now fitted on training data and re-applied to test data; the original re-selected columns on every file. Columns that are constant in training are also dropped. |
| `src/checkout_anomaly/models.py` → `IsolationForestDetector` | `detect_anomalies` (`fraud_detection (1).py:481-497`), `train_models` (`test1.py:222-225`) | Standardisation added in all cases. `test1.py`/`333 clean.py` fitted the forest on unscaled features. Scores are `-decision_function`, as in the original. |
| `models.py` → `KMeansDistanceDetector` | `train_models` / `test_new_data` (`test1.py:206-283`); MiniBatchKMeans distance (`fraud beta.py:819-833`) | PCA(95 %) → KMeans(8) → distance to the nearest centroid, as in `test1.py`. Standardisation added; `test1.py` ran PCA on raw features. |
| `models.py` → `MLPAutoencoderDetector` | `build_meta_ensemble` (`fraud beta.py:805-818`) | Same `(d/2, d/4, d/2)` MLPRegressor with early stopping. `max_iter` raised from 100 to 200. |
| `models.py` → `LOFDetector`, `OneClassSVMDetector`, `GMMDetector` | `detect_anomalies` (`fraud_detection (1).py:501-567`), `thursday (1).py:416-618` | LOF now uses `novelty=True`, so it can score unseen visits; the original fit and scored the same file. The GMM uses diagonal covariances, 4 components and `reg_covar=1e-4` for numerical stability with ~100 features. The original used full covariances, up to 10 components and `reg_covar=1e-6`. The OCSVM defaults (`nu=0.14`, `gamma='auto'`) are unchanged. |
| `models.py` → `KerasAutoencoderDetector` | `FraudDetector._build_autoencoder` / `fit` (`autoecoder.py:342-450`) | Same dense 64-32-64 architecture and MSE loss. **Removed:** the TPU strategy setup, which was Colab-specific. Also removed is the rule layer, because `_apply_rules` (`autoecoder.py:312-339`) implemented only 3 of the 25 thresholds it computed (lines 323-325). That made `suspicious_rule_count=3` require all three, and `high_risk_rule_count=5` could never fire. TensorFlow import is guarded. **Not executed** here, because TensorFlow is not installed. |
| `src/checkout_anomaly/ensemble.py` → `combine_normalised_scores`, `AnomalyEnsemble` | `ensemble_anomaly_detection` (`thursday (1).py:620-678`); `test_new_data` (`test1.py:242-283`, `333 clean.py:609-632`) | **Bug fix 1:** in `thursday (1).py` the block starting at line 659 (`if normalized_scores:`) is indented inside the `for` loop at line 651. The function therefore returned at line 675 after normalising only the first method, so the "ensemble" was the first detector's score. The fixed version averages all methods; the regression test is in `tests/test_ensemble.py`. **Bug fix 2:** `test_new_data` refitted `MinMaxScaler` on the test scores (`test1.py:272-274`) and flagged above the test file's own 85th percentile (`test1.py:278`, `333 clean.py:629`), so every scored file had exactly 15 % flagged. Normalisation constants and the threshold are now fitted on **training** scores. A rank (empirical-CDF) normalisation option was added. |
| `ensemble.py` → `CheckoutAnomalyPipeline` | `__main__` blocks of `test1.py` / `333 clean.py` | One fitted object replaces pickled PCA/KMeans/IF files. **Not ported:** the LSTM "next-department" sequence embeddings (`test1.py:26-94, 129-203`, `fraud beta.py:201-252, 537-575`). They need PyTorch, and the original re-trained a freshly initialised encoder on every scored file (`test_new_data` → `extract_all_features`). Test-time embeddings were therefore not comparable with the ones the models were trained on. |
| `src/checkout_anomaly/evaluation.py` → `evaluate_synthetic` | `evaluate_model_robustness` (`333 clean.py:575-607`) | **Replaced.** The original perturbed rows of the *feature matrix* (×10, ×5/×0.2, + noise). It used an unseeded `np.random`, and it ran after `train_models` had already appended `km_distance` and `iso_score` to the features (`333 clean.py:536-537`), so model scores leaked into the input. The new protocol injects fraud into **raw visits** with the generator, fits on a training set and evaluates on a separate test set. |
| `evaluation.py` → `feature_perturbation_check` | `evaluate_model_robustness` (`333 clean.py:575-607`) | Kept for comparison, with a seed and without the score leak. It is a weak test, because perturbed rows are extreme by construction. |
| `src/checkout_anomaly/synthetic.py` | new | Replaces the course dataset, which cannot be redistributed. It reproduces the schema only (department, seconds since previous scan, price). Its parameters are hand-chosen and **not** fitted to the course data. |
| `src/checkout_anomaly/cli.py`, `examples/run_synthetic_demo.py`, `tests/*` | new | — |

## Other original defects found during re-verification (not ported)

- `fraud beta.py:1061-1062`: `main` calls `analyze_results()` and `visualize_anomalies()`, but neither is defined in that file, so the script would raise `AttributeError`.
- `thursday (1).py:1012-1046`: `__main__` fits every detector on the test file it scores (`case12.csv`) and prints "Top 150 … in case11.csv". It min-max normalises per price segment and then ranks the two segments' scores together.
- `autoecoder.py:674`: `get_top_suspicious_transactions` fits the model on the suspicious file itself.
- `autoecoder.py:684`: `combined_risk_score = 0.7 * raw MSE + 0.3 * rule score scaled to 0–100`, which mixes two scales.
- `autoecoder.py:276-280`: `_get_price_runs` is a stub.
- `autoecoder.py:78, 249`: `max_dept_id=15`, so departments 16–18 never get per-department features.
- `test1.py` / `333 clean.py`: hard-coded Colab paths (`/content/...`). The other scripts have absolute local paths.

## Framing (18 Nov 2026)
- The case-study framing was replaced by explicit individual / shared / teammates' contribution statements (author decision). See `CONTRIBUTIONS.md`.
