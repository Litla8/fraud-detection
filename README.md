# Credit Card Fraud Detection System

Detect fraudulent card transactions in extremely imbalanced data (0.17% fraud), choose a decision threshold from a
business cost matrix, and serve the model as a low-latency API. Dataset: Kaggle *Credit Card Fraud Detection* (ULB).

## Status

- [x] Step 1: Define the problem and costs (cost matrix, targets, metrics, config)
- [x] Step 2: Data exploration
- [x] Step 3: Preprocessing (duplicates, RobustScaler, time-based and stratified splits)
- [x] Step 4: Handling imbalance (class weights, under-sampling, SMOTE, ADASYN, focal-style weighting)
- [x] Step 5: Supervised models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- [x] Step 6: Anomaly detection (Isolation Forest, One-Class SVM, autoencoder) and the hybrid
- [ ] Step 7: Feature engineering (not possible on this file: no customer column)
- [x] Step 8: Threshold optimisation and risk tiers
- [ ] Step 9: SHAP, FastAPI, Docker, drift monitoring

## Targets

PR-AUC above 0.80; recall of at least 0.80 at precision of at least 0.85; documented cost savings. See [docs/project_brief.md](docs/project_brief.md).

## Setup

Download `creditcard.csv` from Kaggle (`mlg-ulb/creditcardfraud`) into `data/raw/`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python -m fraud.define     # writes reports/problem_definition.json
```

## Step 1: problem and costs

`python -m fraud.define` loads and validates the file, then writes `reports/problem_definition.json`: the imbalance,
amount statistics for fraud and genuine rows, the accuracy of a model that never flags anything, the cost of approving
everything and of blocking everything, and the break-even precision implied by the cost matrix.

* `fraud/config.py`: every setting, including `CostConfig` (the cost matrix, written as assumptions to confirm with the business).
* `fraud/metrics.py`: PR-AUC, recall at a precision floor, F2, MCC and the target check. No accuracy.
* `fraud/costs.py`: total cost of a set of decisions, the cost of every possible threshold from one sort, and the cheapest threshold.
* `fraud/data.py`: loading with validation and a profile of the file.

## Step 2: data exploration

```powershell
python -m fraud.explore     # reports/eda_summary.json, feature_separation.csv, split_drift.csv and 7 charts in reports/figures/
```

Answers: how rare is fraud, how do amounts and timing differ for fraud and genuine transactions, which single features
separate fraud best (KS statistic, Cohen's d, single-feature PR-AUC), whether the PCA features are really uncorrelated,
how many frauds each part of the time-based split gets, and how much the newest transactions drift from the oldest
(KS distance per feature). `Time` is seconds since the first transaction, so the "hour" charts use the data's own clock.

## Step 3: preprocessing

```powershell
python -m fraud.preprocess                  # both splits (about 10 seconds)
python -m fraud.preprocess --split time     # only the time-based split
```

1. **Duplicates:** the 1,081 exact duplicate rows (19 frauds) are dropped before splitting, so a copy can never sit in both train and test (`--keep-duplicates` turns this off).
2. **Two splits**, with the same sizes (70% train, 15% validation, 20% test of the rows; test is the newest 20% for the time split):
   `time` (oldest rows train, newest test, mimics production, the primary split) and `stratified` (random, same fraud rate everywhere).
   Comparing a model on both shows how much the drift from Step 2 costs.
3. **Scaling:** `RobustScaler` on `Amount` and `Time`, **fitted on the training rows only** and applied to validation and test. V1-V28 are left as they are.
4. **The test set is never resampled or rebalanced**; resampling (Step 4) happens only inside cross-validation folds on training data.
5. Safety checks before saving: parts are disjoint, every part contains fraud, and the time split is in chronological order.

Output per split in `data/processed/<split>/` (`train.pkl`, `validation.pkl`, `test.pkl`): the 30 model features
(`V1`-`V28`, `Amount_scaled`, `Time_scaled`), `Class`, the raw `Amount_raw` and `Time_raw` (the cost matrix needs the real amount)
and `row_id` (position in the original file). Scalers go to `models/scaler_<split>.joblib`, and
`reports/preprocessing_report.json` records the duplicates, the size and fraud count of every part, the checks and the scaler values.
Later steps read the data with `fraud.preprocess.load_split(cfg, "time")`.

## Step 4: handling imbalance

```powershell
python -m fraud.run_imbalance --sample   # quick check, about 1 minute
python -m fraud.run_imbalance            # full comparison, about 5-8 minutes
python -m fraud.run_imbalance --split stratified   # the same on the random split
```

Six strategies, each used to train the same probe model (scikit-learn histogram gradient boosting, L2-regularised, fixed 150 iterations):
no handling (baseline), balanced class weights, random under-sampling, SMOTE, ADASYN and focal-style weighting.

* **Resampling only inside folds.** Each strategy is applied to the training rows of every cross-validation fold; validation and test
  rows are never resampled, so scores reflect the real 0.17% fraud rate. SMOTE, ADASYN and under-sampling are written in
  numpy / scikit-learn in `fraud/imbalance.py` (fully unit-tested) rather than taken from imbalanced-learn.
* **Focal-style weighting** approximates focal loss with sample weights: a row's weight is `floor + (1 - p_true) ** gamma`, where `p_true`
  comes from out-of-fold predictions, then the weights are normalised to mean 1. It is not a custom loss function.
* **Selection** uses the mean 5-fold PR-AUC on the training data. Strategies within one standard deviation of the best are reported as a
  tie. The probability threshold behind the cost columns is chosen from out-of-fold scores, never from validation.
* Outputs: `reports/imbalance_comparison.csv` and `.json` (the winner and the ties), `reports/figures/imbalance_cv.png` and `imbalance_pr_curves.png`.
  The test set is not used in this step.

## Step 5: supervised models

```powershell
python -m fraud.train_supervised --sample     # quick check, about 1-2 minutes
python -m fraud.train_supervised              # full run: 4 models x 2 variants, 5-fold CV each (10-25 minutes)
python -m fraud.train_supervised --models logistic_regression xgboost   # only some models
```

Logistic Regression (baseline), Random Forest, XGBoost and LightGBM, each trained two ways: plain, and with SMOTE (the Step 4
cross-validation winner, which was statistically tied with doing nothing). Hyper-parameters are fixed and sensible
(`SupervisedConfig`): no tuning, so nothing is fitted to validation or test.

* **Selection** uses only the mean 5-fold PR-AUC on the training data; models within one standard deviation of the best are reported as a tie.
* **Metrics** (all in `reports/model_comparison.csv/json`): PR-AUC with a 95% bootstrap interval, recall at precision of at least 0.85,
  F2 and MCC, and the cost saved against approving everything.
* **Honest thresholds.** The F2 and cost thresholds come from out-of-fold training scores. "Transferred" recall and precision use a
  threshold picked on validation and applied to test. "Recall at precision 0.85" picks the threshold on the test scores themselves, so it is the
  best case and is the number the project target refers to.
* **The test set has only about 74 frauds.** Read every test score together with its interval: differences smaller than the interval are noise.
* Outputs: `models/supervised_best.joblib` plus `supervised_best.json` (features and thresholds), and charts in `reports/figures/`
  (`model_comparison.png`, `model_pr_curves.png`, `feature_importance.png`). If XGBoost or LightGBM is not installed, that model is skipped with a warning.

## Step 6: anomaly detection and the hybrid

```powershell
python -m fraud.train_anomaly --sample     # quick check, about 15 seconds
python -m fraud.train_anomaly              # full run, about 1 minute (needs the Step 5 model: run Step 5 without --sample first)
```

Three detectors learn what a **genuine** transaction looks like (fraud rows are never shown to them) and score how unusual a row is:
Isolation Forest, One-Class SVM (on a random 8,000 genuine rows, because it is quadratic) and an autoencoder (a small
`MLPRegressor` that rebuilds its input through a narrow layer; the score is the reconstruction error). Features are clipped to +-10 first.

* **Percentile scores.** Each detector's raw score is turned into a percentile against held-out *validation* genuine rows
  ("more unusual than 99% of genuine transactions"), so it means the same thing for one transaction or a million.
* **Hybrid = (1 - w) x supervised probability + w x anomaly percentile.** The detector and weight w are picked on validation PR-AUC only,
  and the hybrid replaces the supervised model only if it adds at least 0.01 (`min_gain`); otherwise the report says "hybrid = supervised (no gain)".
* **Rescue analysis.** For the frauds the supervised model misses, how many does each detector alert (above the 99th percentile), and at what cost in extra false alarms?
* Outputs: `reports/anomaly_comparison.csv/json`, `blend_grid.csv`, charts `anomaly_separation.png`, `blend_weights.png`, `hybrid_pr_curves.png`,
  and `models/anomaly_detectors.joblib` plus `models/hybrid.json` (the choice, weights and thresholds Step 8 and 9 will use).

## Step 8: thresholds, risk tiers and cost-benefit

```powershell
python -m fraud.optimize_threshold      # about 10-20 seconds; needs the Step 5 model
```

Uses the Step 5 supervised model (the hybrid of Step 6 was not adopted: it added no validation PR-AUC). **Every threshold is chosen on
validation scores and applied unchanged to the test set.**

* **Block-or-approve thresholds** compared on the test set: F2-optimal, precision-0.85, cost-optimal (one point and the median of 200
  bootstrap resamples of validation, because 49 validation frauds make the optimum jump around), and a clearly labelled *best case* that
  picks the threshold on the test set itself (not achievable in practice).
* **Three risk tiers** (`fraud/policy.py`): score below `t_medium` is approved (low), from `t_medium` to `t_high` gets an OTP check (medium),
  from `t_high` is blocked and reviewed (high). The best pair is found exactly by trying every pair of candidate cut points.
  An OTP stops 90% of fraud (assumption) and costs EUR 0.25 for every medium row.
* **Cost-benefit** (`reports/cost_benefit.csv`, `cost_benefit.md`): cost and saving against "approve everything" for every strategy, a 95%
  bootstrap interval for the tiered saving, the saving per 1,000,000 transactions, and `sensitivity.csv`, which re-chooses the thresholds under
  false-alarm costs of EUR 1 / 3 / 10 and fixed missed-fraud fees of EUR 0 / 15 / 50 and scores them on the test set.
* Outputs: `reports/threshold_report.json`, charts `cost_curves.png`, `risk_tiers.png`, `threshold_stability.png`, and
  **`models/decision_policy.json`** (the thresholds, tier actions and cost assumptions that the Step 9 API will load).
* The cost-optimal block threshold flags generously (break-even precision is about 2%), so it differs from the brief's precision-0.85 target;
  both are reported.

## Structure

```
fraud/     config.py, data.py, metrics.py, costs.py, define.py (Step 1), splits.py, eda.py, plots.py, explore.py (Step 2), preprocess.py (Step 3), imbalance.py + run_imbalance.py (Step 4), models.py + train_supervised.py (Step 5), anomaly.py + train_anomaly.py (Step 6), policy.py + optimize_threshold.py (Step 8), logging_utils.py
tests/     pytest suite (synthetic data, no Kaggle file needed)
docs/      project_brief.md
data/, models/, reports/   generated files (git-ignored)
```
