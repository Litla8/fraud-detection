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
- [x] Step 7: Feature engineering: skipped on purpose (no customer or merchant column in this file, so nothing to aggregate per customer)
- [x] Step 8: Threshold optimisation and risk tiers
- [x] Step 9: SHAP, FastAPI, Docker, drift monitoring

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

Logistic Regression (baseline), Random Forest, XGBoost and LightGBM, each trained two ways: plain, and with SMOTE (statistically tied
with doing nothing in the Step 4 comparison, so both are tried). Hyper-parameters are fixed and sensible
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
  An OTP stops 90% of fraud (assumption) and costs EUR 0.25 for every medium row. The medium tier is capped at 1% of all transactions
  (`max_medium_share`, the step-up capacity): without the cap the cheapest tiers on this data send 5-6% of customers to an OTP check for about one extra fraud stopped.
* **Cost-benefit** (`reports/cost_benefit.csv`, `cost_benefit.md`): cost and saving against "approve everything" for every strategy, a 95%
  bootstrap interval for the tiered saving, the saving per 1,000,000 transactions, and `sensitivity.csv`, which re-chooses the thresholds under
  false-alarm costs of EUR 1 / 3 / 10 and fixed missed-fraud fees of EUR 0 / 15 / 50 and scores them on the test set.
* Outputs: `reports/threshold_report.json`, charts `cost_curves.png`, `risk_tiers.png`, `threshold_stability.png`, and
  **`models/decision_policy.json`** (the thresholds, tier actions and cost assumptions that the Step 9 API will load).
* The cost-optimal block threshold flags generously (break-even precision is about 2%), so it differs from the brief's precision-0.85 target;
  both are reported.

## Step 9: explanations, API, Docker and drift monitoring

```powershell
python -m fraud.explain             # SHAP importance + example explanations   -> reports/shap_importance.csv, explanations.json, 2 charts
python -m fraud.drift               # drift reference from training data + drift check -> models/drift_reference.json, reports/drift_report.*
python -m fraud.benchmark           # latency against the 100 ms budget         -> reports/latency_report.json (exit code 1 if over budget)
uvicorn fraud.api:create_default_app --factory --port 8000      # run the service locally, docs at http://localhost:8000/docs
python -m fraud.smoke_test          # send real test transactions to the running service
```

* **Explanations** (`fraud/explain.py`): LightGBM and XGBoost give exact TreeSHAP values natively (`pred_contrib`), identical to
  `shap.TreeExplainer` for those models and without the shap package in the image; `--method shap` cross-checks with the package. Forests use shap.
  Features are anonymised PCA components, so "V14 raised the risk" is statistically valid but not a business reason a customer could read.
* **Service** (`fraud/scoring.py`, `fraud/api.py`): `POST /score` (one transaction), `POST /score/batch` (up to 1,000), `GET /health`, `GET /policy`, `GET /drift`;
  add `?explain=true` for the top reasons. Every response carries the score, the **tier** (`low` approve, `medium` OTP check, `high` block and review), the action,
  and `latency_ms`; the `X-Process-Time-ms` header gives the server time. Inputs are validated (all 30 fields required, finite, amount >= 0, no extra fields).
  Thresholds come from `models/decision_policy.json` (Step 8); the scaler is the Step 3 one, so raw `Time` and `Amount` go in as in the Kaggle file.
* **Latency**: the scoring core runs single-threaded per request (a thread pool costs more than it saves for one row). `fraud.benchmark` reports p50 / p95 / p99
  for scoring, scoring with reasons, and (when FastAPI and httpx are installed) the HTTP round trip, against the 100 ms budget.
* **Drift** (`fraud/drift.py`): population stability index per input and for the model score, against training data (scores against validation).
  Below 0.10 stable, 0.10 to 0.25 watch, above 0.25 alert; overall `alert` when the score or a quarter of the features alert. The API keeps the latest 5,000
  scored transactions and `GET /drift` reports once 500 have arrived. `Time_scaled` is excluded: a clock only moves forward, so it always looks "drifted".
  `fraud.drift` also checks validation and four slices of the test period so you can see drift grow over time.
* **Docker**: the image holds the `fraud/` package and the `models/` files, nothing else.

```powershell
python -m fraud.pin_requirements                       # writes requirements-api.txt pinned to YOUR installed versions (a saved model must load with the same scikit-learn / lightgbm)
docker build -t fraud-api --build-arg PYTHON_VERSION=3.14 .       # use the version pin_requirements prints
docker run --rm -p 8000:8000 fraud-api
python -m fraud.smoke_test --url http://localhost:8000
```

Example request (`Time`, `V1`-`V28`, `Amount` as in the Kaggle file):

```powershell
curl.exe -X POST "http://localhost:8000/score?explain=true" -H "Content-Type: application/json" -d "@sample.json"
```

Collect everything into one readable document at any point (sections for steps you have not run say which command creates them):

```powershell
python -m fraud.report      # -> reports/RESULTS.md: scorecard against the targets, every step's key numbers, limitations
```

Known limitations: `Time_scaled` is a model input, but time since the first transaction grows without bound in production, so a deployed model would see values far
outside its training range (trees treat them as "latest period"). With LightGBM it is the single most influential feature by SHAP (13% in our run), so a production
version should retrain without it: set `use_time=False` in `PreprocessConfig`, rerun Steps 3, 5, 8 and 9, and compare PR-AUC (see the results section).
The service has no authentication, rate limiting or persistent logging: it is a demonstration of the scoring path.

## Structure

```
fraud/     config.py, data.py, metrics.py, costs.py, define.py (Step 1), splits.py, eda.py, plots.py, explore.py (Step 2), preprocess.py (Step 3), imbalance.py + run_imbalance.py (Step 4), models.py + train_supervised.py (Step 5), anomaly.py + train_anomaly.py (Step 6), policy.py + optimize_threshold.py (Step 8), report.py, explain.py, schemas.py, scoring.py, api.py, drift.py, benchmark.py, smoke_test.py, pin_requirements.py (Step 9), logging_utils.py
tests/     pytest suite (synthetic data, no Kaggle file needed)
docs/      project_brief.md
Dockerfile, .dockerignore, requirements-api.txt (generated)
data/, models/, reports/   generated files (git-ignored)
```
