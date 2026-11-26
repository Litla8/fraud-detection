# Credit Card Fraud Detection System

Detect fraudulent card transactions in extremely imbalanced data (0.17% fraud), choose a decision threshold from a
business cost matrix, and serve the model as a low-latency API. Dataset: Kaggle *Credit Card Fraud Detection* (ULB).

## Status

- [x] Step 1: Define the problem and costs (cost matrix, targets, metrics, config)
- [x] Step 2: Data exploration
- [ ] Step 3: Preprocessing (RobustScaler, time-based split)
- [ ] Step 4: Handling imbalance (class weights, under-sampling, SMOTE, ADASYN, focal-style weighting)
- [ ] Step 5: Supervised models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- [ ] Step 6: Anomaly detection (Isolation Forest, One-Class SVM, autoencoder) and the hybrid
- [ ] Step 7: Feature engineering (not possible on this file: no customer column)
- [ ] Step 8: Threshold optimisation and risk tiers
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

## Structure

```
fraud/     config.py, data.py, metrics.py, costs.py, define.py (Step 1), splits.py, eda.py, plots.py, explore.py (Step 2), logging_utils.py
tests/     pytest suite (synthetic data, no Kaggle file needed)
docs/      project_brief.md
data/, models/, reports/   generated files (git-ignored)
```
