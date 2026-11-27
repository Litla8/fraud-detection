# Credit Card Fraud Detection System

Detect fraudulent card transactions in extremely imbalanced data (0.17% fraud), choose a decision threshold from a
business cost matrix, and serve the model as a low-latency API. Dataset: Kaggle *Credit Card Fraud Detection* (ULB).

## Status

- [x] Step 1: Define the problem and costs (cost matrix, targets, metrics, config)
- [x] Step 2: Data exploration
- [x] Step 3: Preprocessing (duplicates, RobustScaler, time-based and stratified splits)
- [x] Step 4: Handling imbalance (class weights, under-sampling, SMOTE, ADASYN, focal-style weighting)
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

## Structure

```
fraud/     config.py, data.py, metrics.py, costs.py, define.py (Step 1), splits.py, eda.py, plots.py, explore.py (Step 2), preprocess.py (Step 3), imbalance.py + run_imbalance.py (Step 4), logging_utils.py
tests/     pytest suite (synthetic data, no Kaggle file needed)
docs/      project_brief.md
data/, models/, reports/   generated files (git-ignored)
```
