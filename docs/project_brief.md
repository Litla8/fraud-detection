# Project brief: Credit Card Fraud Detection System

**Business problem.** A bank wants to flag suspicious card transactions in real time while keeping false alarms low,
because blocking a genuine customer hurts the customer experience.

**Data.** Kaggle *Credit Card Fraud Detection* (ULB Machine Learning Group): `creditcard.csv`, 284,807 transactions made by
European cardholders over two days in September 2013, 492 of them fraud (0.172%). Columns: `Time` (seconds since the first
transaction), `V1`-`V28` (anonymised PCA components), `Amount`, `Class` (1 = fraud). There is no customer, merchant or
location column, so the behavioural features in the brief's Step 7 are not possible on this file (see below).

## Success metrics
| Metric | Target |
|---|---|
| PR-AUC (test set) | above 0.80 |
| Recall at precision of at least 0.85 | at least 0.80 |
| Cost savings | documented against "approve everything" |

Accuracy is **not** used: always predicting "genuine" scores 99.83%.

## Cost matrix (assumptions to confirm with the business)
| Outcome | Cost |
|---|---|
| Missed fraud (false negative) | the transaction amount plus a EUR 15 chargeback / handling fee |
| False alarm (false positive) | EUR 3 (declined card, OTP prompt, support call, some lost customers) |
| Caught fraud (true positive) | EUR 1 analyst review, and the amount is saved |
| OTP check on a genuine customer | EUR 0.25 |

The threshold in Step 8 is the one that minimises total cost on the validation data, and savings are reported against
approving everything. Change the numbers in `fraud/config.py` (`CostConfig`) and every report follows.

## Decisions made now
- **Split:** time-based (oldest transactions train, newest test) to mimic production; validation is the end of the training period.
  The test set is never resampled and keeps its natural imbalance.
- **Resampling** (SMOTE, ADASYN, under-sampling) happens only inside cross-validation folds, never before the split.
- **Scaling:** `RobustScaler` on `Time` and `Amount` (fitted on training rows only); V1-V28 are already PCA-scaled.
- **Step 7 (behavioural features)** is skipped for this dataset: with no customer id there is nothing to aggregate per customer.
  Velocity-style features are documented as future work for a dataset such as IEEE-CIS.

## Deliverables
Imbalance strategy comparison report, trained model (the Step 6 hybrid with an autoencoder was built and tested but added no validation PR-AUC, so the supervised model is deployed), FastAPI scoring service with Docker, cost-benefit analysis (`reports/cost_benefit.md`).
