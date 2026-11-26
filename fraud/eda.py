"""Step 2 numbers: how rare is fraud, when does it happen, how big is it, and which features separate it?

Pure functions on DataFrames (no file access, no plotting), all vectorised.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score

from fraud.config import DataConfig, SplitConfig
from fraud.splits import time_split_bounds

AMOUNT_EDGES = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf)


def feature_columns(cfg: DataConfig = DataConfig()) -> list:
    """The columns that describe a transaction: V1-V28, Amount and Time."""
    return [*cfg.pca_columns, cfg.amount_column, cfg.time_column]


def require_columns(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> None:
    """Raise a clear error if a needed column is missing."""
    missing = [c for c in cfg.required_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def amount_bands(frame: pd.DataFrame, cfg: DataConfig = DataConfig(), edges: Sequence[float] = AMOUNT_EDGES) -> pd.DataFrame:
    """Fraud rate per transaction-amount band (the lowest band is ``[0, 1)``, so zero amounts are included)."""
    require_columns(frame, cfg)
    band = pd.cut(frame[cfg.amount_column], bins=list(edges), right=False, include_lowest=True)
    grouped = frame.groupby(band, observed=False)[cfg.label_column]
    table = pd.DataFrame({"transactions": grouped.size(), "frauds": grouped.sum()})
    table["fraud_rate"] = (table["frauds"] / table["transactions"].where(table["transactions"] > 0)).round(6)
    table["share_of_frauds"] = (table["frauds"] / max(int(table["frauds"].sum()), 1)).round(4)
    table.index = [str(i) for i in table.index]
    table.index.name = "amount_band"
    return table.reset_index()


def hourly_pattern(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Transactions, frauds and fraud rate by hour of the data's own clock (hours since the first transaction, modulo 24).

    ``Time`` is relative to the first transaction, so this is not necessarily the wall-clock hour.
    """
    require_columns(frame, cfg)
    hour = ((frame[cfg.time_column] // 3600).astype(int)) % 24
    grouped = frame.groupby(hour)[cfg.label_column]
    table = pd.DataFrame({"transactions": grouped.size(), "frauds": grouped.sum()})
    table = table.reindex(range(24), fill_value=0)
    table["fraud_rate"] = (table["frauds"] / table["transactions"].where(table["transactions"] > 0)).round(6)
    table.index.name = "hour"
    return table.reset_index()


def feature_separation(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """How well each single feature separates fraud from genuine rows, ranked by the KS statistic.

    * ``cohens_d``: difference of means in pooled standard deviations (the sign says which way fraud goes).
    * ``ks``: largest gap between the two cumulative distributions (0 = identical, 1 = fully separated).
    * ``single_feature_pr_auc``: PR-AUC of the raw feature used as a score, taking whichever direction is better.
    * ``corr_with_class``: point-biserial (Pearson) correlation with the label.
    """
    require_columns(frame, cfg)
    y = frame[cfg.label_column].to_numpy()
    rows = []
    for column in feature_columns(cfg):
        x = frame[column].to_numpy(dtype=float)
        fraud, genuine = x[y == 1], x[y == 0]
        pooled = np.sqrt(((len(fraud) - 1) * fraud.var(ddof=1) + (len(genuine) - 1) * genuine.var(ddof=1)) / (len(x) - 2))
        ap = max(average_precision_score(y, x), average_precision_score(y, -x))
        rows.append({"feature": column, "mean_fraud": float(fraud.mean()), "mean_genuine": float(genuine.mean()),
                     "cohens_d": float((fraud.mean() - genuine.mean()) / pooled) if pooled > 0 else 0.0,
                     "ks": float(ks_2samp(fraud, genuine).statistic), "single_feature_pr_auc": float(ap),
                     "corr_with_class": float(np.corrcoef(x, y)[0, 1])})
    return pd.DataFrame(rows).sort_values("ks", ascending=False, kind="stable").round(4).reset_index(drop=True)


def feature_correlation(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> Dict[str, Any]:
    """Largest absolute correlation between two *different* features, and the average, among V1-V28 (PCA, so close to 0)."""
    require_columns(frame, cfg)
    corr = frame[list(cfg.pca_columns)].corr().to_numpy()
    off = np.abs(corr[~np.eye(len(corr), dtype=bool)])
    return {"max_abs_between_pca_features": round(float(off.max()), 4), "mean_abs_between_pca_features": round(float(off.mean()), 4),
            "amount_vs_time": round(float(frame[cfg.amount_column].corr(frame[cfg.time_column])), 4)}


def split_balance(frame: pd.DataFrame, split: SplitConfig = SplitConfig(), cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Rows, frauds and fraud rate in the train / validation / test parts of a time-based split (data sorted by time first)."""
    require_columns(frame, cfg)
    label = frame.sort_values(cfg.time_column, kind="stable")[cfg.label_column].to_numpy()
    train_end, val_end = time_split_bounds(len(label), split)
    parts = {"train": label[:train_end], "validation": label[train_end:val_end], "test": label[val_end:]}
    return pd.DataFrame([{"part": k, "rows": len(v), "frauds": int(v.sum()), "fraud_rate": round(float(v.mean()), 6)} for k, v in parts.items()])


def split_drift(frame: pd.DataFrame, split: SplitConfig = SplitConfig(), cfg: DataConfig = DataConfig(), flag: float = 0.10) -> pd.DataFrame:
    """KS distance between each feature's distribution in the training period and in the test period (Time excluded).

    A large value means the newest transactions look different from the oldest: the model will face drift in production.
    """
    require_columns(frame, cfg)
    ordered = frame.sort_values(cfg.time_column, kind="stable")
    train_end, val_end = time_split_bounds(len(ordered), split)
    train, test = ordered.iloc[:train_end], ordered.iloc[val_end:]
    rows = [{"feature": c, "ks_train_vs_test": float(ks_2samp(train[c], test[c]).statistic)}
            for c in feature_columns(cfg) if c != cfg.time_column]
    table = pd.DataFrame(rows).sort_values("ks_train_vs_test", ascending=False, kind="stable").round(4).reset_index(drop=True)
    table["flagged"] = table["ks_train_vs_test"] > flag
    return table


def build_summary(frame: pd.DataFrame, split: SplitConfig = SplitConfig(), cfg: DataConfig = DataConfig()) -> Dict[str, Any]:
    """Every Step 2 finding as a JSON-ready dict."""
    require_columns(frame, cfg)
    amounts = amount_bands(frame, cfg)
    hours = hourly_pattern(frame, cfg)
    sep = feature_separation(frame, cfg)
    drift = split_drift(frame, split, cfg)
    label = frame[cfg.label_column]
    zero = frame[cfg.amount_column] == 0
    busiest = hours.loc[hours["transactions"].idxmax()]
    riskiest = hours[hours["transactions"] >= 500].sort_values("fraud_rate", ascending=False).head(3)
    return {
        "rows": int(len(frame)), "frauds": int(label.sum()), "fraud_rate": round(float(label.mean()), 6),
        "duplicate_rows": int(frame.duplicated().sum()), "duplicate_frauds": int(frame[frame.duplicated(keep="first")][cfg.label_column].sum()),
        "zero_amount": {"rows": int(zero.sum()), "frauds": int(label[zero].sum()), "fraud_rate": round(float(label[zero].mean()), 6) if zero.any() else None},
        "amount_bands": amounts.to_dict(orient="records"),
        "amount_skew": {"fraud": round(float(frame.loc[label == 1, cfg.amount_column].skew()), 2),
                        "genuine": round(float(frame.loc[label == 0, cfg.amount_column].skew()), 2)},
        "hourly": hours.to_dict(orient="records"), "busiest_hour": int(busiest["hour"]),
        "riskiest_hours": riskiest[["hour", "transactions", "frauds", "fraud_rate"]].to_dict(orient="records"),
        "feature_separation_top": sep.head(10).to_dict(orient="records"),
        "weakest_features": sep.tail(5)[["feature", "ks", "single_feature_pr_auc"]].to_dict(orient="records"),
        "correlation": feature_correlation(frame, cfg),
        "time_split": split_balance(frame, split, cfg).to_dict(orient="records"),
        "drift_flagged_features": drift[drift["flagged"]].to_dict(orient="records"),
        "drift_max_ks": float(drift["ks_train_vs_test"].max()),
    }
