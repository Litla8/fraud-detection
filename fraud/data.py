"""Load the ULB credit-card file and check it is what we expect."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from fraud.config import DataConfig

LOGGER = logging.getLogger(__name__)


def validate_columns(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> None:
    """Raise if columns are missing, labels are not 0/1, values are missing, or amounts are negative.

    Raises:
        ValueError: With a message naming what is wrong.
    """
    missing = [c for c in cfg.required_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    numeric = frame[list(cfg.required_columns)]
    if not all(pd.api.types.is_numeric_dtype(t) for t in numeric.dtypes):
        raise ValueError("All model columns must be numeric")
    if numeric.isna().any().any():
        raise ValueError("The data contains missing values")
    if not frame[cfg.label_column].isin([0, 1]).all():
        raise ValueError(f"{cfg.label_column} must contain only 0 and 1")
    if (frame[cfg.amount_column] < 0).any():
        raise ValueError(f"{cfg.amount_column} must not be negative")
    if frame[cfg.label_column].nunique() < 2:
        raise ValueError("The data needs both genuine and fraudulent rows")


def load_raw(path: Path, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Read ``creditcard.csv`` with compact dtypes (float32 features, int8 label), then validate it.

    Raises:
        FileNotFoundError: With a hint on where to put the Kaggle file.
        ValueError: From :func:`validate_columns`.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Download creditcard.csv from Kaggle (mlg-ulb/creditcardfraud) into that folder.")
    dtypes = {c: "float32" for c in cfg.pca_columns}
    dtypes.update({cfg.label_column: "int8", cfg.time_column: "float64", cfg.amount_column: "float64"})
    frame = pd.read_csv(path, dtype=dtypes)
    validate_columns(frame, cfg)
    if len(frame) != cfg.expected_rows or int(frame[cfg.label_column].sum()) != cfg.expected_frauds:
        LOGGER.warning("expected %d rows with %d frauds, found %d with %d (fine for a sample, suspicious for the full file)",
                       cfg.expected_rows, cfg.expected_frauds, len(frame), int(frame[cfg.label_column].sum()))
    return frame


def profile(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> Dict[str, Any]:
    """Size, imbalance, duplicates, time span and amount statistics for fraud vs genuine rows (vectorised)."""
    label = frame[cfg.label_column]
    fraud = frame[label == 1]
    genuine = frame[label == 0]

    def describe(amount: pd.Series) -> Dict[str, float]:
        return {"mean": round(float(amount.mean()), 2), "median": round(float(amount.median()), 2),
                "p95": round(float(amount.quantile(0.95)), 2), "max": round(float(amount.max()), 2),
                "total": round(float(amount.sum()), 2)}

    return {"rows": int(len(frame)), "frauds": int(len(fraud)), "fraud_rate": round(float(label.mean()), 6),
            "imbalance_ratio": round(float(len(genuine) / max(len(fraud), 1)), 1),
            "duplicate_rows": int(frame.duplicated().sum()),
            "hours_covered": round(float(frame[cfg.time_column].max() - frame[cfg.time_column].min()) / 3600, 1),
            "time_sorted": bool(frame[cfg.time_column].is_monotonic_increasing),
            "amount": {"fraud": describe(fraud[cfg.amount_column]), "genuine": describe(genuine[cfg.amount_column])},
            "zero_amount_rows": int((frame[cfg.amount_column] == 0).sum())}
