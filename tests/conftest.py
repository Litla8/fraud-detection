"""Deterministic synthetic transactions shaped like the ULB file (tiny, seeded, with rare fraud)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud.config import DataConfig


def make_transactions(n: int = 2000, frauds: int = 20, seed: int = 42) -> pd.DataFrame:
    """``n`` rows sorted by time; fraud rows have shifted V1-V3 and larger amounts, so models can learn something."""
    rng = np.random.default_rng(seed)
    cfg = DataConfig()
    frame = pd.DataFrame(rng.normal(size=(n, cfg.n_components)), columns=list(cfg.pca_columns))
    label = np.zeros(n, dtype=int)
    label[rng.choice(n, frauds, replace=False)] = 1
    frame.loc[label == 1, ["V1", "V2", "V3"]] += 3.0
    frame[cfg.time_column] = np.sort(rng.uniform(0, 172_800, n))
    frame[cfg.amount_column] = np.where(label == 1, rng.gamma(2.0, 80.0, n), rng.gamma(2.0, 30.0, n)).round(2)
    frame[cfg.label_column] = label
    return frame[[cfg.time_column, *cfg.pca_columns, cfg.amount_column, cfg.label_column]]


@pytest.fixture()
def transactions() -> pd.DataFrame:
    return make_transactions()
