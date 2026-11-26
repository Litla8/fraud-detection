"""Row positions for the time-based train / validation / test split (used by exploration now, preprocessing next)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from fraud.config import DataConfig, SplitConfig


def time_split_bounds(n_rows: int, cfg: SplitConfig = SplitConfig()) -> Tuple[int, int]:
    """``(train_end, val_end)`` row positions: rows ``[:train_end]`` train, ``[train_end:val_end]`` validate, ``[val_end:]`` test.

    The test set is the newest ``test_fraction`` of rows; validation is the ``val_fraction`` of rows just before it.

    Raises:
        ValueError: If there are too few rows for three non-empty parts.
    """
    val_end = n_rows - int(round(n_rows * cfg.test_fraction))
    train_end = val_end - int(round(n_rows * cfg.val_fraction))
    if train_end < 1 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("too few rows for the requested split fractions")
    return train_end, val_end


def split_by_time(frame: pd.DataFrame, cfg: SplitConfig = SplitConfig(), data: DataConfig = DataConfig()
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by time and cut into train / validation / test (copies; the input is untouched).

    Raises:
        ValueError: If the time column is missing or the frame is too small.
    """
    if data.time_column not in frame.columns:
        raise ValueError(f"Missing column: {data.time_column}")
    ordered = frame.sort_values(data.time_column, kind="stable").reset_index(drop=True)
    train_end, val_end = time_split_bounds(len(ordered), cfg)
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:val_end].copy(), ordered.iloc[val_end:].copy()
