"""Tests for fraud.config."""
from __future__ import annotations

from pathlib import Path

import pytest

from fraud.config import CostConfig, DataConfig, PathsConfig, SplitConfig


def test_paths_and_columns() -> None:
    assert PathsConfig(root=Path("x")).raw_path == Path("x/data/raw/creditcard.csv")
    cols = DataConfig().required_columns
    assert cols[0] == "Time" and cols[-1] == "Class" and len(cols) == 31


@pytest.mark.parametrize("kwargs", [{"fp_cost": -1}, {"otp_fraud_catch_rate": 1.5}, {"fp_cost": 0, "fn_fixed": 0, "fn_uses_amount": False}])
def test_cost_config_rejects_bad_values(kwargs) -> None:
    with pytest.raises(ValueError):
        CostConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"method": "random"}, {"test_fraction": 0.7}, {"cv_folds": 1}])
def test_split_config_rejects_bad_values(kwargs) -> None:
    with pytest.raises(ValueError):
        SplitConfig(**kwargs)
