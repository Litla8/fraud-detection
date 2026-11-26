"""Tests for fraud.splits, fraud.eda, fraud.plots and fraud.explore (synthetic data)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud import eda, explore, plots, splits
from fraud.config import ProjectConfig, SplitConfig
from tests.conftest import make_transactions


def test_time_split_bounds_and_split_by_time(transactions) -> None:
    assert splits.time_split_bounds(1000, SplitConfig()) == (650, 800)
    with pytest.raises(ValueError):
        splits.time_split_bounds(3, SplitConfig())
    shuffled = transactions.sample(frac=1.0, random_state=1)
    train, val, test = splits.split_by_time(shuffled)
    assert len(train) + len(val) + len(test) == len(transactions)
    assert train["Time"].max() <= val["Time"].min() and val["Time"].max() <= test["Time"].min()
    with pytest.raises(ValueError, match="Missing column"):
        splits.split_by_time(transactions.drop(columns=["Time"]))


def test_amount_bands_account_for_every_row(transactions) -> None:
    bands = eda.amount_bands(transactions)
    assert bands["transactions"].sum() == len(transactions) and bands["frauds"].sum() == 20
    assert bands["share_of_frauds"].sum() == pytest.approx(1.0, abs=0.01)


def test_hourly_pattern_has_24_hours(transactions) -> None:
    hours = eda.hourly_pattern(transactions)
    assert list(hours["hour"]) == list(range(24)) and hours["transactions"].sum() == len(transactions)


def test_feature_separation_finds_the_planted_signal(transactions) -> None:
    table = eda.feature_separation(transactions)
    assert set(table["feature"][:3]) == {"V1", "V2", "V3"}              # the fixture shifts these three for fraud
    assert (table["cohens_d"].head(3) > 1.5).all() and table["ks"].between(0, 1).all()
    assert len(table) == 30


def test_feature_correlation_small_for_independent_columns(transactions) -> None:
    assert eda.feature_correlation(transactions)["max_abs_between_pca_features"] < 0.2


def test_split_balance_and_drift(transactions) -> None:
    balance = eda.split_balance(transactions)
    assert balance["rows"].sum() == len(transactions) and balance["frauds"].sum() == 20
    drift = eda.split_drift(transactions)
    assert len(drift) == 29 and "Time" not in set(drift["feature"])
    shifted = transactions.copy()
    shifted.loc[shifted["Time"] > shifted["Time"].quantile(0.8), "V10"] += 5     # newest rows drift on one feature
    assert eda.split_drift(shifted).iloc[0]["feature"] == "V10"


def test_build_summary_is_json_ready(transactions) -> None:
    summary = eda.build_summary(transactions)
    assert summary["frauds"] == 20 and summary["fraud_rate"] == 0.01 and len(summary["hourly"]) == 24
    json.dumps(summary)                                                    # must not raise


def test_missing_columns_raise(transactions) -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        eda.build_summary(transactions.drop(columns=["V9"]))


def test_plots_return_figures(transactions) -> None:
    from matplotlib.figure import Figure
    bands, hours, sep = eda.amount_bands(transactions), eda.hourly_pattern(transactions), eda.feature_separation(transactions)
    figures = [plots.plot_class_balance(2000, 20), plots.plot_amounts(transactions, bands), plots.plot_time_patterns(transactions, hours),
               plots.plot_feature_separation(sep), plots.plot_top_distributions(transactions, ["V1", "V2", "V3", "V4"]),
               plots.plot_drift(eda.split_drift(transactions)), plots.plot_correlation(transactions, ["V1", "V2"])]
    assert all(isinstance(f, Figure) for f in figures)
    for f in figures:
        plots.plt.close(f)


def test_run_writes_all_outputs(tmp_path: Path, transactions) -> None:
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=tmp_path))
    cfg.paths.raw_path.parent.mkdir(parents=True)
    transactions.to_csv(cfg.paths.raw_path, index=False)
    explore.run(cfg)
    reports = tmp_path / "reports"
    assert (reports / "eda_summary.json").is_file() and (reports / "feature_separation.csv").is_file()
    assert len(list((reports / "figures").glob("*.png"))) == 7
    with pytest.raises(FileNotFoundError):
        explore.run(replace(cfg, paths=replace(cfg.paths, root=tmp_path / "nowhere")))
