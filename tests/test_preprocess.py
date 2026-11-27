"""Tests for fraud.preprocess and the stratified split (synthetic data)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud import preprocess, splits
from fraud.config import PreprocessConfig, ProjectConfig, SplitConfig
from tests.conftest import make_transactions


def _cfg(tmp_path: Path, **prep) -> ProjectConfig:
    base = ProjectConfig()
    return replace(base, paths=replace(base.paths, root=tmp_path), prep=replace(base.prep, **prep))


def test_feature_names() -> None:
    assert preprocess.feature_names()[-2:] == ["Amount_scaled", "Time_scaled"] and len(preprocess.feature_names()) == 30
    cfg = PreprocessConfig(use_time=False, hour_features=True)
    assert preprocess.feature_names(prep=cfg)[-3:] == ["Amount_scaled", "hour_sin", "hour_cos"]


def test_drop_duplicate_rows(transactions) -> None:
    doubled = pd.concat([transactions, transactions.iloc[:5], transactions[transactions["Class"] == 1].iloc[:2]], ignore_index=True)
    clean, stats = preprocess.drop_duplicate_rows(doubled)
    assert len(clean) == len(transactions) and stats == {"rows_before": len(doubled), "rows_dropped": 7, "frauds_dropped": 2 + int(transactions.iloc[:5]["Class"].sum())}
    kept, none = preprocess.drop_duplicate_rows(doubled, enabled=False)
    assert len(kept) == len(doubled) and none["rows_dropped"] == 0


def test_stratified_split_keeps_fraud_rate(transactions) -> None:
    train, val, test = splits.split_stratified(transactions)
    assert len(train) + len(val) + len(test) == len(transactions)
    assert len(test) == 400 and len(val) == 300 and {len(set(p.index) & set(q.index)) for p, q in ((train, val), (train, test), (val, test))} == {0}
    assert [int(p["Class"].sum()) for p in (train, val, test)] == [13, 3, 4]            # 20 frauds spread by rule
    with pytest.raises(ValueError):
        splits.split_stratified(transactions.assign(Class=0))


def test_scaler_is_fitted_on_train_only(transactions) -> None:
    cfg = ProjectConfig()
    train, _, test = splits.split_by_time(transactions)
    scaler = preprocess.fit_scaler(train)
    assert scaler.center_[0] == pytest.approx(train["Amount"].median())
    out = preprocess.transform(train, scaler, cfg)
    assert abs(out["Amount_scaled"].median()) < 1e-9
    expected = (test["Amount"] - scaler.center_[0]) / scaler.scale_[0]                  # test is scaled with TRAIN statistics
    assert preprocess.transform(test, scaler, cfg)["Amount_scaled"].to_numpy() == pytest.approx(expected.to_numpy())
    with pytest.raises(ValueError):
        preprocess.fit_scaler(train.iloc[0:0])
    with pytest.raises(ValueError, match="Missing columns"):
        preprocess.fit_scaler(train.drop(columns=["Amount"]))


def test_transform_keeps_raw_amount_and_does_not_mutate(transactions) -> None:
    cfg = ProjectConfig()
    before = transactions.copy()
    scaler = preprocess.fit_scaler(transactions)
    out = preprocess.transform(transactions, scaler, cfg)
    pd.testing.assert_frame_equal(transactions, before)
    assert list(out.columns) == [*preprocess.feature_names(), "Class", "Amount_raw", "Time_raw", "row_id"]
    assert out["Amount_raw"].to_numpy() == pytest.approx(transactions["Amount"].to_numpy())
    no_time = preprocess.transform(transactions, scaler, replace(cfg, prep=replace(cfg.prep, use_time=False, hour_features=True)))
    assert "Time_scaled" not in no_time.columns and {"hour_sin", "hour_cos"} <= set(no_time.columns)


@pytest.mark.parametrize("method", preprocess.METHODS)
def test_build_split_passes_checks(transactions, method) -> None:
    cfg = ProjectConfig()
    parts, _ = preprocess.build_split(transactions, method, cfg)
    assert preprocess.check_split(parts, method, cfg)["disjoint"]
    assert sum(len(p) for p in parts.values()) == len(transactions)
    if method == "time":
        assert parts["train"]["Time_raw"].max() <= parts["test"]["Time_raw"].min()


def test_check_split_catches_problems(transactions) -> None:
    cfg = ProjectConfig()
    parts, _ = preprocess.build_split(transactions, "time", cfg)
    with pytest.raises(ValueError, match="share rows"):
        preprocess.check_split({**parts, "test": parts["train"]}, "time", cfg)
    no_fraud = {**parts, "test": parts["test"].assign(Class=0)}
    with pytest.raises(ValueError, match="no fraud"):
        preprocess.check_split(no_fraud, "time", cfg)
    with pytest.raises(ValueError, match="method"):
        preprocess.build_split(transactions, "random", cfg)


def test_run_roundtrip_and_load_split(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.paths.raw_path.parent.mkdir(parents=True)
    raw = make_transactions()
    pd.concat([raw, raw.iloc[:10]], ignore_index=True).to_csv(cfg.paths.raw_path, index=False)    # 10 duplicate rows on purpose
    report = preprocess.run(cfg)
    assert report["duplicates"]["rows_dropped"] == 10 and set(report["splits"]) == {"time", "stratified"}
    saved = json.loads((tmp_path / "reports" / "preprocessing_report.json").read_text())
    assert saved["features"] == preprocess.feature_names()
    parts = preprocess.load_split(cfg, "time")
    assert sum(len(p) for p in parts.values()) == len(raw)
    assert cfg.paths.scaler_path("time").is_file() and cfg.paths.scaler_path("stratified").is_file()


def test_load_split_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="preprocess"):
        preprocess.load_split(_cfg(tmp_path))
