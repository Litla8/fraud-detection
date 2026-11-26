"""Tests for fraud.data and fraud.define (a small synthetic file stands in for Kaggle's)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fraud import data, define
from fraud.config import ProjectConfig
from tests.conftest import make_transactions


def _cfg(tmp_path: Path) -> ProjectConfig:
    base = ProjectConfig()
    return replace(base, paths=replace(base.paths, root=tmp_path))


def test_load_raw_roundtrip(tmp_path: Path, transactions) -> None:
    path = tmp_path / "data" / "raw" / "creditcard.csv"
    path.parent.mkdir(parents=True)
    transactions.to_csv(path, index=False)
    frame = data.load_raw(path)
    assert frame.shape == transactions.shape and str(frame["Class"].dtype) == "int8" and str(frame["V1"].dtype) == "float32"
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        data.load_raw(tmp_path / "missing.csv")


@pytest.mark.parametrize("change,message", [
    (lambda f: f.drop(columns=["V5"]), "Missing columns"),
    (lambda f: f.assign(Class=f["Class"].replace({1: 2})), "only 0 and 1"),
    (lambda f: f.assign(Amount=-f["Amount"]), "negative"),
    (lambda f: f.assign(Class=0), "both genuine"),
    (lambda f: f.assign(V2=f["V2"].where(f.index > 0)), "missing values"),
])
def test_validate_columns_errors(transactions, change, message) -> None:
    with pytest.raises(ValueError, match=message):
        data.validate_columns(change(transactions))


def test_profile(transactions) -> None:
    p = data.profile(transactions)
    assert p["rows"] == 2000 and p["frauds"] == 20 and p["fraud_rate"] == 0.01 and p["imbalance_ratio"] == 99.0
    assert p["time_sorted"] and p["amount"]["fraud"]["mean"] > p["amount"]["genuine"]["mean"]


def test_define_run_writes_report(tmp_path: Path, transactions) -> None:
    cfg = _cfg(tmp_path)
    cfg.paths.raw_path.parent.mkdir(parents=True)
    transactions.to_csv(cfg.paths.raw_path, index=False)
    report = define.run(cfg)
    saved = json.loads(cfg.paths.problem_report_path.read_text())
    assert saved == json.loads(json.dumps(report))
    assert saved["why_accuracy_fails"]["always_genuine_accuracy"] == 0.99
    assert saved["baselines"]["approve_everything"] > 0
    assert 0 < saved["cost_matrix"]["break_even_precision"] < 1


def test_define_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        define.run(_cfg(tmp_path))
