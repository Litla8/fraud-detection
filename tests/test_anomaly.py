"""Tests for fraud.anomaly (detectors, percentiles, blend, rescue) and fraud.train_anomaly."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud import anomaly, preprocess, train_anomaly, train_supervised
from fraud.config import AnomalyConfig, ImbalanceConfig, ProjectConfig, SupervisedConfig
from tests.conftest import make_transactions

FAST = AnomalyConfig(iforest_trees=50, ocsvm_rows=400, ae_epochs=15, ae_rows=2000, blend_weights=(0.0, 0.2, 0.5), sample_normal=1500)


def _data(n: int = 1200, outliers: int = 30):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, 6))
    y = np.zeros(n, dtype=int)
    y[:outliers] = 1
    X[:outliers] += 4.0
    return X, y


@pytest.mark.parametrize("name", ["isolation_forest", "one_class_svm", "autoencoder"])
def test_detectors_score_outliers_higher_and_are_repeatable(name: str) -> None:
    X, y = _data()
    det = anomaly.fit_detector(name, X[y == 0], FAST)
    s = det.score(X)
    assert s.shape == (len(X),) and np.isfinite(s).all() and s[y == 1].mean() > s[y == 0].mean()
    again = anomaly.fit_detector(name, X[y == 0], FAST).score(X)
    assert np.allclose(s, again)                                       # seeded: identical scores


def test_detector_errors() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        anomaly.make_detector("lof")
    with pytest.raises(ValueError, match="at least 20"):
        anomaly.fit_detector("isolation_forest", np.zeros((5, 3)), FAST)


def test_clip_and_subsample() -> None:
    X = np.array([[100.0, -100.0], [1.0, 2.0]])
    assert anomaly.clip_features(X, 10).tolist() == [[10.0, -10.0], [1.0, 2.0]]
    with pytest.raises(ValueError):
        anomaly.clip_features(np.array([[np.nan, 1.0]]), 10)
    with pytest.raises(ValueError):
        anomaly.clip_features(np.zeros(3), 10)
    big = np.arange(1000).reshape(-1, 1).astype(float)
    sub = anomaly.subsample_rows(big, 100)
    assert len(sub) == 100 and (np.diff(sub[:, 0]) > 0).all() and np.array_equal(sub, anomaly.subsample_rows(big, 100))
    assert anomaly.subsample_rows(big, 5000) is big
    with pytest.raises(ValueError):
        anomaly.subsample_rows(big, 0)


def test_percentiles() -> None:
    ref = anomaly.reference_distribution([5, 1, 3, 2, 4, 6, 8, 7, 10, 9])
    assert ref.tolist() == list(range(1, 11))
    got = anomaly.to_percentile(ref, [0, 1, 5.5, 10, 99])
    assert got.tolist() == [0.0, 0.1, 0.5, 1.0, 1.0]
    with pytest.raises(ValueError):
        anomaly.reference_distribution([1.0])
    with pytest.raises(ValueError):
        anomaly.reference_distribution([1.0, np.nan])


def test_blend() -> None:
    s, a = np.array([0.0, 1.0, 0.5]), np.array([1.0, 0.0, 0.5])
    assert np.array_equal(anomaly.blend_scores(s, a, 0.0), s)
    assert anomaly.blend_scores(s, a, 0.5).tolist() == [0.5, 0.5, 0.5]
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            anomaly.blend_scores(s, a, bad)
    with pytest.raises(ValueError):
        anomaly.blend_scores(s, a[:2], 0.1)


def test_rescue_analysis_counts() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    sup = np.array([0.9, 0.1, 0.2, 0.1, 0.1, 0.9, 0.1, 0.1])
    pct = np.array([0.1, 0.995, 0.5, 0.995, 0.2, 0.999, 0.1, 0.1])
    got = anomaly.rescue_analysis(y, sup, 0.5, pct, 0.99)
    assert got["missed_by_supervised"] == 2 and got["rescued"] == 1 and got["extra_false_alarms"] == 1      # row 5 is already blocked
    assert got["frauds_caught_by_supervised"] == 1 and got["rescued_share_of_missed"] == 0.5 and got["precision_of_extra_alerts"] == 0.5
    none = anomaly.rescue_analysis(np.array([1, 0]), np.array([0.9, 0.1]), 0.5, np.array([0.1, 0.1]), 0.99)
    assert none["missed_by_supervised"] == 0 and none["rescued_share_of_missed"] == 0.0 and none["precision_of_extra_alerts"] == 0.0
    with pytest.raises(ValueError):
        anomaly.rescue_analysis(y, sup[:3], 0.5, pct, 0.99)


def test_config_validation() -> None:
    for bad in (dict(models=()), dict(models=("lof",)), dict(clip=0), dict(blend_weights=(0.1, 0.2)), dict(blend_weights=(0.0, 1.0)),
                dict(rescue_quantile=1.0), dict(ocsvm_nu=0.0), dict(ae_hidden=())):
        with pytest.raises(ValueError):
            AnomalyConfig(**bad)


def test_choose_blend_needs_a_real_gain() -> None:
    grid = pd.DataFrame({"detector": ["a", "a", "b", "b"], "weight": [0.0, 0.2, 0.0, 0.2], "val_pr_auc": [0.70, 0.74, 0.70, 0.72]})
    got = train_anomaly.choose_blend(grid, 0.01)
    assert got == {"detector": "a", "weight": 0.2, "best_weight_seen": 0.2, "val_gain": 0.04, "adopted": True}
    flat = train_anomaly.choose_blend(grid.assign(val_pr_auc=[0.70, 0.704, 0.70, 0.70]), 0.01)
    assert flat["adopted"] is False and flat["weight"] == 0.0 and flat["best_weight_seen"] == 0.2
    with pytest.raises(ValueError):
        train_anomaly.choose_blend(grid[grid["weight"] > 0], 0.01)


def _prepared(tmp_path: Path, **anomaly_cfg) -> ProjectConfig:
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=tmp_path), anomaly=replace(FAST, **anomaly_cfg),
                  supervised=SupervisedConfig(models=("logistic_regression", "random_forest"), rf_trees=20, rf_max_depth=6, bootstrap_resamples=20),
                  imbalance=ImbalanceConfig(sampling_ratio=0.2))
    cfg.paths.raw_path.parent.mkdir(parents=True)
    make_transactions(4000, 120).to_csv(cfg.paths.raw_path, index=False)
    preprocess.run(cfg, ("time",))
    train_supervised.run(cfg, method="time")
    return cfg


def test_run_end_to_end(tmp_path: Path) -> None:
    cfg = _prepared(tmp_path)
    summary = train_anomaly.run(cfg, method="time")
    scorers = [r["scorer"] for r in summary["scorers"]]
    assert len(scorers) == 5 and scorers[0].endswith("(supervised)") and scorers[-1].startswith("hybrid")
    table = pd.read_csv(tmp_path / "reports" / "anomaly_comparison.csv")
    assert table["test_pr_auc"].between(0, 1).all() and (table["test_pr_auc_ci_low"] <= table["test_pr_auc_ci_high"]).all()
    assert set(summary["rescue"]) == set(FAST.models) and len(summary["blend_grid"]) == 3 * len(FAST.blend_weights)
    meta = json.loads((tmp_path / "models" / "hybrid.json").read_text())
    assert meta["weight"] in FAST.blend_weights and meta["features"] == preprocess.feature_names()
    saved = joblib.load(tmp_path / "models" / "anomaly_detectors.joblib")
    assert set(saved) == set(FAST.models) and saved["autoencoder"]["reference"].size > 10
    figs = tmp_path / "reports" / "figures"
    assert all((figs / f).is_file() for f in ("anomaly_separation.png", "blend_weights.png", "hybrid_pr_curves.png"))


def test_sample_run_saves_no_models_and_single_detector(tmp_path: Path) -> None:
    cfg = _prepared(tmp_path)
    (tmp_path / "models" / "hybrid.json").unlink(missing_ok=True)
    summary = train_anomaly.run(cfg, method="time", sample=True, only=["isolation_forest"])
    assert summary["sample"] and len(summary["scorers"]) == 3 and summary["genuine_rows_used"] <= FAST.sample_normal
    assert not (tmp_path / "models" / "hybrid.json").exists() and (tmp_path / "reports" / "anomaly_comparison_sample.json").is_file()


def test_missing_inputs_fail_clearly(tmp_path: Path) -> None:
    cfg = _prepared(tmp_path)
    with pytest.raises(ValueError, match="no detector"):
        train_anomaly.run(cfg, method="time", only=["nothing"])
    (tmp_path / "models" / "supervised_best.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="Step 5"):
        train_anomaly.run(cfg, method="time")
