"""Tests for fraud.models, the Step 5 metrics and fraud.train_supervised (XGBoost / LightGBM tests run only where installed)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud import metrics, models, preprocess, train_supervised
from fraud.config import ImbalanceConfig, ProjectConfig, SupervisedConfig
from tests.conftest import make_transactions

SMALL = SupervisedConfig(rf_trees=20, rf_max_depth=6, boost_trees=30, bootstrap_resamples=20)
ALL = ("logistic_regression", "random_forest", "xgboost", "lightgbm")


def _xy(n: int = 600, frauds: int = 30):
    rng = np.random.default_rng(42)
    y = np.zeros(n, dtype=int)
    y[:frauds] = 1
    X = rng.normal(size=(n, 5))
    X[y == 1] += 2.0
    return X, y


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("variant", ["baseline", "smote"])
def test_every_available_model_fits(name, variant) -> None:
    if not models.is_available(name):
        pytest.skip(f"{name} not installed")
    X, y = _xy()
    model = models.fit_model(name, variant, X, y, SMALL, ImbalanceConfig(sampling_ratio=0.2))
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (600,) and ((p >= 0) & (p <= 1)).all() and p[y == 1].mean() > p[y == 0].mean()


def test_model_errors() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        models.make_model("svm")
    with pytest.raises(ValueError, match="unknown variant"):
        models.fit_model("logistic_regression", "adasyn", *_xy())
    with pytest.raises(ValueError):
        SupervisedConfig(models=("svm",))
    with pytest.raises(ValueError):
        SupervisedConfig(variants=())
    assert models.is_available("logistic_regression") and models.is_available("random_forest")


def test_feature_importance_sums_to_one_and_ranks() -> None:
    X, y = _xy()
    names = tuple(f"f{i}" for i in range(5))
    for name in ("logistic_regression", "random_forest"):
        imp = models.feature_importance(models.fit_model(name, "baseline", X, y, SMALL), names)
        assert sum(imp.values()) == pytest.approx(1.0, abs=1e-3) and list(imp.values()) == sorted(imp.values(), reverse=True)
    with pytest.raises(ValueError):
        models.feature_importance(models.fit_model("logistic_regression", "baseline", X, y, SMALL), names[:3])


def test_best_f2_threshold_and_apply_threshold() -> None:
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.1, 0.0, 0.3, 0.2, 0.9, 0.8, 0.95, 0.6])
    t = metrics.best_f2_threshold(y, s)
    assert t == pytest.approx(0.8)                                     # catches all three frauds with no false alarm
    got = metrics.apply_threshold(y, s, t)
    assert (got["tp"], got["fp"], got["fn"]) == (3, 0, 0) and got["f2"] == pytest.approx(1.0)
    assert metrics.apply_threshold(y, s, 0.5)["fp"] == 1


def test_bootstrap_interval_contains_the_estimate_and_widens_with_few_frauds() -> None:
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(20), np.zeros(2000)].astype(int)
    s = np.r_[rng.normal(2, 1, 20), rng.normal(0, 1, 2000)]
    ci = metrics.bootstrap_pr_auc(y, s, 100)
    assert ci["low"] < metrics.pr_auc(y, s) < ci["high"] and ci["std"] > 0
    assert metrics.bootstrap_pr_auc(y, s, 100) == ci                  # seeded: repeatable
    y2 = np.r_[np.ones(200), np.zeros(20000)].astype(int)
    s2 = np.r_[rng.normal(2, 1, 200), rng.normal(0, 1, 20000)]
    big = metrics.bootstrap_pr_auc(y2, s2, 100)
    assert (big["high"] - big["low"]) < (ci["high"] - ci["low"])      # more frauds, tighter interval
    with pytest.raises(ValueError):
        metrics.bootstrap_pr_auc(y, s, 5)


def _cfg(tmp_path: Path, **sup) -> ProjectConfig:
    base = ProjectConfig()
    return replace(base, paths=replace(base.paths, root=tmp_path), supervised=replace(SMALL, **sup),
                   imbalance=ImbalanceConfig(sampling_ratio=0.2))


def test_cross_validate_model_returns_oof_for_every_row() -> None:
    X, y = _xy()
    out = train_supervised.cross_validate_model("logistic_regression", "baseline", X, y, 3, ProjectConfig())
    assert out["oof"].shape == (600,) and (out["oof"] > 0).all() and len(out["folds"]) == 3 and out["mean"] > 0.5


def test_run_end_to_end(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, models=("logistic_regression", "random_forest"))
    cfg.paths.raw_path.parent.mkdir(parents=True)
    make_transactions(4000, 120).to_csv(cfg.paths.raw_path, index=False)
    preprocess.run(cfg, ("time",))
    summary = train_supervised.run(cfg, method="time")
    assert len(summary["models"]) == 4 and summary["skipped_models"] == [] and summary["test_frauds"] > 5
    reports = tmp_path / "reports"
    table = pd.read_csv(reports / "model_comparison.csv")
    assert set(table["model"]) == {"logistic_regression", "random_forest"} and table["test_pr_auc"].between(0, 1).all()
    assert (table["test_pr_auc_ci_low"] <= table["test_pr_auc"]).all() and (table["test_pr_auc"] <= table["test_pr_auc_ci_high"]).all()
    assert (tmp_path / "models" / "supervised_best.joblib").is_file() and (tmp_path / "models" / "supervised_best.json").is_file()
    assert len(list((reports / "figures").glob("model_*.png"))) == 2 and (reports / "figures" / "feature_importance.png").is_file()


def test_run_skips_missing_packages_and_rejects_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, models=("logistic_regression", "xgboost"), variants=("baseline",))
    cfg.paths.raw_path.parent.mkdir(parents=True)
    make_transactions(4000, 120).to_csv(cfg.paths.raw_path, index=False)
    preprocess.run(cfg, ("time",))
    monkeypatch.setattr(models, "is_available", lambda name: name != "xgboost")
    summary = train_supervised.run(cfg, method="time", sample=True)
    assert summary["skipped_models"] == ["xgboost"] and len(summary["models"]) == 1
    with pytest.raises(ValueError, match="none of the requested"):
        train_supervised.run(cfg, method="time", only=["xgboost"])
    with pytest.raises(FileNotFoundError):
        train_supervised.run(replace(cfg, paths=replace(cfg.paths, root=tmp_path / "nowhere")))
