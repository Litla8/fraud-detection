"""Tests for fraud.imbalance and fraud.run_imbalance (synthetic data; the probe model is real but tiny)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud import imbalance, preprocess, run_imbalance
from fraud.config import ImbalanceConfig, ProjectConfig
from tests.conftest import make_transactions

SMALL = ImbalanceConfig(max_iter=20, max_leaf_nodes=7, sampling_ratio=0.2, focal_inner_folds=2)


def _xy(n: int = 600, frauds: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:frauds] = 1
    X = rng.normal(size=(n, 4))
    X[y == 1] += 2.5
    return X, y


def test_undersample_keeps_all_frauds_and_ratio() -> None:
    X, y = _xy()
    Xs, ys = imbalance.undersample(X, y, 0.2)
    assert ys.sum() == 30 and (ys == 0).sum() == 150 and Xs.shape == (180, 4)
    Xa, ya = imbalance.undersample(X, y, 1.0)
    assert (ya == 0).sum() == 30
    assert np.array_equal(imbalance.undersample(X, y, 0.2)[0], Xs)                  # deterministic


def test_smote_adds_synthetic_frauds_inside_the_fraud_cloud() -> None:
    X, y = _xy()
    Xs, ys = imbalance.smote(X, y, 0.2, k=5)
    assert len(ys) == len(y) + (114 - 30) and ys.sum() == 114 and (ys == 0).sum() == 570
    assert np.array_equal(Xs[:600], X)                                              # originals untouched, new rows appended
    fraud = X[y == 1]
    new = Xs[600:]
    assert (new >= fraud.min(axis=0) - 1e-9).all() and (new <= fraud.max(axis=0) + 1e-9).all()   # interpolation never leaves the box


def test_adasyn_focuses_on_hard_frauds() -> None:
    X, y = _xy()
    X[:5] -= 2.5                                                                     # five frauds now sit inside the genuine cloud
    Xs, ys = imbalance.adasyn(X, y, 0.2, k=5)
    assert ys.sum() == 114 and len(Xs) == 600 + 84
    hard_centre = X[:5].mean(axis=0)
    easy_centre = X[5:30].mean(axis=0)
    new = Xs[600:]
    near_hard = (np.linalg.norm(new - hard_centre, axis=1) < np.linalg.norm(new - easy_centre, axis=1)).mean()
    assert near_hard > 5 / 30                                                        # more than their one-in-six share of the frauds


def test_resamplers_do_nothing_when_already_balanced_enough() -> None:
    X, y = _xy()
    for fn in (imbalance.smote, imbalance.adasyn):
        Xs, ys = fn(X, y, 0.01)
        assert len(ys) == len(y)


def test_validation_errors() -> None:
    X, y = _xy()
    with pytest.raises(ValueError, match="at least 2 frauds"):
        imbalance.smote(X, np.zeros_like(y), 0.1)
    with pytest.raises(ValueError, match="only 0 and 1"):
        imbalance.undersample(X, y + 1, 0.1)
    with pytest.raises(ValueError, match="one label per row"):
        imbalance.undersample(X, y[:10], 0.1)
    with pytest.raises(ValueError, match="unknown strategy"):
        imbalance.Strategy("x", "magic", "")
    with pytest.raises(ValueError):
        ImbalanceConfig(sampling_ratio=0)


def test_focal_weights_rank_hard_rows_higher() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.95, 0.05, 0.02, 0.90])             # row 0 and 2 easy, rows 1 and 3 badly wrong
    w = imbalance.focal_style_weights(p, y, gamma=2.0, floor=0.01)
    assert w[1] > w[0] and w[3] > w[2] and w.min() >= 0.01 and w.max() <= 1.01
    with pytest.raises(ValueError):
        imbalance.focal_style_weights(np.array([1.5]), np.array([1]))


@pytest.mark.parametrize("strategy", imbalance.default_strategies(SMALL), ids=lambda s: s.name)
def test_every_strategy_fits_and_predicts(strategy) -> None:
    X, y = _xy()
    model = imbalance.fit_strategy(strategy, X, y, SMALL)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (600,) and ((proba >= 0) & (proba <= 1)).all()
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_cross_validation_returns_oof_scores_for_every_row() -> None:
    X, y = _xy()
    out = imbalance.cross_validate_strategy(imbalance.default_strategies(SMALL)[3], X, y, 3, SMALL)
    assert len(out["fold_pr_auc"]) == 3 and out["oof_scores"].shape == (600,) and (out["oof_scores"] > 0).all()
    assert out["pr_auc_mean"] > 0.5


def test_resampling_never_sees_validation_rows() -> None:
    """The model must be fitted on fold-train rows only: a spy factory records how many rows each fit receives."""
    X, y = _xy()
    seen = []

    class Spy:
        def fit(self, A, b, sample_weight=None):
            seen.append(len(A))
            self.p = b.mean()
            return self

        def predict_proba(self, A):
            return np.column_stack([1 - np.full(len(A), self.p), np.full(len(A), self.p)])

    imbalance.cross_validate_strategy(imbalance.Strategy("baseline", "none", ""), X, y, 3, SMALL, factory=lambda balanced: Spy())
    assert seen == [400, 400, 400]                                                  # two thirds of 600 each time


def test_pick_winner_and_ties() -> None:
    rows = [{"strategy": "a", "cv_pr_auc_mean": 0.80, "cv_pr_auc_std": 0.05}, {"strategy": "b", "cv_pr_auc_mean": 0.78, "cv_pr_auc_std": 0.02},
            {"strategy": "c", "cv_pr_auc_mean": 0.60, "cv_pr_auc_std": 0.02}]
    verdict = imbalance.pick_winner(rows)
    assert verdict["winner"] == "a" and verdict["tied_with_winner"] == ["b"]
    assert imbalance.pick_winner(rows[::2])["note"] == "Clear winner."
    with pytest.raises(ValueError):
        imbalance.pick_winner([])


def test_run_writes_outputs(tmp_path: Path) -> None:
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=tmp_path), imbalance=SMALL)
    cfg.paths.raw_path.parent.mkdir(parents=True)
    make_transactions(3000, 90).to_csv(cfg.paths.raw_path, index=False)
    preprocess.run(cfg, ("time",))
    summary = run_imbalance.run(cfg, method="time", strategies=imbalance.default_strategies(SMALL)[:3])
    assert summary["winner"] in {"baseline", "class_weight", "undersample"} and len(summary["strategies"]) == 3
    reports = tmp_path / "reports"
    assert (reports / "imbalance_comparison.csv").is_file() and (reports / "figures" / "imbalance_pr_curves.png").is_file()
    assert pd.read_csv(reports / "imbalance_comparison.csv")["strategy"].tolist() == ["baseline", "class_weight", "undersample"]
    with pytest.raises(FileNotFoundError):
        run_imbalance.run(replace(cfg, paths=replace(cfg.paths, root=tmp_path / "nowhere")))
