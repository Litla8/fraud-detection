"""Step 4 building blocks: resampling and reweighting strategies for 0.2% fraud, plus a fair way to compare them.

Resampling is implemented here (numpy and scikit-learn only) instead of with imbalanced-learn so that every line is
unit-tested. All of it happens on *training* rows inside each cross-validation fold; validation and test rows are
never resampled.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from fraud.config import ImbalanceConfig, SEED
from fraud.metrics import pr_auc

LOGGER = logging.getLogger(__name__)
ModelFactory = Callable[[bool], Any]            # factory(balanced) -> an unfitted classifier with fit / predict_proba


@dataclass(frozen=True)
class Strategy:
    """One way of dealing with imbalance. ``kind`` is none, class_weight, undersample, smote, adasyn or focal."""

    name: str
    kind: str
    description: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown strategy kind {self.kind!r}; choose from {KINDS}")


KINDS = ("none", "class_weight", "undersample", "smote", "adasyn", "focal")


def default_strategies(cfg: ImbalanceConfig = ImbalanceConfig()) -> Tuple[Strategy, ...]:
    """The six strategies compared in Step 4."""
    r = f"{cfg.sampling_ratio:g}"
    return (Strategy("baseline", "none", "no handling"),
            Strategy("class_weight", "class_weight", "balanced class weights"),
            Strategy("undersample", "undersample", f"keep all frauds, sample genuine to a {r} fraud:genuine ratio"),
            Strategy("smote", "smote", f"synthetic frauds by interpolation up to a {r} ratio"),
            Strategy("adasyn", "adasyn", f"synthetic frauds near hard cases up to a {r} ratio"),
            Strategy("focal_style", "focal", f"hard-example weights (gamma={cfg.focal_gamma:g}) from out-of-fold predictions"))


def _check_xy(X: np.ndarray, y: np.ndarray, minimum_minority: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("X must be 2-D and y must have one label per row")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    if (y == 1).sum() < minimum_minority or (y == 0).sum() == 0:
        raise ValueError(f"need at least {minimum_minority} frauds and one genuine row")
    return X, y


def undersample(X: np.ndarray, y: np.ndarray, ratio: float, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Keep every fraud and a random sample of genuine rows so that ``frauds / genuine = ratio`` (or all genuine if fewer)."""
    X, y = _check_xy(X, y, 1)
    rng = np.random.default_rng(seed)
    fraud, genuine = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = rng.choice(genuine, size=min(len(genuine), int(round(len(fraud) / ratio))), replace=False)
    idx = np.sort(np.concatenate([fraud, keep]))
    return X[idx], y[idx]


def _synthetic(minority: np.ndarray, base: np.ndarray, neighbours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """New points on the line between each chosen minority row and one of its minority neighbours (vectorised)."""
    pick = neighbours[base, rng.integers(0, neighbours.shape[1], len(base))]
    return minority[base] + rng.random((len(base), 1)) * (minority[pick] - minority[base])


def _minority_neighbours(minority: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest *other* minority rows for every minority row."""
    k = min(k, len(minority) - 1)
    return NearestNeighbors(n_neighbors=k + 1).fit(minority).kneighbors(minority, return_distance=False)[:, 1:]


def _n_new(y: np.ndarray, ratio: float) -> int:
    return max(int(round(ratio * (y == 0).sum())) - int((y == 1).sum()), 0)


def smote(X: np.ndarray, y: np.ndarray, ratio: float, k: int = 5, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """SMOTE: add synthetic frauds, each between a real fraud and one of its k nearest frauds, until ``frauds / genuine = ratio``."""
    X, y = _check_xy(X, y, 2)
    n_new = _n_new(y, ratio)
    if n_new == 0:
        return X.copy(), y.copy()
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    new = _synthetic(minority, rng.integers(0, len(minority), n_new), _minority_neighbours(minority, k), rng)
    return np.vstack([X, new]), np.concatenate([y, np.ones(n_new, dtype=int)])


def adasyn(X: np.ndarray, y: np.ndarray, ratio: float, k: int = 5, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """ADASYN: like SMOTE, but frauds surrounded by genuine rows (hard cases) get proportionally more synthetic neighbours.

    Falls back to uniform SMOTE sampling when no fraud has a genuine neighbour.
    """
    X, y = _check_xy(X, y, 2)
    n_new = _n_new(y, ratio)
    if n_new == 0:
        return X.copy(), y.copy()
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    k_all = min(k, len(X) - 1)
    around = NearestNeighbors(n_neighbors=k_all + 1).fit(X).kneighbors(minority, return_distance=False)[:, 1:]
    difficulty = (y[around] == 0).mean(axis=1)                          # share of genuine rows among the k nearest rows
    weights = difficulty / difficulty.sum() if difficulty.sum() > 0 else np.full(len(minority), 1 / len(minority))
    base = np.repeat(np.arange(len(minority)), rng.multinomial(n_new, weights))
    new = _synthetic(minority, base, _minority_neighbours(minority, k), rng)
    return np.vstack([X, new]), np.concatenate([y, np.ones(len(base), dtype=int)])


def focal_style_weights(p_fraud: np.ndarray, y: np.ndarray, gamma: float = 2.0, floor: float = 0.01) -> np.ndarray:
    """Sample weights ``floor + (1 - p_true) ** gamma``, where ``p_true`` is the probability given to the *true* class.

    Rows the model already gets right (p_true near 1) fade towards ``floor``; rows it gets wrong keep weight near 1.
    This approximates the focal-loss idea with ordinary sample weights, using out-of-fold probabilities so a row is never
    judged by a model that has seen it.
    """
    p = np.asarray(p_fraud, dtype=float)
    labels = np.asarray(y).astype(int)
    if p.shape != labels.shape:
        raise ValueError("p_fraud and y must have the same shape")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p_fraud must be probabilities in [0, 1]")
    p_true = np.where(labels == 1, p, 1 - p)
    return floor + (1 - p_true) ** gamma


def make_probe_model(cfg: ImbalanceConfig = ImbalanceConfig(), balanced: bool = False, seed: int = SEED) -> HistGradientBoostingClassifier:
    """Histogram gradient boosting with fixed iterations (no early stopping: it would peek at too few frauds) and L2 regularisation."""
    return HistGradientBoostingClassifier(max_iter=cfg.max_iter, learning_rate=cfg.learning_rate, max_leaf_nodes=cfg.max_leaf_nodes,
                                          l2_regularization=cfg.l2_regularization, early_stopping=False,
                                          class_weight="balanced" if balanced else None, random_state=seed)


def fit_strategy(strategy: Strategy, X: np.ndarray, y: np.ndarray, cfg: ImbalanceConfig = ImbalanceConfig(),
                 factory: Optional[ModelFactory] = None, seed: int = SEED) -> Any:
    """Fit one model on ``X, y`` using ``strategy`` (resampling or weighting touches only these rows)."""
    X, y = _check_xy(X, y, 2)
    make = factory or (lambda balanced: make_probe_model(cfg, balanced, seed))
    if strategy.kind == "none":
        return make(False).fit(X, y)
    if strategy.kind == "class_weight":
        return make(True).fit(X, y)
    if strategy.kind == "undersample":
        return make(False).fit(*undersample(X, y, cfg.sampling_ratio, seed))
    if strategy.kind == "smote":
        return make(False).fit(*smote(X, y, cfg.sampling_ratio, cfg.k_neighbors, seed))
    if strategy.kind == "adasyn":
        return make(False).fit(*adasyn(X, y, cfg.sampling_ratio, cfg.k_neighbors, seed))
    oof = np.zeros(len(y))                                              # focal: how hard is each row for a model that has not seen it?
    for train_idx, test_idx in StratifiedKFold(cfg.focal_inner_folds, shuffle=True, random_state=seed).split(X, y):
        oof[test_idx] = make(False).fit(X[train_idx], y[train_idx]).predict_proba(X[test_idx])[:, 1]
    weights = focal_style_weights(oof, y, cfg.focal_gamma, cfg.focal_floor)
    weights = weights / weights.mean()                                  # mean 1: otherwise the L2 penalty swamps tiny hessians
    return make(False).fit(X, y, sample_weight=weights)


def cross_validate_strategy(strategy: Strategy, X: np.ndarray, y: np.ndarray, folds: int, cfg: ImbalanceConfig = ImbalanceConfig(),
                            factory: Optional[ModelFactory] = None, seed: int = SEED) -> Dict[str, Any]:
    """Stratified k-fold cross-validation with the strategy applied inside every fold.

    Returns the PR-AUC of each fold, the out-of-fold score of every row (for threshold choice), and the seconds taken.
    """
    X, y = _check_xy(X, y, folds)
    start = time.perf_counter()
    oof = np.zeros(len(y))
    fold_scores: List[float] = []
    for train_idx, test_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        model = fit_strategy(strategy, X[train_idx], y[train_idx], cfg, factory, seed)
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        fold_scores.append(pr_auc(y[test_idx], oof[test_idx]))
    return {"fold_pr_auc": [round(s, 4) for s in fold_scores], "pr_auc_mean": float(np.mean(fold_scores)),
            "pr_auc_std": float(np.std(fold_scores, ddof=1)) if folds > 1 else 0.0, "oof_scores": oof,
            "seconds": round(time.perf_counter() - start, 1)}


def pick_winner(table: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Highest mean cross-validated PR-AUC, and every strategy within one standard deviation of it (a statistical tie).

    Raises:
        ValueError: If ``table`` is empty.
    """
    if not table:
        raise ValueError("no strategies to compare")
    best = max(table, key=lambda r: r["cv_pr_auc_mean"])
    tied = [r["strategy"] for r in table if r["cv_pr_auc_mean"] >= best["cv_pr_auc_mean"] - best["cv_pr_auc_std"] and r is not best]
    return {"winner": best["strategy"], "tied_with_winner": tied,
            "note": "Differences smaller than one standard deviation across folds are not reliable." if tied else "Clear winner."}
