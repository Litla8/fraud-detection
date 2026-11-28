"""Step 6 building blocks: detectors trained on genuine rows only, percentile scores, the hybrid blend and the rescue analysis.

Every detector exposes ``fit(X) -> self`` and ``score(X) -> array`` where a larger score means "more unusual".
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

from fraud.config import AnomalyConfig, SEED


def clip_features(X: np.ndarray, clip: float) -> np.ndarray:
    """Float copy of ``X`` limited to ``[-clip, clip]`` (tames extreme amounts for distance-based models)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if np.isnan(X).any():
        raise ValueError("X contains NaN")
    return np.clip(X, -clip, clip)


def subsample_rows(X: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    """At most ``n`` random rows of ``X`` (all rows, in order, when ``X`` is smaller)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if len(X) <= n:
        return X
    return X[np.sort(np.random.default_rng(seed).choice(len(X), n, replace=False))]


class IsolationForestDetector:
    """Random trees isolate unusual rows in few splits: short average path = anomalous."""

    def __init__(self, cfg: AnomalyConfig = AnomalyConfig(), seed: int = SEED) -> None:
        self.cfg, self.seed = cfg, seed
        self.model = IsolationForest(n_estimators=cfg.iforest_trees, max_samples=cfg.iforest_max_samples, random_state=seed, n_jobs=-1)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(clip_features(X, self.cfg.clip))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(clip_features(X, self.cfg.clip))


class OneClassSvmDetector:
    """RBF boundary around the genuine rows (on a random subset, because the SVM is quadratic): outside the boundary = anomalous."""

    def __init__(self, cfg: AnomalyConfig = AnomalyConfig(), seed: int = SEED) -> None:
        self.cfg, self.seed = cfg, seed
        self.model = OneClassSVM(kernel="rbf", nu=cfg.ocsvm_nu, gamma=cfg.ocsvm_gamma)

    def fit(self, X: np.ndarray) -> "OneClassSvmDetector":
        self.model.fit(subsample_rows(clip_features(X, self.cfg.clip), self.cfg.ocsvm_rows, self.seed))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(clip_features(X, self.cfg.clip))


class AutoencoderDetector:
    """A small neural network that learns to rebuild genuine rows through a narrow middle layer.

    Rows unlike anything it saw during training are rebuilt badly, so the mean squared reconstruction error is the score.
    """

    def __init__(self, cfg: AnomalyConfig = AnomalyConfig(), seed: int = SEED) -> None:
        self.cfg, self.seed = cfg, seed
        self.model = MLPRegressor(hidden_layer_sizes=cfg.ae_hidden, activation="relu", solver="adam", batch_size=cfg.ae_batch_size,
                                  learning_rate_init=cfg.ae_learning_rate, max_iter=cfg.ae_epochs, early_stopping=False,
                                  n_iter_no_change=cfg.ae_epochs + 1, tol=0.0, random_state=seed)

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        Xc = subsample_rows(clip_features(X, self.cfg.clip), self.cfg.ae_rows, self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                 # fixed number of epochs by design, not run to convergence
            self.model.fit(Xc, Xc)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xc = clip_features(X, self.cfg.clip)
        return ((Xc - self.model.predict(Xc)) ** 2).mean(axis=1)


DETECTORS = {"isolation_forest": IsolationForestDetector, "one_class_svm": OneClassSvmDetector, "autoencoder": AutoencoderDetector}


def make_detector(name: str, cfg: AnomalyConfig = AnomalyConfig(), seed: int = SEED) -> Any:
    """An unfitted detector by name.

    Raises:
        ValueError: For an unknown name.
    """
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}")
    return DETECTORS[name](cfg, seed)


def fit_detector(name: str, X_genuine: np.ndarray, cfg: AnomalyConfig = AnomalyConfig(), seed: int = SEED) -> Any:
    """Fit detector ``name`` on genuine rows only.

    Raises:
        ValueError: If there are fewer than 20 rows or an unknown name.
    """
    if len(X_genuine) < 20:
        raise ValueError("need at least 20 genuine rows to fit a detector")
    return make_detector(name, cfg, seed).fit(X_genuine)


def reference_distribution(genuine_scores: Sequence[float]) -> np.ndarray:
    """Sorted scores of held-out genuine rows: the yardstick that turns a raw score into a percentile."""
    ref = np.sort(np.asarray(genuine_scores, dtype=float))
    if ref.size < 2 or np.isnan(ref).any():
        raise ValueError("need at least 2 non-NaN reference scores")
    return ref


def to_percentile(reference: np.ndarray, scores: Sequence[float]) -> np.ndarray:
    """Share of reference genuine rows that score at or below each score: 0.99 means "more unusual than 99% of genuine rows".

    Unlike a rank inside one batch, it does not depend on what else is being scored, so it can be used on a single transaction.
    """
    return np.searchsorted(reference, np.asarray(scores, dtype=float), side="right") / len(reference)


def blend_scores(supervised: Sequence[float], percentile: Sequence[float], weight: float) -> np.ndarray:
    """Hybrid score ``(1 - weight) * supervised + weight * anomaly percentile`` (both in [0, 1]); weight 0 is the supervised model."""
    s, a = np.asarray(supervised, dtype=float), np.asarray(percentile, dtype=float)
    if s.shape != a.shape or s.ndim != 1:
        raise ValueError("supervised and percentile must be one-dimensional and the same length")
    if not 0 <= weight < 1:
        raise ValueError("weight must be in [0, 1)")
    return (1 - weight) * s + weight * a


def rescue_analysis(y_true: Sequence[int], supervised: Sequence[float], supervised_threshold: float,
                    percentile: Sequence[float], quantile: float) -> Dict[str, Any]:
    """What the detector adds on top of the supervised model's own alerts.

    Frauds the supervised model misses (score below its threshold) are "rescued" if the detector puts them above ``quantile``;
    ``extra_false_alarms`` are the genuine rows the supervised model approved but the detector would also alert.
    """
    y, s, a = np.asarray(y_true).astype(int), np.asarray(supervised, dtype=float), np.asarray(percentile, dtype=float)
    if not (y.shape == s.shape == a.shape) or y.ndim != 1:
        raise ValueError("inputs must be one-dimensional and the same length")
    missed = (y == 1) & (s < supervised_threshold)
    alerted = a >= quantile
    rescued = int((missed & alerted).sum())
    extra_false = int(((y == 0) & (s < supervised_threshold) & alerted).sum())
    total_alerts = int((alerted & (s < supervised_threshold)).sum())
    return {"frauds": int(y.sum()), "missed_by_supervised": int(missed.sum()), "rescued": rescued,
            "rescued_share_of_missed": round(rescued / missed.sum(), 4) if missed.any() else 0.0,
            "extra_false_alarms": extra_false, "precision_of_extra_alerts": round(rescued / total_alerts, 4) if total_alerts else 0.0,
            "frauds_caught_by_supervised": int(((y == 1) & ~missed).sum())}
