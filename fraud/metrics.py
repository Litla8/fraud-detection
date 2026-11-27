"""Metrics for extremely imbalanced classification. Accuracy is deliberately absent: "always genuine" scores 99.83%."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef, precision_recall_curve

from fraud.config import TargetConfig


def _as_arrays(y_true: Sequence, y_score: Sequence) -> tuple:
    """Validate and convert to numpy: same length, labels 0/1 with both classes present."""
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("y_true and y_score must be one-dimensional and the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if len(np.unique(y)) < 2:
        raise ValueError("y_true needs both classes for precision-recall metrics")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN")
    return y.astype(int), s


def pr_auc(y_true: Sequence, y_score: Sequence) -> float:
    """Average precision (area under the precision-recall curve), the headline metric for this project."""
    y, s = _as_arrays(y_true, y_score)
    return float(average_precision_score(y, s))


def recall_at_precision(y_true: Sequence, y_score: Sequence, min_precision: float) -> Dict[str, float]:
    """Best recall reachable while precision stays at or above ``min_precision``, and the threshold that gets it.

    Returns recall 0 and threshold ``inf`` (flag nothing) when no threshold reaches that precision.
    """
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")
    y, s = _as_arrays(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y, s)
    ok = precision[:-1] >= min_precision                    # the last point (precision 1, recall 0) has no threshold
    if not ok.any():
        return {"recall": 0.0, "precision": 0.0, "threshold": float("inf")}
    best = int(np.flatnonzero(ok)[np.argmax(recall[:-1][ok])])
    return {"recall": float(recall[best]), "precision": float(precision[best]), "threshold": float(thresholds[best])}


def threshold_metrics(y_true: Sequence, flagged: Sequence) -> Dict[str, Any]:
    """Confusion counts, precision, recall, F2 (recall counts double) and MCC for hard 0/1 decisions."""
    y = np.asarray(y_true).astype(int)
    f = np.asarray(flagged).astype(int)
    if y.shape != f.shape or y.ndim != 1:
        raise ValueError("y_true and flagged must be one-dimensional and the same length")
    tp, fp = int(((y == 1) & (f == 1)).sum()), int(((y == 0) & (f == 1)).sum())
    fn, tn = int(((y == 1) & (f == 0)).sum()), int(((y == 0) & (f == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f2": float(fbeta_score(y, f, beta=2, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, f)) if len(np.unique(f)) > 1 or len(np.unique(y)) > 1 else 0.0}


def evaluate_scores(y_true: Sequence, y_score: Sequence, threshold: float, targets: TargetConfig = TargetConfig()) -> Dict[str, Any]:
    """Every Step 5 number for one model: PR-AUC, recall at the precision floor, and F2 / MCC at ``threshold``."""
    y, s = _as_arrays(y_true, y_score)
    at_floor = recall_at_precision(y, s, targets.precision_floor)
    return {"pr_auc": pr_auc(y, s), "recall_at_precision_floor": at_floor,
            "at_threshold": {"threshold": float(threshold), **threshold_metrics(y, (s >= threshold).astype(int))},
            "meets_targets": check_targets(pr_auc(y, s), at_floor["recall"], targets)}


def check_targets(pr_auc_value: float, recall_at_floor: float, targets: TargetConfig = TargetConfig()) -> Dict[str, bool]:
    """Which success metrics are met (strictly above the PR-AUC target; recall at or above its target)."""
    return {"pr_auc": bool(pr_auc_value > targets.pr_auc), "recall_at_precision": bool(recall_at_floor >= targets.recall)}


def always_genuine_accuracy(y_true: Sequence) -> float:
    """Accuracy of a model that never flags anything: the number that shows why accuracy is useless here."""
    y = np.asarray(y_true).astype(int)
    if y.size == 0:
        raise ValueError("y_true is empty")
    return float((y == 0).mean())


def best_f2_threshold(y_true: Sequence, y_score: Sequence) -> float:
    """The score threshold that maximises F2 (recall counts double), searched over every distinct threshold."""
    y, s = _as_arrays(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y, s)
    p, r = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = np.where(4 * p + r > 0, 5 * p * r / (4 * p + r), 0.0)
    return float(thresholds[int(np.argmax(f2))])


def bootstrap_pr_auc(y_true: Sequence, y_score: Sequence, n_resamples: int = 500, seed: int = 42, confidence: float = 0.95) -> Dict[str, float]:
    """Confidence interval for PR-AUC from a stratified bootstrap (frauds and genuine rows are resampled separately).

    With only a few dozen frauds in a test set, the interval is wide: it shows how much of a score is luck.
    """
    y, s = _as_arrays(y_true, y_score)
    if n_resamples < 10 or not 0 < confidence < 1:
        raise ValueError("need n_resamples >= 10 and confidence in (0, 1)")
    rng = np.random.default_rng(seed)
    fraud, genuine = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = np.concatenate([rng.choice(fraud, len(fraud)), rng.choice(genuine, len(genuine))])
        values[i] = average_precision_score(y[idx], s[idx])
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return {"low": float(low), "high": float(high), "std": float(values.std(ddof=1))}


def apply_threshold(y_true: Sequence, y_score: Sequence, threshold: float) -> Dict[str, Any]:
    """Precision, recall, F2 and MCC when ``y_score >= threshold`` is flagged (a threshold chosen elsewhere, so the numbers are honest)."""
    y, s = _as_arrays(y_true, y_score)
    return {"threshold": float(threshold), **threshold_metrics(y, (s >= threshold).astype(int))}
