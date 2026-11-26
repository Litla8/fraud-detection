"""Tests for fraud.metrics."""
from __future__ import annotations

import numpy as np
import pytest

from fraud import metrics

Y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
PERFECT = np.array([0.1, 0.2, 0.1, 0.0, 0.3, 0.2, 0.9, 0.8, 0.95, 0.4])


def test_pr_auc_perfect_and_random_order() -> None:
    assert metrics.pr_auc(Y, PERFECT) == pytest.approx(1.0)
    assert metrics.pr_auc(Y, 1 - PERFECT) < 0.5


def test_recall_at_precision() -> None:
    got = metrics.recall_at_precision(Y, PERFECT, 0.85)
    assert got["recall"] == 1.0 and got["precision"] == 1.0 and got["threshold"] == pytest.approx(0.8)
    nothing = metrics.recall_at_precision(Y, 1 - PERFECT, 0.99)
    assert nothing == {"recall": 0.0, "precision": 0.0, "threshold": float("inf")}
    with pytest.raises(ValueError):
        metrics.recall_at_precision(Y, PERFECT, 0)


def test_threshold_metrics_counts() -> None:
    m = metrics.threshold_metrics(Y, (PERFECT >= 0.5).astype(int))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 0, 0, 7)
    assert m["f2"] == pytest.approx(1.0) and m["mcc"] == pytest.approx(1.0)
    flag_all = metrics.threshold_metrics(Y, np.ones(10))
    assert flag_all["recall"] == 1.0 and flag_all["precision"] == pytest.approx(0.3)


def test_evaluate_scores_and_targets() -> None:
    out = metrics.evaluate_scores(Y, PERFECT, 0.5)
    assert out["meets_targets"] == {"pr_auc": True, "recall_at_precision": True}
    assert metrics.check_targets(0.8, 0.79) == {"pr_auc": False, "recall_at_precision": False}   # PR-AUC must be strictly above


def test_always_genuine_accuracy() -> None:
    assert metrics.always_genuine_accuracy(Y) == 0.7
    with pytest.raises(ValueError):
        metrics.always_genuine_accuracy([])


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="both classes"):
        metrics.pr_auc([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="same length"):
        metrics.pr_auc([0, 1], [0.1])
    with pytest.raises(ValueError, match="only 0 and 1"):
        metrics.pr_auc([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        metrics.pr_auc([0, 1], [0.1, np.nan])
