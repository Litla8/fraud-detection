"""Tests for fraud.costs: hand-computed cases plus a brute-force check of the vectorised cost curve."""
from __future__ import annotations

import numpy as np
import pytest

from fraud import costs
from fraud.config import CostConfig

CFG = CostConfig(fn_fixed=10.0, fp_cost=2.0, tp_review_cost=1.0)
Y = np.array([1, 1, 0, 0, 0, 1])
AMOUNT = np.array([100.0, 50.0, 20.0, 10.0, 5.0, 30.0])


def test_total_cost_by_hand() -> None:
    flagged = np.array([1, 0, 1, 0, 0, 0])      # catches fraud 1, misses frauds 2 and 6, one false alarm
    got = costs.total_cost(Y, flagged, AMOUNT, CFG)
    assert got["missed_fraud"] == (50 + 10) + (30 + 10)
    assert got["false_alarms"] == 2.0 and got["review_of_caught_fraud"] == 1.0 and got["total"] == 103.0


def test_amount_ignored_when_configured() -> None:
    cfg = CostConfig(fn_fixed=10.0, fn_uses_amount=False)
    assert costs.total_cost(Y, np.zeros(6), AMOUNT, cfg)["missed_fraud"] == 30.0


def test_baselines() -> None:
    base = costs.baseline_costs(Y, AMOUNT, CFG)
    assert base["approve_all"] == (100 + 10) + (50 + 10) + (30 + 10) == 210
    assert base["block_all"] == 3 * 2.0 + 3 * 1.0
    assert costs.savings(105.0, base) == {"saved": 105.0, "saved_share": 0.5}


def test_cost_curve_matches_brute_force() -> None:
    rng = np.random.default_rng(42)
    y = (rng.random(300) < 0.1).astype(int)
    amount = rng.gamma(2, 40, 300).round(2)
    score = np.round(rng.random(300) + 0.5 * y, 1)             # rounded on purpose: many ties
    curve = costs.cost_curve(y, score, amount, CFG)
    assert curve["threshold"].iloc[0] == np.inf and curve["flagged"].iloc[0] == 0
    for _, row in curve.iterrows():
        brute = costs.total_cost(y, (score >= row["threshold"]).astype(int), amount, CFG)["total"]
        assert row["cost"] == pytest.approx(brute)
        assert row["flagged"] == (score >= row["threshold"]).sum()
    assert curve["cost"].iloc[0] == pytest.approx(costs.baseline_costs(y, amount, CFG)["approve_all"])


def test_best_threshold_picks_cheapest() -> None:
    score = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.7])
    best = costs.best_threshold(costs.cost_curve(Y, score, AMOUNT, CFG))
    assert best["threshold"] == 0.7 and best["tp"] == 3 and best["fp"] == 0 and best["fn"] == 0
    assert best["cost"] == 3.0                                   # only the three review costs remain


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="same length"):
        costs.total_cost(Y, np.zeros(6), AMOUNT[:3], CFG)
    with pytest.raises(ValueError, match="non-negative"):
        costs.total_cost(Y, np.zeros(6), -AMOUNT, CFG)
    with pytest.raises(ValueError, match="empty"):
        costs.best_threshold(costs.cost_curve(Y, np.zeros(6), AMOUNT, CFG).iloc[0:0])
