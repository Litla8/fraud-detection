"""The cost matrix in code: what a threshold costs, and which threshold costs least.

Everything is vectorised: one sort of the scores gives the cost of *every* possible threshold in O(n log n).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from fraud.config import CostConfig


def _check(y_true: Sequence, amount: Sequence, other: Optional[Sequence] = None) -> tuple:
    y = np.asarray(y_true).astype(int)
    a = np.asarray(amount, dtype=float)
    if y.ndim != 1 or y.shape != a.shape:
        raise ValueError("y_true and amount must be one-dimensional and the same length")
    if other is not None and np.asarray(other).shape != y.shape:
        raise ValueError("scores / decisions must be the same length as y_true")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if (a < 0).any() or np.isnan(a).any():
        raise ValueError("amount must be non-negative with no missing values")
    return y, a


def missed_fraud_cost(amount: np.ndarray, cfg: CostConfig) -> np.ndarray:
    """Cost of each missed fraud: the amount lost (if ``fn_uses_amount``) plus the fixed fee."""
    return (amount if cfg.fn_uses_amount else np.zeros_like(amount)) + cfg.fn_fixed


def total_cost(y_true: Sequence, flagged: Sequence, amount: Sequence, cfg: CostConfig = CostConfig()) -> Dict[str, float]:
    """Total cost of a set of block/approve decisions, split into its three parts."""
    y, a = _check(y_true, amount, flagged)
    f = np.asarray(flagged).astype(bool)
    fn = (y == 1) & ~f
    fp = (y == 0) & f
    tp = (y == 1) & f
    parts = {"missed_fraud": float(missed_fraud_cost(a, cfg)[fn].sum()), "false_alarms": float(fp.sum() * cfg.fp_cost),
             "review_of_caught_fraud": float(tp.sum() * cfg.tp_review_cost)}
    return {**parts, "total": float(sum(parts.values()))}


def cost_curve(y_true: Sequence, y_score: Sequence, amount: Sequence, cfg: CostConfig = CostConfig()) -> pd.DataFrame:
    """Cost of flagging every transaction with ``score >= threshold``, for every distinct threshold (plus "flag nothing").

    Columns: threshold, flagged, tp, fp, fn, cost. Ties in the score are flagged together, as a real threshold would.
    """
    y, a = _check(y_true, amount, y_score)
    s = np.asarray(y_score, dtype=float)
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN")
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    last_of_tie = np.r_[ss[1:] != ss[:-1], True]            # the cut points: after the last row of each score value
    cum_tp = np.cumsum(ys == 1)[last_of_tie]
    cum_fp = np.cumsum(ys == 0)[last_of_tie]
    miss = missed_fraud_cost(a, cfg)[order] * (ys == 1)
    caught_cost = np.cumsum(miss)[last_of_tie]
    frauds, all_miss = int(y.sum()), float(miss.sum())
    table = pd.DataFrame({"threshold": np.r_[np.inf, ss[last_of_tie]],
                          "flagged": np.r_[0, cum_tp + cum_fp], "tp": np.r_[0, cum_tp], "fp": np.r_[0, cum_fp]})
    table["fn"] = frauds - table["tp"]
    missed = np.r_[all_miss, all_miss - caught_cost]
    table["cost"] = missed + table["fp"] * cfg.fp_cost + table["tp"] * cfg.tp_review_cost
    return table


def best_threshold(curve: pd.DataFrame) -> Dict[str, Any]:
    """The cheapest row of a cost curve; among equal costs the highest threshold (the fewest flags)."""
    if curve.empty:
        raise ValueError("empty cost curve")
    best = curve.loc[curve["cost"] == curve["cost"].min()].iloc[0]
    return {"threshold": float(best["threshold"]), "cost": float(best["cost"]), "flagged": int(best["flagged"]),
            "tp": int(best["tp"]), "fp": int(best["fp"]), "fn": int(best["fn"])}


def baseline_costs(y_true: Sequence, amount: Sequence, cfg: CostConfig = CostConfig()) -> Dict[str, float]:
    """What doing nothing costs (approve everything) and what blocking everything costs."""
    y, a = _check(y_true, amount)
    approve_all = float(missed_fraud_cost(a, cfg)[y == 1].sum())
    block_all = float((y == 0).sum() * cfg.fp_cost + (y == 1).sum() * cfg.tp_review_cost)
    return {"approve_all": approve_all, "block_all": block_all}


def savings(model_cost: float, baselines: Dict[str, float]) -> Dict[str, float]:
    """Money saved against the do-nothing baseline, in total and as a share of it."""
    base = baselines["approve_all"]
    return {"saved": round(base - model_cost, 2), "saved_share": round((base - model_cost) / base, 4) if base else 0.0}


def break_even_precision(avg_missed_fraud_cost: float, cfg: CostConfig = CostConfig()) -> float:
    """Precision above which flagging a transaction saves money on average.

    Flagging gains ``avg_missed_fraud_cost - tp_review_cost`` when the transaction is fraud and costs ``fp_cost`` when it
    is not, so the break-even precision is ``fp_cost / (fp_cost + gain)``. With a cheap false alarm and an expensive
    missed fraud it is low: the cost-optimal threshold flags generously (compare the stricter targets of the brief).
    """
    gain = avg_missed_fraud_cost - cfg.tp_review_cost
    if gain <= 0:
        return 1.0
    return cfg.fp_cost / (cfg.fp_cost + gain)
