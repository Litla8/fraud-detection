"""Decision policy maths for Step 8: tier costs, the best pair of tier thresholds, robust thresholds and savings intervals.

A policy has two cut points on the model score. ``score < t_medium`` is approved (low risk); ``t_medium <= score < t_high``
gets an OTP check (medium); ``score >= t_high`` is blocked and reviewed (high). A plain block-or-approve rule is the special
case ``t_medium == t_high``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from fraud import costs, metrics
from fraud.config import CostConfig, SEED, ThresholdConfig

TIERS = ("low", "medium", "high")


def _prepare(y_true: Sequence, score: Sequence, amount: Sequence) -> tuple:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    a = np.asarray(amount, dtype=float)
    if not (y.shape == s.shape == a.shape) or y.ndim != 1:
        raise ValueError("y_true, score and amount must be one-dimensional and the same length")
    if np.isnan(s).any() or np.isnan(a).any() or (a < 0).any():
        raise ValueError("score and amount must have no NaN and amount must be non-negative")
    return y, s, a


def tier_of(score: Sequence, t_medium: float, t_high: float) -> np.ndarray:
    """Tier name for each score: ``low`` below ``t_medium``, ``high`` from ``t_high``, otherwise ``medium``.

    Raises:
        ValueError: If ``t_medium > t_high``.
    """
    if t_medium > t_high:
        raise ValueError("t_medium must not exceed t_high")
    s = np.asarray(score, dtype=float)
    return np.where(s >= t_high, "high", np.where(s >= t_medium, "medium", "low"))


def tier_cost(y_true: Sequence, score: Sequence, amount: Sequence, t_medium: float, t_high: float, cfg: CostConfig = CostConfig()) -> Dict[str, float]:
    """Total cost of a tiered policy.

    low: a fraud costs its full missed cost. medium: every row pays the OTP cost, and a fraud also costs its missed cost unless the OTP
    stops it (``otp_fraud_catch_rate``, applied as an expected value). high: a fraud costs the review, a genuine row the false-alarm cost.
    """
    y, s, a = _prepare(y_true, score, amount)
    tier = tier_of(s, t_medium, t_high)
    miss = costs.missed_fraud_cost(a, cfg)
    fraud, genuine = y == 1, y == 0
    parts = {"missed_fraud_low": float(miss[fraud & (tier == "low")].sum()),
             "missed_fraud_after_otp": float((1 - cfg.otp_fraud_catch_rate) * miss[fraud & (tier == "medium")].sum()),
             "otp_checks": float(cfg.otp_cost * (tier == "medium").sum()),
             "review_of_blocked_fraud": float(cfg.tp_review_cost * (fraud & (tier == "high")).sum()),
             "false_alarms_blocked": float(cfg.fp_cost * (genuine & (tier == "high")).sum())}
    return {**parts, "total": float(sum(parts.values()))}


def candidate_thresholds(score: Sequence, n: int, min_share: float, max_share: float) -> np.ndarray:
    """``n`` cut points between the top ``min_share`` and top ``max_share`` of the scores (log-spaced), plus ``inf`` ("no such tier")."""
    s = np.asarray(score, dtype=float)
    if s.size == 0 or n < 2 or not 0 < min_share < max_share <= 1:
        raise ValueError("need scores, n >= 2 and 0 < min_share < max_share <= 1")
    shares = np.logspace(np.log10(min_share), np.log10(max_share), n)
    cuts = np.unique(np.quantile(s, 1 - shares))
    return np.r_[cuts, np.inf]


def optimise_tiers(y_true: Sequence, score: Sequence, amount: Sequence, cfg: CostConfig, candidates: Sequence[float],
                   max_medium_share: float = 1.0) -> Dict[str, Any]:
    """The cheapest pair ``t_medium <= t_high`` among ``candidates`` (every pair is tried, using cumulative sums, so it is exact).

    ``max_medium_share`` caps the share of all transactions that may get an OTP check (the step-up capacity of the business); pairs
    with no medium tier are always allowed, so a solution always exists. Ties go to the pair with the highest thresholds.
    """
    if not 0 < max_medium_share <= 1:
        raise ValueError("max_medium_share must be in (0, 1]")
    y, s, a = _prepare(y_true, score, amount)
    c = np.unique(np.asarray(candidates, dtype=float))
    if c.size < 2 or (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("need at least 2 candidates and both classes")
    miss = costs.missed_fraud_cost(a, cfg)
    f_order = np.argsort(s[y == 1], kind="stable")
    f_scores = s[y == 1][f_order]
    cum_miss = np.r_[0.0, np.cumsum(miss[y == 1][f_order])]
    g_scores = np.sort(s[y == 0])
    nf, ng = len(f_scores), len(g_scores)
    fi, gi = np.searchsorted(f_scores, c, side="left"), np.searchsorted(g_scores, c, side="left")      # rows below each cut
    total = (cum_miss[fi][:, None]                                                                     # frauds approved (low tier)
             + (1 - cfg.otp_fraud_catch_rate) * (cum_miss[fi][None, :] - cum_miss[fi][:, None])        # medium frauds that pass the OTP
             + cfg.otp_cost * ((fi[None, :] - fi[:, None]) + (gi[None, :] - gi[:, None]))              # every medium row is checked
             + cfg.tp_review_cost * (nf - fi)[None, :] + cfg.fp_cost * (ng - gi)[None, :])             # high tier
    medium_rows = (fi[None, :] - fi[:, None]) + (gi[None, :] - gi[:, None])
    allowed = np.triu(np.ones_like(total, dtype=bool)) & (medium_rows <= max_medium_share * (nf + ng))
    total = np.where(allowed, total, np.inf)                                                           # medium cut <= high cut, OTP volume capped
    best = total.min()
    rows, cols = np.nonzero(total == best)
    pick = np.lexsort((cols, rows))[-1]                                                                # highest thresholds among ties
    return {"t_medium": float(c[rows[pick]]), "t_high": float(c[cols[pick]]), "cost": float(best)}


def best_binary_threshold(y_true: Sequence, score: Sequence, amount: Sequence, cfg: CostConfig) -> float:
    """The cost-minimising block threshold (``inf`` means "never block")."""
    return float(costs.best_threshold(costs.cost_curve(y_true, score, amount, cfg))["threshold"])


def bootstrap_thresholds(y_true: Sequence, score: Sequence, amount: Sequence, cfg: CostConfig, n_resamples: int, seed: int = SEED) -> np.ndarray:
    """Cost-optimal block threshold on each stratified bootstrap resample: how much the choice depends on which frauds happened to be sampled."""
    y, s, a = _prepare(y_true, score, amount)
    if n_resamples < 10:
        raise ValueError("n_resamples must be at least 10")
    rng = np.random.default_rng(seed)
    fraud, genuine = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if fraud.size == 0 or genuine.size == 0:
        raise ValueError("need both classes")
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = np.concatenate([rng.choice(fraud, fraud.size), rng.choice(genuine, genuine.size)])
        out[i] = best_binary_threshold(y[idx], s[idx], a[idx], cfg)
    return out


def robust_threshold(thresholds: Sequence[float]) -> Dict[str, float]:
    """Median and inter-quartile range of bootstrap thresholds (infinite values, i.e. "block nothing", are ignored)."""
    t = np.asarray(thresholds, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        raise ValueError("no finite thresholds")
    q1, med, q3 = np.quantile(t, [0.25, 0.5, 0.75])
    return {"median": float(med), "q25": float(q1), "q75": float(q3), "min": float(t.min()), "max": float(t.max())}


def saved_share_interval(y_true: Sequence, score: Sequence, amount: Sequence, t_medium: float, t_high: float, cfg: CostConfig,
                         n_resamples: int, seed: int = SEED, confidence: float = 0.95) -> Dict[str, float]:
    """Interval for the share of the do-nothing cost a policy saves, from a stratified bootstrap of the (fixed-threshold) test set."""
    y, s, a = _prepare(y_true, score, amount)
    if n_resamples < 10 or not 0 < confidence < 1:
        raise ValueError("need n_resamples >= 10 and confidence in (0, 1)")
    rng = np.random.default_rng(seed)
    fraud, genuine = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    shares = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = np.concatenate([rng.choice(fraud, fraud.size), rng.choice(genuine, genuine.size)])
        base = costs.baseline_costs(y[idx], a[idx], cfg)["approve_all"]
        shares[i] = 1 - tier_cost(y[idx], s[idx], a[idx], t_medium, t_high, cfg)["total"] / base if base else 0.0
    tail = (1 - confidence) / 2
    low, high = np.quantile(shares, [tail, 1 - tail])
    return {"low": float(low), "high": float(high)}


def tier_table(y_true: Sequence, score: Sequence, t_medium: float, t_high: float) -> pd.DataFrame:
    """Rows, frauds, genuine rows, share of traffic and fraud rate per tier."""
    y = np.asarray(y_true).astype(int)
    tier = tier_of(score, t_medium, t_high)
    table = pd.DataFrame({"tier": list(TIERS)})
    table["rows"] = [int((tier == t).sum()) for t in TIERS]
    table["frauds"] = [int(((tier == t) & (y == 1)).sum()) for t in TIERS]
    table["genuine"] = table["rows"] - table["frauds"]
    table["share_of_traffic"] = (table["rows"] / max(len(y), 1)).round(6)
    table["fraud_rate"] = (table["frauds"] / table["rows"].where(table["rows"] > 0)).fillna(0.0).round(4)
    return table


def strategy_row(name: str, y: np.ndarray, s: np.ndarray, a: np.ndarray, t_medium: float, t_high: float, cfg: CostConfig, per: int) -> Dict[str, Any]:
    """One line of the cost-benefit table for a policy (a block-only rule has ``t_medium == t_high``)."""
    cost = tier_cost(y, s, a, t_medium, t_high, cfg)["total"]
    base = costs.baseline_costs(y, a, cfg)["approve_all"]
    tier = tier_of(s, t_medium, t_high)
    blocked = tier == "high"
    stepped = tier == "medium"
    caught = int(((y == 1) & blocked).sum() + round(cfg.otp_fraud_catch_rate * ((y == 1) & stepped).sum()))
    return {"strategy": name, "t_medium": t_medium, "t_high": t_high, "blocked": int(blocked.sum()), "otp_checks": int(stepped.sum()),
            "frauds_stopped": caught, "frauds": int((y == 1).sum()), "cost": round(cost, 2), "approve_all_cost": round(base, 2),
            "saved": round(base - cost, 2), "saved_share": round((base - cost) / base, 4) if base else 0.0,
            "saved_per_million": round((base - cost) / len(y) * per, 0)}


def sensitivity(y_val: np.ndarray, s_val: np.ndarray, a_val: np.ndarray, y_test: np.ndarray, s_test: np.ndarray, a_test: np.ndarray,
                cfg: CostConfig, tcfg: ThresholdConfig) -> pd.DataFrame:
    """Re-choose the thresholds on validation under other cost assumptions and score them on test: is the saving robust to the matrix?"""
    rows: List[Dict[str, Any]] = []
    candidates = candidate_thresholds(s_val, tcfg.tier_candidates, tcfg.min_flag_share, tcfg.max_flag_share)
    for fp in tcfg.sensitivity_fp_costs:
        for fn in tcfg.sensitivity_fn_fixed:
            alt = replace(cfg, fp_cost=fp, fn_fixed=fn)
            t = best_binary_threshold(y_val, s_val, a_val, alt)
            tiers = optimise_tiers(y_val, s_val, a_val, alt, candidates, tcfg.max_medium_share)
            block = strategy_row("block", y_test, s_test, a_test, t, t, alt, tcfg.per_transactions)
            tiered = strategy_row("tiers", y_test, s_test, a_test, tiers["t_medium"], tiers["t_high"], alt, tcfg.per_transactions)
            rows.append({"fp_cost": fp, "fn_fixed": fn, "block_threshold": t, "block_saved_share": block["saved_share"],
                         "t_medium": tiers["t_medium"], "t_high": tiers["t_high"], "tiers_saved_share": tiered["saved_share"],
                         "tiers_saved_per_million": tiered["saved_per_million"]})
    return pd.DataFrame(rows)


def otp_sensitivity(y_val: np.ndarray, s_val: np.ndarray, a_val: np.ndarray, y_test: np.ndarray, s_test: np.ndarray, a_test: np.ndarray,
                    cfg: CostConfig, tcfg: ThresholdConfig) -> pd.DataFrame:
    """How much of the tiered saving survives weaker or costlier OTP checks (the OTP numbers are assumptions, not data).

    Tiers are re-chosen on validation for every (catch rate, cost) pair and scored on test; ``block_saved_share`` is the best block-only rule
    under the same costs, so ``tiers_beat_block`` says whether the medium tier is still worth having.
    """
    rows: List[Dict[str, Any]] = []
    candidates = candidate_thresholds(s_val, tcfg.tier_candidates, tcfg.min_flag_share, tcfg.max_flag_share)
    t = best_binary_threshold(y_val, s_val, a_val, cfg)
    block = strategy_row("block", y_test, s_test, a_test, t, t, cfg, tcfg.per_transactions)
    for catch in tcfg.sensitivity_otp_catch:
        for otp in tcfg.sensitivity_otp_cost:
            alt = replace(cfg, otp_fraud_catch_rate=catch, otp_cost=otp)
            tiers = optimise_tiers(y_val, s_val, a_val, alt, candidates, tcfg.max_medium_share)
            row = strategy_row("tiers", y_test, s_test, a_test, tiers["t_medium"], tiers["t_high"], alt, tcfg.per_transactions)
            rows.append({"otp_catch_rate": catch, "otp_cost": otp, "t_medium": tiers["t_medium"], "t_high": tiers["t_high"],
                         "otp_checks": row["otp_checks"], "otp_share_of_traffic": round(row["otp_checks"] / len(y_test), 5),
                         "tiers_saved_share": row["saved_share"], "block_saved_share": block["saved_share"],
                         "tiers_beat_block": bool(row["saved_share"] > block["saved_share"])})
    return pd.DataFrame(rows)
