"""Tests for fraud.policy (tier costs, tier search, robust thresholds, intervals) and fraud.optimize_threshold."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud import costs, optimize_threshold, policy, preprocess, train_supervised
from fraud.config import CostConfig, ImbalanceConfig, ProjectConfig, SupervisedConfig, ThresholdConfig
from tests.conftest import make_transactions

CFG = CostConfig()


def _scores(n: int = 3000, frauds: int = 60, seed: int = 7):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:frauds] = 1
    s = np.where(y == 1, rng.beta(5, 2, n), rng.beta(1, 30, n))
    a = rng.gamma(2.0, 40.0, n)
    return y, s, a


def test_tier_of_boundaries_and_errors() -> None:
    got = policy.tier_of([0.1, 0.5, 0.7, 0.9], 0.5, 0.8)
    assert got.tolist() == ["low", "medium", "medium", "high"]               # a score equal to a cut point goes to the higher tier
    assert policy.tier_of([0.1, 0.9], 0.5, 0.5).tolist() == ["low", "high"]    # no medium tier when the cuts coincide
    with pytest.raises(ValueError):
        policy.tier_of([0.1], 0.9, 0.5)


def test_tier_cost_hand_computed() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.1, 0.5, 0.9, 0.5, 0.9, 0.1])
    a = np.array([100.0, 200.0, 50.0, 10.0, 10.0, 10.0])
    cfg = CostConfig(fn_fixed=10.0, fp_cost=3.0, tp_review_cost=1.0, otp_cost=0.5, otp_fraud_catch_rate=0.8)
    got = policy.tier_cost(y, s, a, 0.4, 0.8, cfg)
    assert got["missed_fraud_low"] == pytest.approx(110.0)                    # fraud at 0.1 approved: 100 + 10
    assert got["missed_fraud_after_otp"] == pytest.approx(0.2 * 210.0)        # fraud at 0.5: 20% pass the OTP
    assert got["otp_checks"] == pytest.approx(0.5 * 2)                        # fraud + genuine at 0.5
    assert got["review_of_blocked_fraud"] == pytest.approx(1.0) and got["false_alarms_blocked"] == pytest.approx(3.0)
    assert got["total"] == pytest.approx(110 + 42 + 1 + 1 + 3)


def test_block_only_matches_the_step_1_cost_function() -> None:
    y, s, a = _scores()
    for t in (0.1, 0.5, 0.9, float("inf")):
        flagged = (s >= t).astype(int)
        assert policy.tier_cost(y, s, a, t, t, CFG)["total"] == pytest.approx(costs.total_cost(y, flagged, a, CFG)["total"])


def test_optimise_tiers_is_exact_against_brute_force() -> None:
    y, s, a = _scores(800, 25)
    cand = policy.candidate_thresholds(s, 25, 0.01, 0.5)
    got = policy.optimise_tiers(y, s, a, CFG, cand)
    brute = min(policy.tier_cost(y, s, a, lo, hi, CFG)["total"] for i, lo in enumerate(cand) for hi in cand[i:])
    assert got["cost"] == pytest.approx(brute) and got["t_medium"] <= got["t_high"]
    assert policy.tier_cost(y, s, a, got["t_medium"], got["t_high"], CFG)["total"] == pytest.approx(got["cost"])
    same_cut = min(policy.tier_cost(y, s, a, t, t, CFG)["total"] for t in cand)          # block-only rules are pairs with equal cuts
    assert got["cost"] <= same_cut + 1e-9


def test_free_otp_makes_a_wide_medium_tier() -> None:
    y, s, a = _scores()
    cheap = replace(CFG, otp_cost=0.0, otp_fraud_catch_rate=1.0)
    cand = policy.candidate_thresholds(s, 60, 0.001, 0.5)
    t = policy.optimise_tiers(y, s, a, cheap, cand)
    assert policy.tier_table(y, s, t["t_medium"], t["t_high"]).set_index("tier").loc["medium", "frauds"] > 0
    costly = replace(CFG, otp_cost=1000.0)                                   # an absurdly expensive OTP is never used
    t2 = policy.optimise_tiers(y, s, a, costly, cand)
    assert policy.tier_table(y, s, t2["t_medium"], t2["t_high"]).set_index("tier").loc["medium", "rows"] == 0


def test_candidates_and_validation_errors() -> None:
    s = np.random.default_rng(0).random(1000)
    c = policy.candidate_thresholds(s, 20, 0.001, 0.5)
    assert np.isinf(c[-1]) and (np.diff(c[:-1]) > 0).all() and len(c) <= 21
    for bad in (dict(n=1, min_share=0.1, max_share=0.5), dict(n=10, min_share=0.5, max_share=0.1), dict(n=10, min_share=0.0, max_share=0.5)):
        with pytest.raises(ValueError):
            policy.candidate_thresholds(s, bad["n"], bad["min_share"], bad["max_share"])
    with pytest.raises(ValueError):
        policy.candidate_thresholds([], 10, 0.1, 0.5)
    with pytest.raises(ValueError):
        policy.optimise_tiers([0, 0, 0], [0.1, 0.2, 0.3], [1, 1, 1], CFG, c)
    with pytest.raises(ValueError):
        policy.tier_cost([1, 0], [0.5, np.nan], [1, 1], 0.1, 0.5, CFG)
    with pytest.raises(ValueError):
        policy.tier_cost([1, 0], [0.5, 0.2], [1, -1], 0.1, 0.5, CFG)


def test_bootstrap_thresholds_and_robust_summary() -> None:
    y, s, a = _scores()
    boot = policy.bootstrap_thresholds(y, s, a, CFG, 20)
    assert boot.shape == (20,) and np.array_equal(boot, policy.bootstrap_thresholds(y, s, a, CFG, 20))      # seeded
    summary = policy.robust_threshold(boot)
    assert summary["min"] <= summary["q25"] <= summary["median"] <= summary["q75"] <= summary["max"]
    assert policy.robust_threshold([float("inf"), 0.3, 0.5])["median"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        policy.robust_threshold([float("inf")])
    with pytest.raises(ValueError):
        policy.bootstrap_thresholds(y, s, a, CFG, 5)


def test_saved_share_interval_brackets_the_estimate() -> None:
    y, s, a = _scores()
    t = policy.optimise_tiers(y, s, a, CFG, policy.candidate_thresholds(s, 40, 0.001, 0.5))
    row = policy.strategy_row("x", y, s, a, t["t_medium"], t["t_high"], CFG, 1_000_000)
    ci = policy.saved_share_interval(y, s, a, t["t_medium"], t["t_high"], CFG, 50)
    assert ci["low"] < row["saved_share"] < ci["high"] <= 1.0
    with pytest.raises(ValueError):
        policy.saved_share_interval(y, s, a, 0.1, 0.5, CFG, 5)


def test_strategy_row_and_tier_table_consistency() -> None:
    y, s, a = _scores()
    nothing = policy.strategy_row("approve", y, s, a, float("inf"), float("inf"), CFG, 1_000_000)
    assert nothing["saved"] == 0 and nothing["blocked"] == 0 and nothing["frauds_stopped"] == 0
    table = policy.tier_table(y, s, 0.2, 0.7)
    assert table["rows"].sum() == len(y) and table["frauds"].sum() == y.sum() and table["share_of_traffic"].sum() == pytest.approx(1.0, abs=1e-4)
    assert table.set_index("tier").loc["high", "fraud_rate"] > table.set_index("tier").loc["low", "fraud_rate"]


def test_sensitivity_has_one_row_per_scenario() -> None:
    y, s, a = _scores()
    tcfg = ThresholdConfig(tier_candidates=30, sensitivity_fp_costs=(1.0, 10.0), sensitivity_fn_fixed=(0.0, 50.0))
    table = policy.sensitivity(y, s, a, y, s, a, CFG, tcfg)
    assert len(table) == 4 and table["tiers_saved_share"].between(0, 1).all()
    cheap_fp = table[(table["fp_cost"] == 1.0) & (table["fn_fixed"] == 0.0)].iloc[0]["block_threshold"]
    dear_fp = table[(table["fp_cost"] == 10.0) & (table["fn_fixed"] == 0.0)].iloc[0]["block_threshold"]
    assert cheap_fp <= dear_fp                                                # costlier false alarms push the threshold up


def test_threshold_config_validation() -> None:
    for bad in (dict(bootstrap_resamples=5), dict(tier_candidates=2), dict(min_flag_share=0.6), dict(sensitivity_fp_costs=()), dict(sensitivity_fn_fixed=(-1.0,))):
        with pytest.raises(ValueError):
            ThresholdConfig(**bad)


def test_markdown_table() -> None:
    text = optimize_threshold.markdown_table(pd.DataFrame({"a": ["x"], "b": [0.5], "c": [12345.6]}))
    assert text.splitlines() == ["| a | b | c |", "|---|---|---|", "| x | 0.5000 | 12,346 |"]


def test_run_end_to_end(tmp_path: Path) -> None:
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=tmp_path), thresholds=ThresholdConfig(bootstrap_resamples=15, savings_resamples=15, tier_candidates=30),
                  supervised=SupervisedConfig(models=("logistic_regression", "random_forest"), rf_trees=20, rf_max_depth=6, bootstrap_resamples=20),
                  imbalance=ImbalanceConfig(sampling_ratio=0.2))
    cfg.paths.raw_path.parent.mkdir(parents=True)
    make_transactions(4000, 120).to_csv(cfg.paths.raw_path, index=False)
    preprocess.run(cfg, ("time",))
    train_supervised.run(cfg, method="time")
    summary = optimize_threshold.run(cfg, method="time")
    names = [r["strategy"] for r in summary["strategies"]]
    assert names[0] == "approve everything" and "three tiers (OTP + block)" in names and len(summary["sensitivity"]) == 9
    assert summary["recommended"]["saved_share"] <= 1.0 and summary["saved_share_interval"]["low"] <= summary["saved_share_interval"]["high"]
    reports = tmp_path / "reports"
    assert (reports / "cost_benefit.csv").is_file() and (reports / "sensitivity.csv").is_file() and (reports / "threshold_report.json").is_file()
    assert "Cost-benefit analysis" in (reports / "cost_benefit.md").read_text(encoding="utf-8")
    assert all((reports / "figures" / f).is_file() for f in ("cost_curves.png", "risk_tiers.png", "threshold_stability.png"))
    saved = json.loads((tmp_path / "models" / "decision_policy.json").read_text())
    assert saved["otp_threshold"] <= saved["high_threshold"] and saved["features"] == preprocess.feature_names()
    assert set(saved["tiers"]) == {"low", "medium", "high"}
    with pytest.raises(FileNotFoundError):
        optimize_threshold.run(replace(cfg, paths=replace(cfg.paths, root=tmp_path / "nowhere")))
