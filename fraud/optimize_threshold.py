"""Step 8 pipeline: cost-optimal threshold, risk tiers (approve / OTP / block), cost-benefit table and sensitivity analysis.

Thresholds are chosen on validation scores only. The test set is scored once with those fixed thresholds.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from fraud import costs, metrics, plots, policy, preprocess
from fraud.config import ProjectConfig
from fraud.logging_utils import configure_logging
from fraud.train_anomaly import load_supervised

LOGGER = logging.getLogger(__name__)


def markdown_table(table: pd.DataFrame) -> str:
    """A GitHub-style Markdown table (no optional ``tabulate`` dependency needed)."""
    head = "| " + " | ".join(map(str, table.columns)) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join((f"{v:,.0f}" if abs(v) >= 100 else f"{v:.4f}") if isinstance(v, float) else str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def markdown_report(summary: Dict[str, Any], cfg: ProjectConfig) -> str:
    """The cost-benefit analysis as a short Markdown document (assumptions, results, caveats)."""
    c = cfg.costs
    table = pd.DataFrame(summary["strategies"])[["strategy", "blocked", "otp_checks", "frauds_stopped", "cost", "saved", "saved_share", "saved_per_million"]]
    tiers = pd.DataFrame(summary["test_tiers"])
    rec = summary["recommended"]
    lines = ["# Cost-benefit analysis", "",
             f"Model: `{summary['model_variant']}`. Thresholds were chosen on validation data and applied unchanged to the test set "
             f"({summary['test_rows']:,} transactions, {summary['test_frauds']} frauds).", "", "## Cost assumptions (change in `CostConfig`)", "",
             f"- Missed fraud: transaction amount + EUR {c.fn_fixed:g}", f"- False alarm (genuine blocked): EUR {c.fp_cost:g}",
             f"- Caught fraud (review): EUR {c.tp_review_cost:g}", f"- OTP check: EUR {c.otp_cost:g}, stops {c.otp_fraud_catch_rate:.0%} of fraud", "",
             "## Strategies on the test set", "", markdown_table(table), "",
             "## Risk tiers on the test set", "", markdown_table(tiers), "",
             "## How much depends on the OTP assumptions", "", markdown_table(pd.DataFrame(summary["otp_sensitivity"])[
                 ["otp_catch_rate", "otp_cost", "otp_share_of_traffic", "tiers_saved_share", "block_saved_share", "tiers_beat_block"]]), "",
             "## Result", "",
             f"The recommended tiered policy saves EUR {rec['saved']:,.2f} of EUR {rec['approve_all_cost']:,.2f} ({rec['saved_share']:.1%}; 95% interval "
             f"{summary['saved_share_interval']['low']:.1%} to {summary['saved_share_interval']['high']:.1%}), about EUR {rec['saved_per_million']:,.0f} "
             f"per {cfg.thresholds.per_transactions:,} transactions.", "",
             "## Caveats", "",
             f"- The test set has only {summary['test_frauds']} frauds, so every saving carries a wide interval.",
             "- The cost matrix is an assumption; see `sensitivity.csv` for how the saving moves when it changes.",
             "- The file covers two days of one card network: extrapolating a saving to a year is not justified by this data.", ""]
    return "\n".join(lines)


def run(cfg: ProjectConfig, method: Optional[str] = None) -> Dict[str, Any]:
    """Choose thresholds and tiers on validation, evaluate on test, write reports, charts and ``models/decision_policy.json``.

    Raises:
        FileNotFoundError: If Step 3 or Step 5 outputs are missing.
    """
    tcfg, ccfg = cfg.thresholds, cfg.costs
    parts = preprocess.load_split(cfg, method)
    val, test = parts["validation"], parts["test"]
    names = preprocess.feature_names(cfg.data, cfg.prep)
    label, raw = cfg.data.label_column, cfg.prep.raw_amount_name
    sup = load_supervised(cfg)
    yv, av, ys, as_ = val[label].to_numpy(), val[raw].to_numpy(), test[label].to_numpy(), test[raw].to_numpy()
    sv, ss = sup["model"].predict_proba(val[names].to_numpy(float))[:, 1], sup["model"].predict_proba(test[names].to_numpy(float))[:, 1]
    # 1. block-or-approve thresholds, all picked on validation
    point = policy.best_binary_threshold(yv, sv, av, ccfg)
    boot = policy.bootstrap_thresholds(yv, sv, av, ccfg, tcfg.bootstrap_resamples)
    stability = policy.robust_threshold(boot)
    robust = stability["median"]
    f2_t = metrics.best_f2_threshold(yv, sv)
    p85_t = metrics.recall_at_precision(yv, sv, cfg.targets.precision_floor)["threshold"]
    # 2. risk tiers, picked on validation
    candidates = policy.candidate_thresholds(sv, tcfg.tier_candidates, tcfg.min_flag_share, tcfg.max_flag_share)
    tiers = policy.optimise_tiers(yv, sv, av, ccfg, candidates, tcfg.max_medium_share)
    LOGGER.info("validation: block threshold %.4f (bootstrap median %.4f, IQR %.4f-%.4f); tiers OTP>=%.4f block>=%.4f", point, robust,
                stability["q25"], stability["q75"], tiers["t_medium"], tiers["t_high"])
    # 3. everything below is the test set, with those fixed thresholds
    per = tcfg.per_transactions
    oracle = policy.best_binary_threshold(ys, ss, as_, ccfg)              # best case: threshold picked on test itself, not achievable
    inf = float("inf")
    strategies: List[Dict[str, Any]] = [
        policy.strategy_row("approve everything", ys, ss, as_, inf, inf, ccfg, per),
        policy.strategy_row("block everything", ys, ss, as_, -inf, -inf, ccfg, per),
        policy.strategy_row("block at F2 threshold", ys, ss, as_, f2_t, f2_t, ccfg, per),
        policy.strategy_row("block at precision-0.85 threshold", ys, ss, as_, p85_t, p85_t, ccfg, per),
        policy.strategy_row("block at cost-optimal threshold (point)", ys, ss, as_, point, point, ccfg, per),
        policy.strategy_row("block at cost-optimal threshold (bootstrap median)", ys, ss, as_, robust, robust, ccfg, per),
        policy.strategy_row("three tiers (OTP + block)", ys, ss, as_, tiers["t_medium"], tiers["t_high"], ccfg, per),
        policy.strategy_row("[best case] block at test-optimal threshold", ys, ss, as_, oracle, oracle, ccfg, per)]
    table = pd.DataFrame(strategies)
    recommended = next(r for r in strategies if r["strategy"] == "three tiers (OTP + block)")
    interval = policy.saved_share_interval(ys, ss, as_, tiers["t_medium"], tiers["t_high"], ccfg, tcfg.savings_resamples)
    sens = policy.sensitivity(yv, sv, av, ys, ss, as_, ccfg, tcfg)
    otp_sens = policy.otp_sensitivity(yv, sv, av, ys, ss, as_, ccfg, tcfg)
    val_tiers = policy.tier_table(yv, sv, tiers["t_medium"], tiers["t_high"])
    test_tiers = policy.tier_table(ys, ss, tiers["t_medium"], tiers["t_high"])
    for row in strategies:
        LOGGER.info("%-52s cost %9.2f | saved %8.2f (%5.1f%%) | blocked %5d | OTP %5d | frauds stopped %d/%d", row["strategy"], row["cost"], row["saved"],
                    100 * row["saved_share"], row["blocked"], row["otp_checks"], row["frauds_stopped"], row["frauds"])
    weak = otp_sens[(otp_sens["otp_catch_rate"] == otp_sens["otp_catch_rate"].min()) & (otp_sens["otp_cost"] == otp_sens["otp_cost"].max())].iloc[0]
    LOGGER.info("with the weakest OTP assumptions (catch %.0f%%, cost %.2f) the tiers save %.1f%% vs %.1f%% for block-only", 100 * weak["otp_catch_rate"],
                weak["otp_cost"], 100 * weak["tiers_saved_share"], 100 * weak["block_saved_share"])
    LOGGER.info("tiered policy saves %.1f%% (95%% interval %.1f%% to %.1f%%)", 100 * recommended["saved_share"], 100 * interval["low"], 100 * interval["high"])
    reports, figures = cfg.paths.reports_path, cfg.paths.reports_path / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "cost_benefit.csv", index=False)
    sens.to_csv(reports / "sensitivity.csv", index=False)
    otp_sens.to_csv(reports / "otp_sensitivity.csv", index=False)
    val_curve, test_curve = costs.cost_curve(yv, sv, av, ccfg), costs.cost_curve(ys, ss, as_, ccfg)
    chosen = {"validation": costs.best_threshold(val_curve), "test": {"flagged": int((ss >= point).sum()), "cost": next(r for r in strategies if r["strategy"].endswith("(point)"))["cost"]}}
    plots.save_figure(plots.plot_cost_curves({"validation": val_curve, "test": test_curve}, chosen, {"validation": len(yv), "test": len(ys)}), figures / "cost_curves.png")
    plots.save_figure(plots.plot_tiers(test_tiers), figures / "risk_tiers.png")
    plots.save_figure(plots.plot_threshold_stability(boot, robust), figures / "threshold_stability.png")
    summary = {"split": method or cfg.split.method, "model_variant": sup["meta"]["model_variant"], "validation_rows": int(len(yv)), "validation_frauds": int(yv.sum()),
               "test_rows": int(len(ys)), "test_frauds": int(ys.sum()),
               "thresholds": {"block_point": point, "block_bootstrap_median": robust, "bootstrap_spread": stability, "f2": f2_t, "precision_0.85": p85_t,
                              "tier_medium_otp": tiers["t_medium"], "tier_high_block": tiers["t_high"], "test_optimal_best_case": oracle},
               "recommended": recommended, "saved_share_interval": interval, "validation_tiers": val_tiers.to_dict("records"),
               "test_tiers": test_tiers.to_dict("records"), "strategies": strategies, "sensitivity": sens.to_dict("records"), "otp_sensitivity": otp_sens.to_dict("records"),
               "costs": {"fn_fixed": ccfg.fn_fixed, "fp_cost": ccfg.fp_cost, "tp_review_cost": ccfg.tp_review_cost, "otp_cost": ccfg.otp_cost,
                         "otp_fraud_catch_rate": ccfg.otp_fraud_catch_rate}}
    (reports / f"{tcfg.report_name}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (reports / "cost_benefit.md").write_text(markdown_report(summary, cfg), encoding="utf-8")
    cfg.paths.models_path.mkdir(parents=True, exist_ok=True)
    (cfg.paths.models_path / tcfg.policy_name).write_text(json.dumps(
        {"model_variant": sup["meta"]["model_variant"], "features": names, "split": method or cfg.split.method,
         "block_threshold": robust, "otp_threshold": tiers["t_medium"], "high_threshold": tiers["t_high"],
         "tiers": {"low": "approve", "medium": "OTP step-up check", "high": "block and send to review"}, "costs": summary["costs"]},
        indent=2, default=float), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Threshold optimisation, risk tiers and cost-benefit (Step 8).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--split", choices=preprocess.METHODS, default=None, help="which split to use (default: the configured one)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    run(replace(base, paths=replace(base.paths, root=args.root)), method=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
