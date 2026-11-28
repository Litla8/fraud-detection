"""Collect every report the pipeline wrote into one readable results document (``reports/RESULTS.md``).

Idempotent and tolerant: a section whose report does not exist yet says which command creates it, so the document can be regenerated
after any step and always matches the numbers on disk.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from fraud.config import ProjectConfig
from fraud.logging_utils import configure_logging
from fraud.optimize_threshold import markdown_table

LOGGER = logging.getLogger(__name__)


def _load(reports: Path, name: str) -> Optional[Dict[str, Any]]:
    path = reports / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _missing(command: str) -> List[str]:
    return [f"_Not available yet: run `{command}`._", ""]


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _percent_columns(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Show share columns as percentages (``0.6939`` -> ``69.4%``)."""
    out = table.copy()
    for column in columns:
        if column in out:
            out[column] = out[column].map(lambda v: f"{100 * v:.2f}%" if abs(v) < 1 and column != "saved_share" else f"{100 * v:.1f}%")
    return out


def scorecard(models: Optional[Dict[str, Any]], thresholds: Optional[Dict[str, Any]], cfg: ProjectConfig) -> List[str]:
    """The three success metrics from the brief against what was achieved."""
    t = cfg.targets
    lines = ["## Scorecard", "", "| Target | Required | Achieved | Met |", "|---|---|---|---|"]
    if models:
        best = next(m for m in models["models"] if m["model_variant"] == models["winner"])
        lines.append(f"| PR-AUC on the untouched test set | above {t.pr_auc} | {best['test_pr_auc']:.3f} (95% interval {best['test_pr_auc_ci_low']:.3f} to "
                     f"{best['test_pr_auc_ci_high']:.3f}), {best['model_variant']} | {'yes' if best['meets_pr_auc_target'] else 'no'} |")
        lines.append(f"| Recall at precision of at least {t.precision_floor} | at least {t.recall} | {best['test_recall_at_precision_0.85']:.3f} "
                     f"(threshold picked on test: best case); {best['test_recall_transferred']:.3f} with a validation-picked threshold | "
                     f"{'yes' if best['meets_recall_target'] else 'no'} |")
    else:
        lines.append("| PR-AUC / recall | | not available yet | |")
    if thresholds:
        r = thresholds["recommended"]
        lines.append(f"| Documented cost savings | against approving everything | {_pct(r['saved_share'])} saved (EUR {r['saved']:,.0f} of EUR {r['approve_all_cost']:,.0f}; "
                     f"95% interval {_pct(thresholds['saved_share_interval']['low'])} to {_pct(thresholds['saved_share_interval']['high'])}) | yes |")
    else:
        lines.append("| Documented cost savings | | not available yet | |")
    return [*lines, ""]


def section_problem(problem: Optional[Dict[str, Any]], prep: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["## 1. The problem", ""]
    if not problem:
        return [*lines, *_missing("python -m fraud.define")]
    d, c = problem["data"], problem["cost_matrix"]
    lines += [f"- {d['rows']:,} transactions over about {d['hours_covered']:.0f} hours, {d['frauds']} frauds ({100 * d['fraud_rate']:.3f}%, 1 in {d['imbalance_ratio']:.0f}).",
              f"- A model that never flags anything is {100 * problem['why_accuracy_fails']['always_genuine_accuracy']:.2f}% accurate, so accuracy is not used; PR-AUC is the headline metric.",
              f"- Cost matrix (assumptions): a missed fraud costs its amount plus EUR {c['fn_fixed']:g}, a false alarm EUR {c['fp_cost']:g}, a reviewed fraud EUR {c['tp_review_cost']:g}, an OTP check EUR {c['otp_cost']:g}.",
              f"- Approving everything costs EUR {problem['baselines']['approve_everything']:,.0f}; blocking everything EUR {problem['baselines']['block_everything']:,.0f}."]
    if prep:
        dup = prep["duplicates"]
        lines.append(f"- {dup['rows_dropped']:,} exact duplicate rows ({dup['frauds_dropped']} frauds) were dropped before splitting, so a copy can never sit in both train and test.")
    return [*lines, ""]


def section_imbalance(data: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["## 2. Handling the imbalance", ""]
    if not data:
        return [*lines, *_missing("python -m fraud.run_imbalance")]
    table = pd.DataFrame(data["strategies"])
    table["cv_pr_auc"] = table.apply(lambda r: f"{r['cv_pr_auc_mean']:.3f} +- {r['cv_pr_auc_std']:.3f}", axis=1)
    table = table.sort_values("cv_pr_auc_mean", ascending=False)[["strategy", "cv_pr_auc", "val_pr_auc", "val_cost_saved_vs_approve_all"]]
    tied = ", ".join(data["tied_with_winner"]) or "none"
    return [*lines, f"Cross-validated PR-AUC on the training period ({data['folds']} folds; resampling only inside folds). Best: **{data['winner']}**; within one standard deviation: {tied}.",
            "", markdown_table(table), ""]


def section_models(data: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["## 3. Supervised models", ""]
    if not data:
        return [*lines, *_missing("python -m fraud.train_supervised")]
    table = pd.DataFrame(data["models"])
    table["test_pr_auc_interval"] = table.apply(lambda r: f"{r['test_pr_auc']:.3f} ({r['test_pr_auc_ci_low']:.3f} to {r['test_pr_auc_ci_high']:.3f})", axis=1)
    table["cv_pr_auc"] = table.apply(lambda r: f"{r['cv_pr_auc_mean']:.3f} +- {r['cv_pr_auc_std']:.3f}", axis=1)
    table = table.sort_values("cv_pr_auc_mean", ascending=False)[["model_variant", "cv_pr_auc", "test_pr_auc_interval", "test_recall_at_precision_0.85", "test_recall_transferred"]]
    top = ", ".join(f"{k} {100 * v:.0f}%" for k, v in list(data["importance_top10"].items())[:5])
    return [*lines, f"Chosen by cross-validation only: **{data['winner']}** (statistically tied: {', '.join(data['tied_with_winner']) or 'none'}). The test set holds {data['test_frauds']} frauds, "
            "so read each test score with its interval.", "", markdown_table(table), "",
            f"Built-in importance of the chosen model (how often a feature is used in splits): {top}. SHAP values in section 6 show how much each feature moves the score and are the better guide.", ""]


def section_anomaly(data: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["## 4. Anomaly detection and the hybrid", ""]
    if not data:
        return [*lines, *_missing("python -m fraud.train_anomaly")]
    table = pd.DataFrame(data["scorers"])
    table["test_pr_auc_interval"] = table.apply(lambda r: f"{r['test_pr_auc']:.3f} ({r['test_pr_auc_ci_low']:.3f} to {r['test_pr_auc_ci_high']:.3f})", axis=1)
    table = table[["scorer", "val_pr_auc", "test_pr_auc_interval", "test_cost_saved"]]
    choice = data["blend_choice"]
    rescue = ", ".join(f"{name}: {r['rescued']} of {r['missed_by_supervised']} missed frauds for {r['extra_false_alarms']} extra false alarms" for name, r in data["rescue"].items())
    supervised, hybrid = data["scorers"][0], data["scorers"][-1]
    if choice["adopted"]:
        gain = hybrid["test_pr_auc"] - supervised["test_pr_auc"]
        half_width = (supervised["test_pr_auc_ci_high"] - supervised["test_pr_auc_ci_low"]) / 2
        size = "well inside" if abs(gain) < half_width / 2 else "comparable to"
        verdict = (f"The best blend ({choice['detector']}, weight {choice['weight']}) was selected on validation (gain {choice['val_gain']:.4f} PR-AUC). On test it scores "
                   f"{hybrid['test_pr_auc']:.3f} against {supervised['test_pr_auc']:.3f} for the supervised model alone; that {gain:+.3f} difference is {size} the test interval. "
                   "The thresholds, tiers, API and Docker image (sections 5 and 6) use the supervised model alone: adding the detectors to the service is optional follow-up work.")
    else:
        verdict = f"No blend added the required validation PR-AUC (best gain {choice['val_gain']:.4f}), so the hybrid is the supervised model alone."
    return [*lines, "Detectors learn from genuine transactions only. " + verdict, "", markdown_table(table), "", f"Rescue of frauds the supervised model misses: {rescue}.", ""]


def section_thresholds(data: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["## 5. Thresholds, risk tiers and cost-benefit", ""]
    if not data:
        return [*lines, *_missing("python -m fraud.optimize_threshold")]
    table = _percent_columns(pd.DataFrame(data["strategies"])[["strategy", "blocked", "otp_checks", "frauds_stopped", "cost", "saved", "saved_share"]], ["saved_share"])
    t = data["thresholds"]
    otp = pd.DataFrame(data["otp_sensitivity"])
    weakest = otp[(otp["otp_catch_rate"] == otp["otp_catch_rate"].min()) & (otp["otp_cost"] == otp["otp_cost"].max())].iloc[0]
    rec, tiers = data["recommended"], _percent_columns(pd.DataFrame(data["test_tiers"]), ["share_of_traffic", "fraud_rate"])
    return [*lines, f"Thresholds were chosen on validation data and applied unchanged to the {data['test_rows']:,} test transactions ({data['test_frauds']} frauds). "
            f"Tiers: approve below {t['tier_medium_otp']:.4f}, OTP check from there, block and review from {t['tier_high_block']:.4f}.", "", markdown_table(table), "",
            f"The tiered policy saves {_pct(rec['saved_share'])} (95% interval {_pct(data['saved_share_interval']['low'])} to {_pct(data['saved_share_interval']['high'])}), "
            f"about EUR {rec['saved_per_million']:,.0f} per million transactions. Under the weakest OTP assumptions (catch {weakest['otp_catch_rate']:.0%}, cost EUR {weakest['otp_cost']:g}) "
            f"it saves {_pct(weakest['tiers_saved_share'])} against {_pct(weakest['block_saved_share'])} for block-only, so the tier advantage depends on how well OTP works.", "",
            "Test-set tiers:", "", markdown_table(tiers), ""]


def section_service(explanations: Optional[Dict[str, Any]], latency: Optional[Dict[str, Any]], drift: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["## 6. Explanations, service and monitoring", ""]
    if explanations:
        top = ", ".join(f"{r['feature']} {100 * r['share']:.0f}%" for r in explanations["top_features"][:5])
        lines += [f"- Explanations ({explanations['method']}): top features {top}. Features are anonymised PCA components, so reasons are statistical, not business-readable."]
    else:
        lines += [f"- Explanations: run `python -m fraud.explain`."]
    if latency:
        parts = [f"{name.replace('_', ' ')} p99 {v['p99']} ms" for name, v in latency.items() if isinstance(v, dict) and "p99" in v]
        lines += [f"- Latency against the {latency['budget_ms']:g} ms budget: {', '.join(parts)}: {'within budget' if latency['within_budget'] else 'OVER BUDGET'}."]
    else:
        lines += ["- Latency: run `python -m fraud.benchmark`."]
    if drift:
        windows = drift["windows"]
        last = list(windows.items())[-1]
        lines += [f"- Drift against the training period (PSI; {drift['thresholds']['warn']} warn, {drift['thresholds']['alert']} alert): "
                  + "; ".join(f"{name}: {w['overall']} ({w['features_alert']} features alert)" for name, w in windows.items()) + f". Worst features in the last window: "
                  + ", ".join(f"{f['feature']} {f['psi']:.2f}" for f in last[1]["top_features"][:3]) + "."]
    else:
        lines += ["- Drift: run `python -m fraud.drift`."]
    return [*lines, ""]


LIMITATIONS = ["## Limitations", "",
               "- The test set has only a few dozen frauds, so every test score and saving carries a wide interval; differences smaller than the interval are noise.",
               "- The cost matrix, including the OTP catch rate and cost, is an assumption to confirm with the business; the sensitivity tables show how results move.",
               "- The file covers two days of one card network, so nothing here says how the model behaves over months; drift monitoring exists for that reason.",
               "- The features are anonymised PCA components and there is no customer or merchant column, so behavioural features (the brief's Step 7) were not possible.",
               "- The service has no authentication, rate limiting or persistent logging: it demonstrates the scoring path, not a production deployment.", ""]


def build(cfg: ProjectConfig) -> str:
    """The whole document as text, from whatever reports exist."""
    r = cfg.paths.reports_path
    problem, prep = _load(r, "problem_definition.json"), _load(r, "preprocessing_report.json")
    models, anomaly, thresholds = _load(r, "model_comparison.json"), _load(r, "anomaly_comparison.json"), _load(r, "threshold_report.json")
    lines = ["# Credit card fraud detection: results", "",
             f"Generated from the reports in `{cfg.paths.reports_dir}/` by `python -m fraud.report`. Split: {cfg.split.method}; scaled Time as a feature: {cfg.prep.use_time}.", "",
             *scorecard(models, thresholds, cfg), *section_problem(problem, prep), *section_imbalance(_load(r, "imbalance_comparison.json")), *section_models(models),
             *section_anomaly(anomaly), *section_thresholds(thresholds),
             *section_service(_load(r, "explanations.json"), _load(r, "latency_report.json"), _load(r, "drift_report.json")), *LIMITATIONS]
    return "\n".join(lines)


def run(cfg: ProjectConfig) -> Path:
    """Write ``reports/RESULTS.md`` and return its path."""
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.reports_path / "RESULTS.md"
    path.write_text(build(cfg), encoding="utf-8")
    LOGGER.info("wrote %s", path)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Collect all reports into reports/RESULTS.md.")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    run(replace(base, paths=replace(base.paths, root=args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
