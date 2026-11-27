"""Step 4 pipeline: compare imbalance strategies with cross-validation on the training data, then check them on validation.

Test data is not touched here. The strategy that wins on cross-validated PR-AUC is recorded for Step 5.
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
from sklearn.metrics import precision_recall_curve

from fraud import costs, imbalance, metrics, plots, preprocess
from fraud.config import ProjectConfig, SEED
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def _arrays(part: pd.DataFrame, cfg: ProjectConfig):
    return (part[preprocess.feature_names(cfg.data, cfg.prep)].to_numpy(dtype=float), part[cfg.data.label_column].to_numpy(),
            part[cfg.prep.raw_amount_name].to_numpy())


def subsample_train(train: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """All frauds plus ``sample_genuine`` random genuine rows (for the quick check)."""
    label = cfg.data.label_column
    genuine = train[train[label] == 0]
    keep = genuine.sample(n=min(cfg.imbalance.sample_genuine, len(genuine)), random_state=SEED)
    return pd.concat([train[train[label] == 1], keep]).sort_index()


def evaluate_strategy(strategy: imbalance.Strategy, train: pd.DataFrame, val: pd.DataFrame, cfg: ProjectConfig) -> Dict[str, Any]:
    """Cross-validate on train, fit on all of train, and score validation (threshold chosen from the out-of-fold scores only)."""
    Xt, yt, at = _arrays(train, cfg)
    Xv, yv, av = _arrays(val, cfg)
    cv = imbalance.cross_validate_strategy(strategy, Xt, yt, cfg.split.cv_folds, cfg.imbalance)
    threshold = costs.best_threshold(costs.cost_curve(yt, cv["oof_scores"], at, cfg.costs))["threshold"]
    model = imbalance.fit_strategy(strategy, Xt, yt, cfg.imbalance)
    score = model.predict_proba(Xv)[:, 1]
    at_floor = metrics.recall_at_precision(yv, score, cfg.targets.precision_floor)
    flagged = (score >= threshold).astype(int)
    model_cost = costs.total_cost(yv, flagged, av, cfg.costs)["total"]
    base = costs.baseline_costs(yv, av, cfg.costs)
    precision, recall, _ = precision_recall_curve(yv, score)
    row = {"strategy": strategy.name, "description": strategy.description, "cv_pr_auc_mean": round(cv["pr_auc_mean"], 4),
           "cv_pr_auc_std": round(cv["pr_auc_std"], 4), "cv_fold_pr_auc": cv["fold_pr_auc"],
           "val_pr_auc": round(metrics.pr_auc(yv, score), 4), "val_recall_at_precision_0.85": round(at_floor["recall"], 4),
           "oof_threshold": round(float(threshold), 6), "val_flagged": int(flagged.sum()), "val_frauds_caught": int(((flagged == 1) & (yv == 1)).sum()),
           "val_cost": round(model_cost, 2), "val_cost_saved_vs_approve_all": costs.savings(model_cost, base)["saved"], "seconds": cv["seconds"]}
    return {"row": row, "curve": (precision, recall)}


def run(cfg: ProjectConfig, sample: bool = False, method: Optional[str] = None,
        strategies: Optional[Sequence[imbalance.Strategy]] = None) -> Dict[str, Any]:
    """Compare every strategy and write the table, the winner and two charts.

    Raises:
        FileNotFoundError: If Step 3 has not been run.
    """
    parts = preprocess.load_split(cfg, method)
    train, val = parts["train"], parts["validation"]
    if sample:
        train = subsample_train(train, cfg)
        cfg = replace(cfg, split=replace(cfg.split, cv_folds=3))
    rows, curves = [], {}
    for strategy in strategies or imbalance.default_strategies(cfg.imbalance):
        result = evaluate_strategy(strategy, train, val, cfg)
        rows.append(result["row"])
        curves[strategy.name] = result["curve"]
        LOGGER.info("%-12s CV PR-AUC %.3f +- %.3f | val PR-AUC %.3f | %.0fs", strategy.name, result["row"]["cv_pr_auc_mean"],
                    result["row"]["cv_pr_auc_std"], result["row"]["val_pr_auc"], result["row"]["seconds"])
    verdict = imbalance.pick_winner(rows)
    table = pd.DataFrame(rows)
    suffix = "_sample" if sample else ""
    reports = cfg.paths.reports_path
    reports.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["cv_fold_pr_auc"]).to_csv(reports / f"{cfg.imbalance.report_name}{suffix}.csv", index=False)
    plots.save_figure(plots.plot_strategy_cv(table, verdict["winner"]), reports / "figures" / f"imbalance_cv{suffix}.png")
    plots.save_figure(plots.plot_pr_curves(curves, float(val[cfg.data.label_column].mean())), reports / "figures" / f"imbalance_pr_curves{suffix}.png")
    summary = {"sample": sample, "split": method or cfg.split.method, "folds": cfg.split.cv_folds, "train_rows": int(len(train)),
               "train_frauds": int(train[cfg.data.label_column].sum()), "validation_frauds": int(val[cfg.data.label_column].sum()),
               "validation_approve_all_cost": round(costs.baseline_costs(val[cfg.data.label_column].to_numpy(), val[cfg.prep.raw_amount_name].to_numpy(), cfg.costs)["approve_all"], 2),
               **verdict, "strategies": rows}
    (reports / f"{cfg.imbalance.report_name}{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Compare imbalance strategies (Step 4).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--split", choices=preprocess.METHODS, default=None, help="which split to use (default: the configured one)")
    parser.add_argument("--sample", action="store_true", help="quick check: all frauds + 40,000 genuine rows, 3 folds")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    summary = run(replace(base, paths=replace(base.paths, root=args.root)), sample=args.sample, method=args.split)
    LOGGER.info("winner: %s (tied: %s)", summary["winner"], summary["tied_with_winner"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
