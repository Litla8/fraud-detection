"""Step 5 pipeline: train the four supervised models (with and without SMOTE), pick one by cross-validation, report on test.

Selection uses only cross-validated PR-AUC on training data. Validation supplies the threshold for the "transferred" recall /
precision; the test set is used once, for the numbers in the report, and never for choosing anything.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from fraud import costs, imbalance, metrics, models, plots, preprocess, run_imbalance
from fraud.config import ProjectConfig, SEED
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def cross_validate_model(name: str, variant: str, X: np.ndarray, y: np.ndarray, folds: int, cfg: ProjectConfig) -> Dict[str, Any]:
    """Stratified k-fold cross-validation (the variant is applied inside each fold); returns fold PR-AUCs and out-of-fold scores."""
    oof = np.zeros(len(y))
    scores: List[float] = []
    for train_idx, test_idx in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        model = models.fit_model(name, variant, X[train_idx], y[train_idx], cfg.supervised, cfg.imbalance)
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        scores.append(metrics.pr_auc(y[test_idx], oof[test_idx]))
    return {"oof": oof, "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)), "folds": [round(s, 4) for s in scores]}


def evaluate_model(name: str, variant: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: ProjectConfig) -> Dict[str, Any]:
    """Cross-validate, fit on all training rows, and score validation and test with thresholds chosen on training / validation only."""
    start = time.perf_counter()
    names = preprocess.feature_names(cfg.data, cfg.prep)
    label, raw = cfg.data.label_column, cfg.prep.raw_amount_name
    Xt, yt, at = train[names].to_numpy(float), train[label].to_numpy(), train[raw].to_numpy()
    Xv, yv, av = val[names].to_numpy(float), val[label].to_numpy(), val[raw].to_numpy()
    Xs, ys, as_ = test[names].to_numpy(float), test[label].to_numpy(), test[raw].to_numpy()
    cv = cross_validate_model(name, variant, Xt, yt, cfg.split.cv_folds, cfg)
    f2_threshold = metrics.best_f2_threshold(yt, cv["oof"])                                   # chosen on out-of-fold training scores
    cost_threshold = costs.best_threshold(costs.cost_curve(yt, cv["oof"], at, cfg.costs))["threshold"]
    model = models.fit_model(name, variant, Xt, yt, cfg.supervised, cfg.imbalance)
    sv, ss = model.predict_proba(Xv)[:, 1], model.predict_proba(Xs)[:, 1]
    floor = cfg.targets.precision_floor
    own = metrics.recall_at_precision(ys, ss, floor)                                            # best case: threshold picked on test itself
    carried = metrics.apply_threshold(ys, ss, metrics.recall_at_precision(yv, sv, floor)["threshold"])   # honest: threshold picked on validation
    f2 = metrics.apply_threshold(ys, ss, f2_threshold)
    ci = metrics.bootstrap_pr_auc(ys, ss, cfg.supervised.bootstrap_resamples)
    flagged = (ss >= cost_threshold).astype(int)
    test_cost = costs.total_cost(ys, flagged, as_, cfg.costs)["total"]
    approve_all = costs.baseline_costs(ys, as_, cfg.costs)["approve_all"]
    test_pr = metrics.pr_auc(ys, ss)
    row = {"model_variant": f"{name}/{variant}", "model": name, "variant": variant,
           "cv_pr_auc_mean": round(cv["mean"], 4), "cv_pr_auc_std": round(cv["std"], 4), "val_pr_auc": round(metrics.pr_auc(yv, sv), 4),
           "test_pr_auc": round(test_pr, 4), "test_pr_auc_ci_low": round(ci["low"], 4), "test_pr_auc_ci_high": round(ci["high"], 4),
           "test_recall_at_precision_0.85": round(own["recall"], 4),
           "test_recall_transferred": round(carried["recall"], 4), "test_precision_transferred": round(carried["precision"], 4),
           "f2_threshold": round(f2_threshold, 6), "test_f2": round(f2["f2"], 4), "test_mcc": round(f2["mcc"], 4),
           "test_precision_at_f2": round(f2["precision"], 4), "test_recall_at_f2": round(f2["recall"], 4),
           "test_tp_at_f2": f2["tp"], "test_fp_at_f2": f2["fp"], "test_fn_at_f2": f2["fn"],
           "cost_threshold": round(float(cost_threshold), 6), "test_cost": round(test_cost, 2), "test_approve_all_cost": round(approve_all, 2),
           "test_cost_saved": costs.savings(test_cost, {"approve_all": approve_all})["saved"],
           "meets_pr_auc_target": bool(test_pr > cfg.targets.pr_auc), "meets_recall_target": bool(own["recall"] >= cfg.targets.recall),
           "seconds": round(time.perf_counter() - start, 1)}
    return {"row": row, "model": model, "scores": ss, "fold_pr_auc": cv["folds"]}


def run(cfg: ProjectConfig, sample: bool = False, method: Optional[str] = None, only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Train and evaluate every available model and variant, save the table, charts and the best model.

    Raises:
        FileNotFoundError: If Step 3 has not been run.
        ValueError: If none of the requested models can be built.
    """
    parts = preprocess.load_split(cfg, method)
    train, val, test = parts["train"], parts["validation"], parts["test"]
    if sample:
        train = run_imbalance.subsample_train(train, cfg)
        cfg = replace(cfg, split=replace(cfg.split, cv_folds=3), supervised=replace(cfg.supervised, bootstrap_resamples=50))
    wanted = [m for m in cfg.supervised.models if not only or m in only]
    skipped = [m for m in wanted if not models.is_available(m)]
    for name in skipped:
        LOGGER.warning("skipping %s: package not installed", name)
    runnable = [m for m in wanted if m not in skipped]
    if not runnable:
        raise ValueError("none of the requested models can be built")
    rows, fitted, curves = [], {}, {}
    for name in runnable:
        for variant in cfg.supervised.variants:
            result = evaluate_model(name, variant, train, val, test, cfg)
            rows.append(result["row"])
            fitted[result["row"]["model_variant"]] = result["model"]
            LOGGER.info("%-28s CV %.3f +- %.3f | test PR-AUC %.3f [%.3f, %.3f] | %.0fs", result["row"]["model_variant"], result["row"]["cv_pr_auc_mean"],
                        result["row"]["cv_pr_auc_std"], result["row"]["test_pr_auc"], result["row"]["test_pr_auc_ci_low"],
                        result["row"]["test_pr_auc_ci_high"], result["row"]["seconds"])
            curves[result["row"]["model_variant"]] = result["scores"]
    verdict = imbalance.pick_winner(rows, key="model_variant")
    table = pd.DataFrame(rows)
    suffix = "_sample" if sample else ""
    reports = cfg.paths.reports_path
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / f"{cfg.supervised.report_name}{suffix}.csv", index=False)
    best_name = verdict["winner"]
    names = tuple(preprocess.feature_names(cfg.data, cfg.prep))
    importance = models.feature_importance(fitted[best_name], names)
    plots.save_figure(plots.plot_model_comparison(table, cfg.targets.pr_auc), reports / "figures" / f"model_comparison{suffix}.png")
    plots.save_figure(plots.plot_importance(importance, title=f"Feature importance: {best_name}"), reports / "figures" / f"feature_importance{suffix}.png")
    precisions = {}
    y_test = test[cfg.data.label_column].to_numpy()
    for key in table.sort_values("test_pr_auc", ascending=False).groupby("model").head(1)["model_variant"]:
        p, r, _ = precision_recall_curve(y_test, curves[key])
        precisions[key] = (p, r)
    plots.save_figure(plots.plot_pr_curves(precisions, float(y_test.mean())), reports / "figures" / f"model_pr_curves{suffix}.png")
    if not sample:
        cfg.paths.models_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[best_name], cfg.paths.models_path / "supervised_best.joblib")
        best_row = next(r for r in rows if r["model_variant"] == best_name)
        (cfg.paths.models_path / "supervised_best.json").write_text(json.dumps(
            {"model_variant": best_name, "features": list(names), "f2_threshold": best_row["f2_threshold"],
             "cost_threshold": best_row["cost_threshold"], "split": method or cfg.split.method}, indent=2), encoding="utf-8")
    summary = {"sample": sample, "split": method or cfg.split.method, "folds": cfg.split.cv_folds, "skipped_models": skipped,
               "train_frauds": int(train[cfg.data.label_column].sum()), "test_frauds": int(y_test.sum()),
               "targets": {"pr_auc_above": cfg.targets.pr_auc, "recall_at_least": cfg.targets.recall, "at_precision": cfg.targets.precision_floor},
               **verdict, "importance_top10": dict(list(importance.items())[:10]), "models": rows}
    (reports / f"{cfg.supervised.report_name}{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train and compare supervised models (Step 5).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--split", choices=preprocess.METHODS, default=None, help="which split to use (default: the configured one)")
    parser.add_argument("--models", nargs="+", default=None, help="only these models, e.g. logistic_regression xgboost")
    parser.add_argument("--sample", action="store_true", help="quick check: all frauds + 40,000 genuine rows, 3 folds, files get a _sample suffix")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    summary = run(replace(base, paths=replace(base.paths, root=args.root)), sample=args.sample, method=args.split, only=args.models)
    LOGGER.info("best by cross-validation: %s (tied: %s)", summary["winner"], summary["tied_with_winner"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
