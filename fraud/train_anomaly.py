"""Step 6 pipeline: fit the anomaly detectors on genuine training rows, compare them with the supervised model, build the hybrid.

Order of trust: detectors learn from training genuine rows only; the validation set sets the percentile yardstick, the blend weight
and the cost threshold; the test set is used once, for the reported numbers.
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

from fraud import anomaly, costs, metrics, plots, preprocess
from fraud.config import ProjectConfig
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def load_supervised(cfg: ProjectConfig) -> Dict[str, Any]:
    """The Step 5 model and its metadata.

    Raises:
        FileNotFoundError: If Step 5 has not been run (full run, not ``--sample``).
        ValueError: If the saved feature list does not match the current preprocessing.
    """
    models_dir = cfg.paths.models_path
    meta_path, model_path = models_dir / "supervised_best.json", models_dir / "supervised_best.joblib"
    if not meta_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Step 5 model not found: run `python -m fraud.train_supervised` (without --sample) first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["features"] != preprocess.feature_names(cfg.data, cfg.prep):
        raise ValueError("the saved model was trained on different features: rerun Step 5")
    return {"model": joblib.load(model_path), "meta": meta}


def score_row(label: str, y_val: np.ndarray, s_val: np.ndarray, y_test: np.ndarray, s_test: np.ndarray, amount_val: np.ndarray,
              amount_test: np.ndarray, cfg: ProjectConfig) -> Dict[str, Any]:
    """The same metrics for any scorer: PR-AUC with interval, recall at the precision floor, and the cost with a validation threshold."""
    floor = cfg.targets.precision_floor
    own = metrics.recall_at_precision(y_test, s_test, floor)
    carried = metrics.apply_threshold(y_test, s_test, metrics.recall_at_precision(y_val, s_val, floor)["threshold"])
    cost_threshold = costs.best_threshold(costs.cost_curve(y_val, s_val, amount_val, cfg.costs))["threshold"]
    test_cost = costs.total_cost(y_test, (s_test >= cost_threshold).astype(int), amount_test, cfg.costs)["total"]
    approve_all = costs.baseline_costs(y_test, amount_test, cfg.costs)["approve_all"]
    ci = metrics.bootstrap_pr_auc(y_test, s_test, cfg.supervised.bootstrap_resamples)
    test_pr = metrics.pr_auc(y_test, s_test)
    return {"scorer": label, "val_pr_auc": round(metrics.pr_auc(y_val, s_val), 4), "test_pr_auc": round(test_pr, 4),
            "test_pr_auc_ci_low": round(ci["low"], 4), "test_pr_auc_ci_high": round(ci["high"], 4),
            "test_recall_at_precision_0.85": round(own["recall"], 4), "test_recall_transferred": round(carried["recall"], 4),
            "test_precision_transferred": round(carried["precision"], 4), "cost_threshold_val": float(cost_threshold),
            "test_cost": round(test_cost, 2), "test_cost_saved": costs.savings(test_cost, {"approve_all": approve_all})["saved"],
            "meets_pr_auc_target": bool(test_pr > cfg.targets.pr_auc), "meets_recall_target": bool(own["recall"] >= cfg.targets.recall)}


def choose_blend(grid: pd.DataFrame, min_gain: float) -> Dict[str, Any]:
    """Pick the (detector, weight) with the best validation PR-AUC; keep the supervised model alone unless the gain is at least ``min_gain``.

    ``grid`` has columns detector, weight, val_pr_auc and must contain weight 0 (the supervised model).
    """
    if grid.empty or not (grid["weight"] == 0).any():
        raise ValueError("grid must contain weight 0")
    base = float(grid.loc[grid["weight"] == 0, "val_pr_auc"].max())
    best = grid.sort_values(["val_pr_auc", "weight"], ascending=[False, True]).iloc[0]
    gain = float(best["val_pr_auc"]) - base
    adopted = bool(best["weight"] > 0 and gain >= min_gain)
    return {"detector": str(best["detector"]), "weight": float(best["weight"]) if adopted else 0.0,
            "best_weight_seen": float(best["weight"]), "val_gain": round(gain, 4), "adopted": adopted}


def run(cfg: ProjectConfig, sample: bool = False, method: Optional[str] = None, only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Fit and compare the detectors, blend with the Step 5 model, save the report, charts and (full run) the hybrid.

    Raises:
        FileNotFoundError: If Step 3 or Step 5 outputs are missing.
        ValueError: If no detector is requested.
    """
    acfg = cfg.anomaly
    parts = preprocess.load_split(cfg, method)
    train, val, test = parts["train"], parts["validation"], parts["test"]
    names = preprocess.feature_names(cfg.data, cfg.prep)
    label, raw = cfg.data.label_column, cfg.prep.raw_amount_name
    sup = load_supervised(cfg)
    wanted = [m for m in acfg.models if not only or m in only]
    if not wanted:
        raise ValueError("no detector requested")
    Xv, yv, av = val[names].to_numpy(float), val[label].to_numpy(), val[raw].to_numpy()
    Xs, ys, as_ = test[names].to_numpy(float), test[label].to_numpy(), test[raw].to_numpy()
    genuine = train.loc[train[label] == 0, names].to_numpy(float)
    if sample:
        genuine = anomaly.subsample_rows(genuine, acfg.sample_normal)
        cfg = replace(cfg, supervised=replace(cfg.supervised, bootstrap_resamples=50))
    sup_val, sup_test = sup["model"].predict_proba(Xv)[:, 1], sup["model"].predict_proba(Xs)[:, 1]
    rows: List[Dict[str, Any]] = [score_row(f"{sup['meta']['model_variant']} (supervised)", yv, sup_val, ys, sup_test, av, as_, cfg)]
    fitted, raw_test, pct_val, pct_test, grid = {}, {}, {}, {}, []
    for name in wanted:
        start = time.perf_counter()
        det = anomaly.fit_detector(name, genuine, acfg)
        s_val, s_test = det.score(Xv), det.score(Xs)
        reference = anomaly.reference_distribution(s_val[yv == 0])
        fitted[name] = {"detector": det, "reference": reference}
        pct_val[name], pct_test[name], raw_test[name] = anomaly.to_percentile(reference, s_val), anomaly.to_percentile(reference, s_test), s_test
        row = score_row(f"{name} (anomaly only)", yv, s_val, ys, s_test, av, as_, cfg)
        rows.append(row)
        for w in acfg.blend_weights:
            grid.append({"detector": name, "weight": w, "val_pr_auc": round(metrics.pr_auc(yv, anomaly.blend_scores(sup_val, pct_val[name], w)), 4)})
        LOGGER.info("%-18s val PR-AUC %.3f | test PR-AUC %.3f [%.3f, %.3f] | %.0fs", name, row["val_pr_auc"], row["test_pr_auc"],
                    row["test_pr_auc_ci_low"], row["test_pr_auc_ci_high"], time.perf_counter() - start)
    grid_table = pd.DataFrame(grid)
    choice = choose_blend(grid_table, acfg.min_gain)
    name, weight = choice["detector"], choice["weight"]
    hybrid_val = anomaly.blend_scores(sup_val, pct_val[name], weight)
    hybrid_test = anomaly.blend_scores(sup_test, pct_test[name], weight)
    hybrid_row = score_row(f"hybrid: supervised + {name} (w={weight})" if choice["adopted"] else "hybrid = supervised (no gain)",
                           yv, hybrid_val, ys, hybrid_test, av, as_, cfg)
    rows.append(hybrid_row)
    supervised_cut = metrics.recall_at_precision(yv, sup_val, cfg.targets.precision_floor)["threshold"]
    rescue = {n: anomaly.rescue_analysis(ys, sup_test, supervised_cut, pct_test[n], acfg.rescue_quantile) for n in wanted}
    LOGGER.info("hybrid: best detector %s | adopted=%s | test PR-AUC %.3f vs supervised %.3f", name, choice["adopted"],
                hybrid_row["test_pr_auc"], rows[0]["test_pr_auc"])
    for n, r in rescue.items():
        LOGGER.info("rescue %-18s %d of %d frauds missed by the supervised model alerted, for %d extra false alarms", n, r["rescued"],
                    r["missed_by_supervised"], r["extra_false_alarms"])
    suffix = "_sample" if sample else ""
    reports = cfg.paths.reports_path
    reports.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(reports / f"{acfg.report_name}{suffix}.csv", index=False)
    grid_table.to_csv(reports / f"blend_grid{suffix}.csv", index=False)
    figures = reports / "figures"
    plots.save_figure(plots.plot_anomaly_separation({n: (pct_test[n][ys == 0], pct_test[n][ys == 1]) for n in wanted}), figures / f"anomaly_separation{suffix}.png")
    plots.save_figure(plots.plot_blend_weights(grid_table), figures / f"blend_weights{suffix}.png")
    curves = {"supervised": precision_recall_curve(ys, sup_test)[:2], f"best detector: {name}": precision_recall_curve(ys, raw_test[name])[:2],
              "hybrid": precision_recall_curve(ys, hybrid_test)[:2]}
    plots.save_figure(plots.plot_pr_curves(curves, float(ys.mean()), "Precision-recall on the test set"), figures / f"hybrid_pr_curves{suffix}.png")
    if not sample:
        cfg.paths.models_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, cfg.paths.models_path / "anomaly_detectors.joblib")
        (cfg.paths.models_path / "hybrid.json").write_text(json.dumps(
            {"supervised": sup["meta"]["model_variant"], "detector": name, "weight": weight, "adopted": choice["adopted"],
             "features": names, "supervised_cost_threshold": rows[0]["cost_threshold_val"], "hybrid_cost_threshold": hybrid_row["cost_threshold_val"],
             "rescue_quantile": acfg.rescue_quantile, "split": method or cfg.split.method}, indent=2), encoding="utf-8")
    summary = {"sample": sample, "split": method or cfg.split.method, "genuine_rows_used": int(len(genuine)), "test_frauds": int(ys.sum()),
               "blend_choice": choice, "rescue": rescue, "scorers": rows, "blend_grid": grid}
    (reports / f"{acfg.report_name}{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Anomaly detectors and the hybrid model (Step 6).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--split", choices=preprocess.METHODS, default=None, help="which split to use (default: the configured one)")
    parser.add_argument("--models", nargs="+", default=None, help="only these detectors, e.g. isolation_forest autoencoder")
    parser.add_argument("--sample", action="store_true", help="quick check: 20,000 genuine rows, files get a _sample suffix, nothing saved to models/")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    run(replace(base, paths=replace(base.paths, root=args.root)), sample=args.sample, method=args.split, only=args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
