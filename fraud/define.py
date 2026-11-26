"""Step 1: define the problem, the costs and the targets, and write ``reports/problem_definition.json``.

The numbers come from the real file: how rare fraud is, what "do nothing" would cost under the cost matrix, and what a
useless always-genuine model scores on accuracy. Every later step is judged against this file.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from fraud import costs, data, metrics
from fraud.config import ProjectConfig
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def build_definition(frame: pd.DataFrame, cfg: ProjectConfig) -> Dict[str, Any]:
    """The problem definition as a JSON-ready dict (no file access)."""
    d = cfg.data
    y, amount = frame[d.label_column].to_numpy(), frame[d.amount_column].to_numpy()
    base = costs.baseline_costs(y, amount, cfg.costs)
    prof = data.profile(frame, d)
    per_fraud = base["approve_all"] / max(prof["frauds"], 1)
    return {
        "problem": "Flag fraudulent card transactions in real time while keeping false alarms low.",
        "data": prof,
        "why_accuracy_fails": {"always_genuine_accuracy": round(metrics.always_genuine_accuracy(y), 5),
                               "frauds_caught": 0, "note": "A model that never flags anything looks 99.8% accurate."},
        "cost_matrix": {**asdict(cfg.costs), "currency": "EUR (dataset units)",
                        "average_cost_per_missed_fraud": round(per_fraud, 2),
                        "break_even_precision": round(per_fraud / (per_fraud + cfg.costs.fp_cost), 4),
                        "note": "Assumptions to confirm with the business; change them in fraud/config.py."},
        "baselines": {"approve_everything": round(base["approve_all"], 2), "block_everything": round(base["block_all"], 2)},
        "targets": {"pr_auc_above": cfg.targets.pr_auc, "recall_at_least": cfg.targets.recall,
                    "at_precision_at_least": cfg.targets.precision_floor, "documented_cost_savings": True},
        "metrics": ["PR-AUC (headline)", "recall at fixed precision", "F2", "MCC", "total business cost"],
        "split": asdict(cfg.split),
    }


def run(cfg: ProjectConfig) -> Dict[str, Any]:
    """Load the data, build the definition and write it.

    Raises:
        FileNotFoundError: If the Kaggle file is not in ``data/raw``.
    """
    frame = data.load_raw(cfg.paths.raw_path, cfg.data)
    LOGGER.info("loaded %d transactions (%d frauds)", len(frame), int(frame[cfg.data.label_column].sum()))
    report = build_definition(frame, cfg)
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    cfg.paths.problem_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", cfg.paths.problem_report_path)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Define the problem and costs (Step 1).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    report = run(replace(base, paths=replace(base.paths, root=args.root)))
    LOGGER.info("fraud rate %.3f%%; approving everything would cost %.0f; break-even precision %.2f",
                100 * report["data"]["fraud_rate"], report["baselines"]["approve_everything"],
                report["cost_matrix"]["break_even_precision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
