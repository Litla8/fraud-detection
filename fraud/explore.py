"""Step 2 pipeline: run the exploration, save the charts and ``reports/eda_summary.json``."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from fraud import data, eda, plots
from fraud.config import ProjectConfig
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
DPI = 150


def analyse(frame: pd.DataFrame, cfg: ProjectConfig, figures_dir: Path, tables_dir: Path) -> Dict[str, Any]:
    """Compute every table, save every chart and table file, and return the summary (no file reading here)."""
    d = cfg.data
    eda.require_columns(frame, d)
    summary = eda.build_summary(frame, cfg.split, d)
    bands = eda.amount_bands(frame, d)
    hourly = eda.hourly_pattern(frame, d)
    separation = eda.feature_separation(frame, d)
    drift = eda.split_drift(frame, cfg.split, d)
    top = separation.head(6)["feature"].tolist()
    plots.save_figure(plots.plot_class_balance(summary["rows"], summary["frauds"]), figures_dir / "class_balance.png", DPI)
    plots.save_figure(plots.plot_amounts(frame, bands, d), figures_dir / "amount_distribution.png", DPI)
    plots.save_figure(plots.plot_time_patterns(frame, hourly, d), figures_dir / "time_patterns.png", DPI)
    plots.save_figure(plots.plot_feature_separation(separation), figures_dir / "feature_separation.png", DPI)
    plots.save_figure(plots.plot_top_distributions(frame, top, d), figures_dir / "top_feature_distributions.png", DPI)
    plots.save_figure(plots.plot_drift(drift), figures_dir / "split_drift.png", DPI)
    plots.save_figure(plots.plot_correlation(frame, separation.head(10)["feature"].tolist(), d), figures_dir / "correlation_top_features.png", DPI)
    tables_dir.mkdir(parents=True, exist_ok=True)
    separation.to_csv(tables_dir / "feature_separation.csv", index=False)
    drift.to_csv(tables_dir / "split_drift.csv", index=False)
    return summary


def run(cfg: ProjectConfig) -> Dict[str, Any]:
    """Load the data and write everything.

    Raises:
        FileNotFoundError: If the Kaggle file is not in ``data/raw``.
    """
    frame = data.load_raw(cfg.paths.raw_path, cfg.data)
    figures_dir = cfg.paths.reports_path / "figures"
    summary = analyse(frame, cfg, figures_dir, cfg.paths.reports_path)
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_path / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s and 7 charts in %s", cfg.paths.reports_path / "eda_summary.json", figures_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Exploratory data analysis (Step 2).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    run(replace(base, paths=replace(base.paths, root=args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
