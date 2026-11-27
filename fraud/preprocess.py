"""Step 3: remove duplicates, split (time-based and stratified), scale on the training rows only, and save.

The test set is only ever transformed with a scaler fitted on training rows, never resampled and never rebalanced.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from fraud import data, splits
from fraud.config import DataConfig, PreprocessConfig, ProjectConfig
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
METHODS = ("time", "stratified")
PARTS = ("train", "validation", "test")


def feature_names(data_cfg: DataConfig = DataConfig(), prep: PreprocessConfig = PreprocessConfig()) -> List[str]:
    """The model input columns, in order: V1-V28, scaled Amount, then scaled Time and hour features if enabled."""
    names = [*data_cfg.pca_columns, data_cfg.amount_column + prep.scaled_suffix]
    if prep.use_time:
        names.append(data_cfg.time_column + prep.scaled_suffix)
    if prep.hour_features:
        names += ["hour_sin", "hour_cos"]
    return names


def drop_duplicate_rows(frame: pd.DataFrame, enabled: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop exact duplicate rows (keeping the first) and report how many rows and frauds went; ``enabled=False`` keeps everything."""
    duplicated = frame.duplicated(keep="first") if enabled else pd.Series(False, index=frame.index)
    label = frame["Class"] if "Class" in frame.columns else pd.Series(0, index=frame.index)
    stats = {"rows_before": int(len(frame)), "rows_dropped": int(duplicated.sum()), "frauds_dropped": int(label[duplicated].sum())}
    return frame[~duplicated].copy(), stats


def fit_scaler(train: pd.DataFrame, prep: PreprocessConfig = PreprocessConfig()) -> RobustScaler:
    """RobustScaler (median and inter-quartile range, so huge amounts do not distort it) fitted on training rows only.

    Raises:
        ValueError: If a column to scale is missing or the training data is empty.
    """
    missing = [c for c in prep.scale_columns if c not in train.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if train.empty:
        raise ValueError("training data is empty")
    return RobustScaler().fit(train[list(prep.scale_columns)])


def transform(frame: pd.DataFrame, scaler: RobustScaler, cfg: ProjectConfig = ProjectConfig(), keep_row_id: bool = True) -> pd.DataFrame:
    """Model-ready frame: features, ``Class``, the raw amount and time (for costs and tracing) and ``row_id``.

    The input is not modified. Row order is preserved.
    """
    d, p = cfg.data, cfg.prep
    scaled = pd.DataFrame(scaler.transform(frame[list(p.scale_columns)]), index=frame.index,
                          columns=[c + p.scaled_suffix for c in p.scale_columns])
    out = pd.concat([frame[list(d.pca_columns)].astype("float64"), scaled], axis=1)
    if p.use_time is False:
        out = out.drop(columns=d.time_column + p.scaled_suffix, errors="ignore")
    if p.hour_features:
        angle = 2 * np.pi * ((frame[d.time_column] // 3600) % 24) / 24
        out["hour_sin"], out["hour_cos"] = np.sin(angle), np.cos(angle)
    out = out[feature_names(d, p)]
    out[d.label_column] = frame[d.label_column].astype("int8").to_numpy()
    out[p.raw_amount_name] = frame[d.amount_column].to_numpy()
    out[p.raw_time_name] = frame[d.time_column].to_numpy()
    if keep_row_id:
        out[p.id_column] = frame.index.to_numpy()
    return out


def check_split(parts: Dict[str, pd.DataFrame], method: str, cfg: ProjectConfig = ProjectConfig()) -> Dict[str, Any]:
    """Safety checks that must hold before any model sees the data; raises if one fails.

    Raises:
        ValueError: If parts overlap, a part has no fraud, or (time split) the parts are not in time order.
    """
    ids = {name: set(part[cfg.prep.id_column]) for name, part in parts.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if ids[a] & ids[b]:
            raise ValueError(f"{a} and {b} share rows")
    empty = [name for name, part in parts.items() if part[cfg.data.label_column].sum() == 0]
    if empty:
        raise ValueError(f"no fraud in: {empty}")
    ordered = None
    if method == "time":
        t = {name: (part[cfg.prep.raw_time_name].min(), part[cfg.prep.raw_time_name].max()) for name, part in parts.items()}
        ordered = bool(t["train"][1] <= t["validation"][0] and t["validation"][1] <= t["test"][0])
        if not ordered:
            raise ValueError("time split is not in chronological order")
    return {"disjoint": True, "every_part_has_fraud": True, "chronological": ordered}


def summarise_parts(parts: Dict[str, pd.DataFrame], cfg: ProjectConfig = ProjectConfig()) -> List[Dict[str, Any]]:
    """Rows, frauds and fraud rate per part."""
    label = cfg.data.label_column
    return [{"part": name, "rows": int(len(part)), "frauds": int(part[label].sum()), "fraud_rate": round(float(part[label].mean()), 6)}
            for name, part in parts.items()]


def build_split(frame: pd.DataFrame, method: str, cfg: ProjectConfig) -> Tuple[Dict[str, pd.DataFrame], RobustScaler]:
    """Split ``frame`` by ``method``, fit the scaler on train, and transform every part.

    Raises:
        ValueError: For an unknown method or a failed safety check.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    cut = splits.split_by_time if method == "time" else splits.split_stratified
    train, val, test = cut(frame, cfg.split, cfg.data)
    scaler = fit_scaler(train, cfg.prep)
    parts = {"train": transform(train, scaler, cfg), "validation": transform(val, scaler, cfg), "test": transform(test, scaler, cfg)}
    return parts, scaler


def write_split(parts: Dict[str, pd.DataFrame], scaler: RobustScaler, method: str, cfg: ProjectConfig) -> Dict[str, Path]:
    """Save the three parts and the scaler; returns their paths."""
    folder = cfg.paths.split_path(method)
    folder.mkdir(parents=True, exist_ok=True)
    cfg.paths.models_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, part in parts.items():
        paths[name] = folder / f"{name}{cfg.prep.file_suffix}"
        part.reset_index(drop=True).to_pickle(paths[name])
    paths["scaler"] = cfg.paths.scaler_path(method)
    joblib.dump(scaler, paths["scaler"])
    return paths


def load_split(cfg: ProjectConfig, method: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Read the saved train / validation / test parts for ``method`` (default: the configured one).

    Raises:
        FileNotFoundError: If Step 3 has not been run.
    """
    method = method or cfg.split.method
    folder = cfg.paths.split_path(method)
    missing = [n for n in PARTS if not (folder / f"{n}{cfg.prep.file_suffix}").is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {missing}. Run `python -m fraud.preprocess` first.")
    return {n: pd.read_pickle(folder / f"{n}{cfg.prep.file_suffix}") for n in PARTS}


def run(cfg: ProjectConfig, methods: Sequence[str] = METHODS) -> Dict[str, Any]:
    """Load, deduplicate, split by each method, scale, check, save and write ``preprocessing_report.json``.

    Raises:
        FileNotFoundError: If the Kaggle file is missing.
    """
    raw = data.load_raw(cfg.paths.raw_path, cfg.data)
    frame, dedupe = drop_duplicate_rows(raw, cfg.prep.drop_duplicates)
    LOGGER.info("dropped %d duplicate rows (%d frauds); %d rows left", dedupe["rows_dropped"], dedupe["frauds_dropped"], len(frame))
    report: Dict[str, Any] = {"duplicates": dedupe, "features": feature_names(cfg.data, cfg.prep), "primary_split": cfg.split.method,
                              "splits": {}}
    for method in methods:
        parts, scaler = build_split(frame, method, cfg)
        checks = check_split(parts, method, cfg)
        write_split(parts, scaler, method, cfg)
        report["splits"][method] = {"parts": summarise_parts(parts, cfg), "checks": checks,
                                    "scaler": {"columns": list(cfg.prep.scale_columns), "center": scaler.center_.round(4).tolist(),
                                               "scale": scaler.scale_.round(4).tolist()}}
        LOGGER.info("%s split: %s", method, [(p["part"], p["rows"], p["frauds"]) for p in report["splits"][method]["parts"]])
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_path / cfg.prep.report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Preprocess and split the data (Step 3).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--split", choices=("time", "stratified", "both"), default="both", help="which split(s) to write")
    parser.add_argument("--keep-duplicates", action="store_true", help="do not drop duplicate rows")
    parser.add_argument("--hour-features", action="store_true", help="add hour_sin / hour_cos from Time")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    prep = replace(base.prep, drop_duplicates=not args.keep_duplicates, hour_features=args.hour_features)
    cfg = replace(base, paths=replace(base.paths, root=args.root), prep=prep)
    run(cfg, METHODS if args.split == "both" else (args.split,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
