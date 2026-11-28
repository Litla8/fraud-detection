"""Data-drift monitoring: compare live transactions with the training data using the population stability index (PSI).

PSI = sum over bins of (live share - reference share) * ln(live share / reference share). Rules of thumb: below 0.10 stable,
0.10 to 0.25 worth watching, above 0.25 the distribution has shifted and the model should be reviewed or retrained.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from fraud import preprocess
from fraud.config import ProjectConfig, ServiceConfig
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
EPS = 1e-4                                   # keeps empty bins from producing log(0)
SCORE_KEY = "__score__"


def _edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Inner bin edges at the reference quantiles (duplicates removed: a feature with many ties gets fewer bins)."""
    return np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    index = np.searchsorted(edges, values, side="right")
    return np.bincount(index, minlength=len(edges) + 1) / max(len(values), 1)


def psi(reference: Sequence[float], live: Sequence[float]) -> float:
    """Population stability index between two bin-share vectors of the same length.

    Raises:
        ValueError: If the vectors differ in length or either is empty.
    """
    r, l = np.clip(np.asarray(reference, dtype=float), EPS, None), np.clip(np.asarray(live, dtype=float), EPS, None)
    if r.shape != l.shape or r.size == 0:
        raise ValueError("reference and live shares must be non-empty and the same length")
    return float(np.sum((l - r) * np.log(l / r)))


def build_reference(features: pd.DataFrame, scores: Optional[Sequence[float]] = None, bins: int = 10) -> Dict[str, Any]:
    """Bin edges and shares per feature (and for the model score, if given) from a reference period.

    Raises:
        ValueError: If there are fewer than 20 rows, ``bins`` < 2, or NaN values.
    """
    if len(features) < 20 or bins < 2:
        raise ValueError("need at least 20 reference rows and bins >= 2")
    if features.isna().any().any():
        raise ValueError("reference features contain NaN")
    ref: Dict[str, Any] = {"bins": bins, "rows": int(len(features)), "features": {}}
    columns = {c: features[c].to_numpy(float) for c in features.columns}
    if scores is not None:
        columns[SCORE_KEY] = np.asarray(scores, dtype=float)
    for name, values in columns.items():
        edges = _edges(values, bins)
        ref["features"][name] = {"edges": edges.tolist(), "shares": _proportions(values, edges).tolist(), "mean": float(values.mean()), "std": float(values.std())}
    return ref


def compare(reference: Dict[str, Any], features: pd.DataFrame, scores: Optional[Sequence[float]] = None,
            service: ServiceConfig = ServiceConfig()) -> Dict[str, Any]:
    """PSI per feature and for the score, with a status each and an overall status (``ok`` / ``warn`` / ``alert``).

    Raises:
        ValueError: If a reference feature is missing from ``features`` or there are fewer than ``min_window`` rows.
    """
    if len(features) < service.min_window:
        raise ValueError(f"need at least {service.min_window} rows to judge drift, got {len(features)}")
    missing = [c for c in reference["features"] if c != SCORE_KEY and c not in service.drift_ignore and c not in features.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    rows: List[Dict[str, Any]] = []
    for name, spec in reference["features"].items():
        if name in service.drift_ignore:
            continue
        if name == SCORE_KEY:
            if scores is None:
                continue
            values = np.asarray(scores, dtype=float)
        else:
            values = features[name].to_numpy(float)
        value = psi(spec["shares"], _proportions(values, np.asarray(spec["edges"])))
        rows.append({"feature": "model_score" if name == SCORE_KEY else name, "psi": round(value, 4),
                     "status": "alert" if value >= service.psi_alert else "warn" if value >= service.psi_warn else "ok",
                     "mean_shift_in_std": round(float((values.mean() - spec["mean"]) / spec["std"]), 3) if spec["std"] > 0 else 0.0})
    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    inputs = table[table["feature"] != "model_score"]
    alert_share = float((inputs["status"] == "alert").mean()) if len(inputs) else 0.0
    score_status = table.loc[table["feature"] == "model_score", "status"]
    score_alert = bool(len(score_status) and score_status.iloc[0] == "alert")
    overall = "alert" if score_alert or alert_share >= service.alert_feature_share else "warn" if (table["status"] != "ok").any() else "ok"
    return {"overall": overall, "rows": int(len(features)), "alert_feature_share": round(alert_share, 3),
            "features_alert": int((inputs["status"] == "alert").sum()), "features_warn": int((inputs["status"] == "warn").sum()), "table": table}


class DriftMonitor:
    """Keeps the most recent scored transactions (thread-safe) and reports drift against a reference once there are enough."""

    def __init__(self, reference: Dict[str, Any], service: ServiceConfig = ServiceConfig()) -> None:
        self.reference, self.service = reference, service
        self._rows: Deque[np.ndarray] = deque(maxlen=service.monitor_buffer)
        self._scores: Deque[float] = deque(maxlen=service.monitor_buffer)
        self._names = [n for n in reference["features"] if n != SCORE_KEY]
        self._lock = threading.Lock()

    def add(self, X: np.ndarray, scores: Sequence[float]) -> None:
        """Remember scored rows (``X`` has one column per reference feature, in order)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self._names) or len(X) != len(scores):
            raise ValueError("X must have one column per reference feature and one score per row")
        with self._lock:
            self._rows.extend(X)
            self._scores.extend(float(s) for s in scores)

    def report(self) -> Dict[str, Any]:
        """Status, or ``not_enough_data`` until ``min_window`` transactions have been seen."""
        with self._lock:
            seen = len(self._rows)
            if seen < self.service.min_window:
                return {"overall": "not_enough_data", "rows": seen, "needed": self.service.min_window}
            frame = pd.DataFrame(np.vstack(self._rows), columns=self._names)
            scores = np.asarray(self._scores)
        result = compare(self.reference, frame, scores, self.service)
        result["top_drifting"] = result.pop("table").head(10).to_dict("records")
        return result


def run(cfg: ProjectConfig, split: Optional[str] = None) -> Dict[str, Any]:
    """Build the reference from the training data, then check validation and the newest test data in four time slices.

    Raises:
        FileNotFoundError: If Step 3 or Step 5 outputs are missing.
    """
    from fraud import plots                                   # imported here: the API image does not ship matplotlib
    from fraud.train_anomaly import load_supervised
    parts = preprocess.load_split(cfg, split)
    names = preprocess.feature_names(cfg.data, cfg.prep)
    sup = load_supervised(cfg)
    train, val, test = parts["train"], parts["validation"], parts["test"]
    val_scores = sup["model"].predict_proba(val[names].to_numpy(float))[:, 1]
    reference = build_reference(train[names], val_scores, cfg.service.psi_bins)       # held-out scores: train scores are over-fitted
    reference["note"] = "feature shares from training rows; score shares from validation rows"
    cfg.paths.models_path.mkdir(parents=True, exist_ok=True)
    (cfg.paths.models_path / cfg.service.reference_name).write_text(json.dumps(reference), encoding="utf-8")
    test = test.sort_values(cfg.prep.raw_time_name)
    test_scores = pd.Series(sup["model"].predict_proba(test[names].to_numpy(float))[:, 1], index=test.index)
    windows = {"validation": (val, val_scores), "test (all)": (test, test_scores.to_numpy())}
    for i, chunk in enumerate(np.array_split(np.arange(len(test)), 4), start=1):
        windows[f"test slice {i} of 4"] = (test.iloc[chunk], test_scores.iloc[chunk].to_numpy())
    results, tables = {}, []
    for label, (frame, scores) in windows.items():
        out = compare(reference, frame[names], scores, cfg.service)
        table = out.pop("table")
        table.insert(0, "window", label)
        tables.append(table)
        results[label] = {**out, "top_features": table.head(5)[["feature", "psi"]].to_dict("records")}
        LOGGER.info("%-38s %s | %2d of %d features alert | score PSI %.3f | worst %s %.2f", label, out["overall"].upper(), out["features_alert"], len(names),
                    float(table.loc[table["feature"] == "model_score", "psi"].iloc[0]), table.iloc[0]["feature"], table.iloc[0]["psi"])
    reports = cfg.paths.reports_path
    reports.mkdir(parents=True, exist_ok=True)
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(reports / "drift_report.csv", index=False)
    plots.save_figure(plots.plot_psi(full[full["window"] == "test (all)"], cfg.service.psi_warn, cfg.service.psi_alert), reports / "figures" / "drift_psi.png")
    summary = {"reference_rows": reference["rows"], "thresholds": {"warn": cfg.service.psi_warn, "alert": cfg.service.psi_alert}, "windows": results}
    (reports / "drift_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: build the reference and run the drift check on validation and test."""
    parser = argparse.ArgumentParser(description="Build the drift reference and check drift (Step 9).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--split", choices=preprocess.METHODS, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    run(replace(base, paths=replace(base.paths, root=args.root)), args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
