"""Step 9 explanations: per-transaction feature contributions (SHAP values) and the global picture.

LightGBM and XGBoost compute exact TreeSHAP values natively (``pred_contrib``), which equal ``shap.TreeExplainer`` for those models and
need no extra package in the API image. Logistic regression contributions are ``coefficient x value``. Other tree models use the
``shap`` package. Contributions are in log-odds for boosted models and logistic regression, and in probability for forests.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fraud import preprocess
from fraud.config import ProjectConfig, SEED
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def _kind(model: Any) -> str:
    module, name = type(model).__module__, type(model).__name__
    if module.startswith("lightgbm"):
        return "lightgbm"
    if module.startswith("xgboost"):
        return "xgboost"
    if name == "LogisticRegression":
        return "linear"
    return "tree"


def contributions(model: Any, X: np.ndarray, method: str = "auto") -> Tuple[np.ndarray, float, str]:
    """``(contributions, base_value, method_used)``: one column per feature, so ``base + row sum`` is the model's raw output.

    ``method`` is ``auto`` (native where possible) or ``shap`` (force the shap package, e.g. to cross-check the native values).

    Raises:
        ValueError: For an unknown method or a non-2-D ``X``.
        ImportError: If the shap package is needed and missing.
    """
    if method not in ("auto", "shap"):
        raise ValueError("method must be 'auto' or 'shap'")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    kind = _kind(model)
    if method == "auto" and kind == "lightgbm":
        raw = np.asarray(model.predict(X, pred_contrib=True))
        return raw[:, :-1], float(raw[0, -1]), "lightgbm_native_treeshap"
    if method == "auto" and kind == "xgboost":
        import xgboost
        raw = np.asarray(model.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True))
        return raw[:, :-1], float(raw[0, -1]), "xgboost_native_treeshap"
    if method == "auto" and kind == "linear":
        return X * np.ravel(model.coef_), float(np.ravel(model.intercept_)[0]), "linear_coefficients"
    try:
        import shap
    except ImportError as error:
        raise ImportError("this model needs the shap package for explanations (pip install shap)") from error
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):                                  # older shap: one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:                                          # newer shap: (rows, features, classes)
        values = values[:, :, -1]
    base = np.ravel(explainer.expected_value)[-1]
    return values, float(base), "shap_tree_explainer"


def top_reasons(contribution_row: Sequence[float], names: Sequence[str], values: Sequence[float], k: int = 3) -> List[Dict[str, Any]]:
    """The ``k`` features that moved one transaction's score most (by absolute contribution), with direction and the feature value."""
    c = np.asarray(contribution_row, dtype=float)
    if not (len(c) == len(names) == len(values)):
        raise ValueError("contributions, names and values must have the same length")
    order = np.argsort(-np.abs(c), kind="stable")[:k]
    return [{"feature": names[i], "contribution": round(float(c[i]), 4), "value": round(float(values[i]), 4),
             "effect": "raises fraud risk" if c[i] > 0 else "lowers fraud risk"} for i in order]


def global_importance(contrib: np.ndarray, names: Sequence[str], is_fraud: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Mean absolute contribution per feature (and among frauds only if labels are given), largest first."""
    c = np.abs(np.asarray(contrib, dtype=float))
    if c.ndim != 2 or c.shape[1] != len(names):
        raise ValueError("contrib must have one column per feature name")
    table = pd.DataFrame({"feature": list(names), "mean_abs_all": c.mean(axis=0)})
    if is_fraud is not None and np.asarray(is_fraud).astype(bool).any():
        table["mean_abs_fraud"] = c[np.asarray(is_fraud).astype(bool)].mean(axis=0)
    table["share"] = table["mean_abs_all"] / table["mean_abs_all"].sum()
    return table.sort_values("mean_abs_all", ascending=False).reset_index(drop=True)


def sample_rows(frame: pd.DataFrame, label: str, n: int, seed: int = SEED) -> pd.DataFrame:
    """All fraud rows plus a random sample of genuine rows, up to about ``n`` rows in total (never fewer than the frauds)."""
    fraud = frame[frame[label] == 1]
    genuine = frame[frame[label] == 0]
    take = max(min(n - len(fraud), len(genuine)), 0)
    return pd.concat([fraud, genuine.sample(n=take, random_state=seed)]).sort_index()


def run(cfg: ProjectConfig, method: str = "auto", split: Optional[str] = None) -> Dict[str, Any]:
    """Global importance on test frauds + a genuine sample, and example explanations (best-caught fraud, a missed fraud, a false alarm)."""
    from fraud import plots                                   # imported here: the API image does not ship matplotlib
    from fraud.train_anomaly import load_supervised
    parts = preprocess.load_split(cfg, split)
    names = preprocess.feature_names(cfg.data, cfg.prep)
    label, raw = cfg.data.label_column, cfg.prep.raw_amount_name
    sup = load_supervised(cfg)
    test = sample_rows(parts["test"], label, cfg.service.shap_rows)
    X = test[names].to_numpy(float)
    contrib, base, used = contributions(sup["model"], X, method)
    score = sup["model"].predict_proba(X)[:, 1]
    y = test[label].to_numpy()
    table = global_importance(contrib, names, y == 1)
    LOGGER.info("explanations by %s on %d rows (%d frauds); base value %.3f", used, len(test), int(y.sum()), base)
    LOGGER.info("top features: %s", ", ".join(f"{r.feature} {r.share:.0%}" for r in table.head(5).itertuples()))
    picks = {"caught_fraud": np.flatnonzero(y == 1)[np.argmax(score[y == 1])], "missed_fraud": np.flatnonzero(y == 1)[np.argmin(score[y == 1])],
             "false_alarm": np.flatnonzero(y == 0)[np.argmax(score[y == 0])]}
    examples = {}
    for kind, i in picks.items():
        examples[kind] = {"score": round(float(score[i]), 4), "amount": round(float(test[raw].iloc[i]), 2),
                          "reasons": top_reasons(contrib[i], names, X[i], cfg.service.reasons + 2)}
    reports = cfg.paths.reports_path
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "shap_importance.csv", index=False)
    plots.save_figure(plots.plot_shap_importance(table, used), reports / "figures" / "shap_importance.png")
    plots.save_figure(plots.plot_example_explanations(examples), reports / "figures" / "example_explanations.png")
    summary = {"method": used, "rows": int(len(test)), "frauds": int(y.sum()), "base_value": base, "top_features": table.head(10).round(5).to_dict("records"),
               "examples": examples,
               "note": "Features are anonymised PCA components, so a reason such as V14 is statistically valid but not business-readable."}
    (reports / "explanations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="SHAP explanations for the supervised model (Step 9).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--method", choices=("auto", "shap"), default="auto", help="auto = native TreeSHAP where possible; shap = force the shap package")
    parser.add_argument("--split", choices=preprocess.METHODS, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    run(replace(base, paths=replace(base.paths, root=args.root)), args.method, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
