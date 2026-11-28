"""The scoring core of the service: raw transaction in, score, risk tier, action and (optionally) reasons out.

Deliberately free of web-framework code so it can be tested, benchmarked and reused. ``fraud.api`` is a thin wrapper around it.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from fraud import explain as explain_module
from fraud import policy, preprocess
from fraud.config import ProjectConfig
from fraud.drift import DriftMonitor

LOGGER = logging.getLogger(__name__)


def build_features(frame: pd.DataFrame, scaler: Any, cfg: ProjectConfig = ProjectConfig()) -> pd.DataFrame:
    """Model input from raw transactions, exactly as ``preprocess.transform`` builds it (same scaler, same column order).

    Raises:
        ValueError: If a raw column is missing or any value is NaN.
    """
    d, p = cfg.data, cfg.prep
    needed = [d.time_column, d.amount_column, *d.pca_columns]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if frame[needed].isna().any().any():
        raise ValueError("transaction contains NaN")
    scaled = pd.DataFrame(scaler.transform(frame[list(p.scale_columns)].astype(float)), index=frame.index,
                          columns=[c + p.scaled_suffix for c in p.scale_columns])
    out = pd.concat([frame[list(d.pca_columns)].astype("float64"), scaled], axis=1)
    if p.hour_features:
        angle = 2 * np.pi * ((frame[d.time_column] // 3600) % 24) / 24
        out["hour_sin"], out["hour_cos"] = np.sin(angle), np.cos(angle)
    return out[preprocess.feature_names(d, p)]


class Scorer:
    """Loads the model, scaler and decision policy once and scores transactions.

    ``contributions`` (optional) is a function ``X -> (contributions, base, method)``; by default it is ``explain.contributions(model, X)``.
    """

    def __init__(self, model: Any, scaler: Any, decision_policy: Mapping[str, Any], cfg: ProjectConfig = ProjectConfig(),
                 monitor: Optional[DriftMonitor] = None, contributions: Optional[Callable[[np.ndarray], Any]] = None) -> None:
        for key in ("features", "otp_threshold", "high_threshold", "tiers", "model_variant"):
            if key not in decision_policy:
                raise ValueError(f"decision policy is missing {key!r}")
        if list(decision_policy["features"]) != preprocess.feature_names(cfg.data, cfg.prep):
            raise ValueError("the policy was built with different features than the current configuration")
        if decision_policy["otp_threshold"] > decision_policy["high_threshold"]:
            raise ValueError("otp_threshold must not exceed high_threshold")
        if "n_jobs" in getattr(model, "get_params", lambda: {})() and model.get_params()["n_jobs"] not in (None, 1):
            model.set_params(n_jobs=1)                    # one row at a time: a thread pool costs more than it saves
        self.model, self.scaler, self.policy, self.cfg, self.monitor = model, scaler, dict(decision_policy), cfg, monitor
        self.names = list(decision_policy["features"])
        self._contributions = contributions or (lambda X: explain_module.contributions(model, X))

    @classmethod
    def load(cls, cfg: ProjectConfig) -> "Scorer":
        """Build a scorer from the files Steps 3, 5, 8 and 9 wrote.

        Raises:
            FileNotFoundError: Naming the missing file and the step that creates it.
        """
        models = cfg.paths.models_path
        decision = models / cfg.thresholds.policy_name
        for path, step in ((models / "supervised_best.joblib", "Step 5 (train_supervised)"), (decision, "Step 8 (optimize_threshold)")):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found: run {step} first")
        policy_data = json.loads(decision.read_text(encoding="utf-8"))
        scaler_path = cfg.paths.scaler_path(policy_data["split"])
        if not scaler_path.is_file():
            raise FileNotFoundError(f"{scaler_path} not found: run Step 3 (preprocess) first")
        reference = models / cfg.service.reference_name
        monitor = DriftMonitor(json.loads(reference.read_text(encoding="utf-8")), cfg.service) if reference.is_file() else None
        if monitor is None:
            LOGGER.warning("no %s: drift monitoring is off (run python -m fraud.drift)", cfg.service.reference_name)
        return cls(joblib.load(models / "supervised_best.joblib"), joblib.load(scaler_path), policy_data, cfg, monitor)

    def score_records(self, records: Sequence[Mapping[str, float]], explain: bool = False) -> List[Dict[str, Any]]:
        """Score raw transactions (dicts with Time, V1-V28, Amount).

        Each result has ``score``, ``tier`` (low / medium / high), ``action`` and, with ``explain``, ``reasons``. ``latency_ms`` is
        the time for the whole call, shared by every transaction in it.

        Raises:
            ValueError: On an empty list, missing fields or NaN.
            ImportError: If ``explain`` is requested and the model needs the shap package that is not installed.
        """
        start = time.perf_counter()
        if not records:
            raise ValueError("no transactions to score")
        features = build_features(pd.DataFrame(list(records)), self.scaler, self.cfg)
        X = features.to_numpy(float)
        scores = self.model.predict_proba(X)[:, 1]
        tiers = policy.tier_of(scores, self.policy["otp_threshold"], self.policy["high_threshold"])
        reasons = None
        if explain:
            contrib, _, method = self._contributions(X)
            reasons = [explain_module.top_reasons(contrib[i], self.names, X[i], self.cfg.service.reasons) for i in range(len(X))]
        if self.monitor is not None:
            self.monitor.add(X, scores)
        latency = round((time.perf_counter() - start) * 1000, 3)
        results = []
        for i, (score, tier) in enumerate(zip(scores, tiers)):
            row: Dict[str, Any] = {"score": round(float(score), 6), "tier": str(tier), "action": self.policy["tiers"][str(tier)], "latency_ms": latency}
            if reasons is not None:
                row["reasons"] = reasons[i]
            results.append(row)
        return results

    def info(self) -> Dict[str, Any]:
        """Model and policy details for the health and policy endpoints."""
        return {"model": self.policy["model_variant"], "otp_threshold": self.policy["otp_threshold"], "high_threshold": self.policy["high_threshold"],
                "tiers": self.policy["tiers"], "costs": self.policy.get("costs", {}), "drift_monitoring": self.monitor is not None}
