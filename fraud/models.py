"""The four supervised models of Step 5, behind one small interface (a name and a configuration)."""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraud import imbalance
from fraud.config import ImbalanceConfig, SupervisedConfig, SEED

OPTIONAL_PACKAGES = {"xgboost": "xgboost", "lightgbm": "lightgbm"}


def is_available(name: str) -> bool:
    """True if the model can be built (scikit-learn models always can; XGBoost / LightGBM need their package)."""
    package = OPTIONAL_PACKAGES.get(name)
    return package is None or importlib.util.find_spec(package) is not None


def make_model(name: str, cfg: SupervisedConfig = SupervisedConfig(), seed: int = SEED) -> Any:
    """An unfitted classifier with ``fit`` and ``predict_proba``.

    Raises:
        ValueError: For an unknown name.
        ImportError: If XGBoost / LightGBM is requested but not installed.
    """
    if name == "logistic_regression":
        return LogisticRegression(C=cfg.lr_c, max_iter=cfg.lr_max_iter, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=cfg.rf_trees, max_depth=cfg.rf_max_depth, min_samples_leaf=cfg.rf_min_leaf,
                                      max_samples=cfg.rf_max_samples, max_features="sqrt", n_jobs=-1, random_state=seed)
    if name == "xgboost":
        if not is_available(name):
            raise ImportError("xgboost is not installed (pip install xgboost)")
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=cfg.boost_trees, max_depth=cfg.boost_depth, learning_rate=cfg.boost_learning_rate,
                             subsample=cfg.boost_subsample, colsample_bytree=cfg.boost_colsample, reg_lambda=cfg.boost_l2,
                             tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=seed)
    if name == "lightgbm":
        if not is_available(name):
            raise ImportError("lightgbm is not installed (pip install lightgbm)")
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=cfg.boost_trees, learning_rate=cfg.boost_learning_rate, num_leaves=cfg.boost_leaves,
                              subsample=cfg.boost_subsample, subsample_freq=1, colsample_bytree=cfg.boost_colsample,
                              reg_lambda=cfg.boost_l2, n_jobs=-1, random_state=seed, verbose=-1)
    raise ValueError(f"unknown model {name!r}")


def fit_model(name: str, variant: str, X: np.ndarray, y: np.ndarray, cfg: SupervisedConfig = SupervisedConfig(),
              imb: ImbalanceConfig = ImbalanceConfig(), seed: int = SEED) -> Any:
    """Fit ``name`` on ``X, y`` after applying the imbalance ``variant`` (``baseline`` = nothing, ``smote`` = SMOTE on these rows only)."""
    if variant not in ("baseline", "smote"):
        raise ValueError(f"unknown variant {variant!r}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if variant == "smote":
        X, y = imbalance.smote(X, y, imb.sampling_ratio, imb.k_neighbors, seed)
    return make_model(name, cfg, seed).fit(X, y)


def feature_importance(model: Any, names: Tuple[str, ...]) -> Dict[str, float]:
    """Importance per feature (coefficient size for logistic regression, built-in importance otherwise), largest first, summing to 1."""
    raw = np.abs(np.ravel(model.coef_)) if hasattr(model, "coef_") else np.asarray(model.feature_importances_, dtype=float)
    if len(raw) != len(names):
        raise ValueError("number of importances does not match the feature names")
    total = raw.sum()
    share = raw / total if total > 0 else raw
    return dict(sorted(zip(names, share.round(5).tolist()), key=lambda kv: -kv[1]))
