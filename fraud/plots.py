"""Charts. Each function returns a matplotlib Figure; ``save_figure`` writes it. Fraud is red, genuine is blue throughout."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")                       # no window: safe in scripts, tests and servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fraud.config import DataConfig

GENUINE, FRAUD = "#4f81bd", "#c0504d"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: Figure, path: Path, dpi: int = 150) -> Path:
    """Write ``fig`` to ``path`` (creating folders) and close it so memory is freed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(rows: int, frauds: int) -> Figure:
    """Genuine vs fraud counts on a log axis, with the fraud share annotated."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(["Genuine", "Fraud"], [rows - frauds, frauds], color=[GENUINE, FRAUD])
    ax.set_yscale("log")
    ax.set_ylabel("transactions (log scale)")
    for bar, value in zip(bars, [rows - frauds, frauds]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    ax.set_title(f"Fraud is {100 * frauds / rows:.3f}% of transactions (1 in {rows // max(frauds, 1)})")
    _style(ax)
    return fig


def plot_amounts(frame: pd.DataFrame, bands: pd.DataFrame, cfg: DataConfig = DataConfig()) -> Figure:
    """Left: amount distributions (log-spaced, each class scaled to 100%). Right: fraud rate per amount band."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    amount, label = frame[cfg.amount_column], frame[cfg.label_column]
    edges = np.logspace(-0.5, np.log10(max(float(amount.max()), 10)), 40)
    for value, name, colour in ((0, "Genuine", GENUINE), (1, "Fraud", FRAUD)):
        subset = amount[(label == value) & (amount > 0)]
        left.hist(subset, bins=edges, weights=np.full(len(subset), 100 / max(len(subset), 1)), alpha=0.6, color=colour, label=name)
    left.set_xscale("log")
    left.set_xlabel("amount (log scale)"); left.set_ylabel("% of that class"); left.legend(frameon=False)
    left.set_title("Fraud has more tiny and more large amounts")
    right.bar(range(len(bands)), 100 * bands["fraud_rate"], color=FRAUD)
    right.set_xticks(range(len(bands)))
    right.set_xticklabels([b.replace("[", "").replace(")", "").replace(", ", "-").replace(".0", "") for b in bands["amount_band"]], rotation=45, ha="right")
    right.axhline(100 * label.mean(), color="black", ls="--", lw=1)
    right.set_ylabel("fraud rate (%)"); right.set_title("Fraud rate by amount band (dashed: overall)")
    for ax in (left, right):
        _style(ax)
    fig.tight_layout()
    return fig


def plot_time_patterns(frame: pd.DataFrame, hourly: pd.DataFrame, cfg: DataConfig = DataConfig()) -> Figure:
    """Top: transactions and fraud rate by hour of the data's clock. Bottom: every fraud over the two days (size = amount)."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7))
    top.bar(hourly["hour"], hourly["transactions"], color=GENUINE, alpha=0.7)
    top.set_ylabel("transactions"); top.set_xlabel("hour (since first transaction, mod 24)"); top.set_xticks(range(24))
    twin = top.twinx()
    twin.plot(hourly["hour"], 100 * hourly["fraud_rate"], color=FRAUD, marker="o")
    twin.set_ylabel("fraud rate (%)", color=FRAUD)
    top.set_title("Volume peaks in the day, fraud rate peaks at night")
    fraud = frame[frame[cfg.label_column] == 1]
    bottom.scatter(fraud[cfg.time_column] / 3600, fraud[cfg.amount_column].clip(upper=1500), s=14, color=FRAUD, alpha=0.7)
    bottom.set_xlabel("hours since first transaction"); bottom.set_ylabel("amount (capped at 1,500)")
    bottom.set_title(f"All {len(fraud)} frauds over the two days")
    for ax in (top, twin, bottom):
        _style(ax)
    fig.tight_layout()
    return fig


def plot_feature_separation(table: pd.DataFrame, top: int = 15) -> Figure:
    """KS statistic of the most separating features; red means fraud has the lower mean, blue the higher."""
    shown = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(shown["feature"], shown["ks"], color=[FRAUD if d < 0 else GENUINE for d in shown["cohens_d"]])
    ax.set_xlabel("KS statistic: fraud vs genuine (0 = identical, 1 = fully separated)")
    ax.set_title("Features that separate fraud best\n(red: lower in fraud, blue: higher in fraud)")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_top_distributions(frame: pd.DataFrame, features: Sequence[str], cfg: DataConfig = DataConfig()) -> Figure:
    """Density histograms of fraud vs genuine for the given features; values beyond the shown range are pooled in the end bars."""
    cols = 3
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    label = frame[cfg.label_column]
    for ax, name in zip(np.atleast_1d(axes).ravel(), features):
        fraud_q = frame.loc[label == 1, name].quantile([0.02, 0.98])         # fraud sits in the tails, so the range must cover it
        genuine_q = frame.loc[label == 0, name].quantile([0.005, 0.995])
        low, high = min(fraud_q.iloc[0], genuine_q.iloc[0]), max(fraud_q.iloc[1], genuine_q.iloc[1])
        bins = np.linspace(low, high, 40)
        ax.hist(frame.loc[label == 0, name].clip(low, high), bins=bins, density=True, alpha=0.6, color=GENUINE, label="Genuine")
        ax.hist(frame.loc[label == 1, name].clip(low, high), bins=bins, density=True, alpha=0.6, color=FRAUD, label="Fraud")
        ax.set_title(name); ax.set_yticks([]); _style(ax)
    for ax in np.atleast_1d(axes).ravel()[len(features):]:
        ax.axis("off")
    np.atleast_1d(axes).ravel()[0].legend(frameon=False)
    fig.suptitle("Top separating features: fraud vs genuine (extreme values pooled in the end bars)", y=1.0)
    fig.tight_layout()
    return fig


def plot_drift(table: pd.DataFrame, flag: float = 0.10, top: int = 15) -> Figure:
    """KS distance between the oldest and newest transactions per feature; bars past the line are drifting."""
    shown = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(shown["feature"], shown["ks_train_vs_test"], color=[FRAUD if f else GENUINE for f in shown["flagged"]])
    ax.axvline(flag, color="black", ls="--", lw=1)
    ax.set_xlabel("KS distance, training period vs test period")
    ax.set_title("Which features drift between the oldest and newest transactions")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_correlation(frame: pd.DataFrame, features: Sequence[str], cfg: DataConfig = DataConfig()) -> Figure:
    """Correlation heatmap of the given features and the label."""
    columns = [*features, cfg.label_column]
    corr = frame[columns].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns))); ax.set_xticklabels(columns, rotation=90)
    ax.set_yticks(range(len(columns))); ax.set_yticklabels(columns)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Correlation of the top features with each other and with Class")
    fig.tight_layout()
    return fig


def plot_strategy_cv(table: pd.DataFrame, winner: str) -> Figure:
    """Cross-validated PR-AUC (mean with one standard deviation) for every imbalance strategy; the winner is dark."""
    ordered = table.sort_values("cv_pr_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = [FRAUD if name == winner else GENUINE for name in ordered["strategy"]]
    ax.barh(ordered["strategy"], ordered["cv_pr_auc_mean"], xerr=ordered["cv_pr_auc_std"], color=colours, capsize=4)
    ax.set_xlabel("PR-AUC across cross-validation folds (bar: mean, whisker: one standard deviation)")
    ax.set_title("Imbalance strategies compared (training data, folds)")
    for y, (value, std) in enumerate(zip(ordered["cv_pr_auc_mean"], ordered["cv_pr_auc_std"])):
        ax.text(value + std + 0.005, y, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, min(1.05, float(ordered["cv_pr_auc_mean"].max() + ordered["cv_pr_auc_std"].max()) + 0.12))
    _style(ax)
    fig.tight_layout()
    return fig


def plot_pr_curves(curves: dict, base_rate: float, title: str = "Precision-recall on validation data") -> Figure:
    """Precision-recall curves, one line per strategy (``curves`` maps name -> (precision, recall))."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (precision, recall) in curves.items():
        ax.plot(recall, precision, label=name, lw=1.6)
    ax.axhline(base_rate, color="black", ls=":", lw=1, label="no skill")
    ax.axhline(0.85, color="grey", ls="--", lw=1)
    ax.text(0.01, 0.86, "precision 0.85", fontsize=8, color="grey")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_model_comparison(table: pd.DataFrame, target: float) -> Figure:
    """Test PR-AUC with its 95% bootstrap interval for every model and variant, and the target line."""
    ordered = table.sort_values("test_pr_auc")
    labels = ordered["model_variant"].str.replace("_", " ").str.replace("/", " / ")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(ordered) + 1.6))
    err = np.vstack([ordered["test_pr_auc"] - ordered["test_pr_auc_ci_low"], ordered["test_pr_auc_ci_high"] - ordered["test_pr_auc"]])
    ax.barh(labels, ordered["test_pr_auc"], xerr=err, color=GENUINE, capsize=3)
    ax.axvline(target, color="black", ls="--", lw=1)
    ax.text(target, len(ordered) - 0.4, f" target {target:.2f}", fontsize=9, va="bottom")
    ax.set_xlim(0, 1.0); ax.set_xlabel("test PR-AUC (bar) with 95% bootstrap interval (whisker)")
    ax.set_title("Supervised models on the untouched test set")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_importance(importance: dict, top: int = 15, title: str = "Feature importance") -> Figure:
    """Horizontal bars of the largest feature importances (share of the total)."""
    items = list(importance.items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.barh([k for k, _ in items], [v for _, v in items], color=GENUINE)
    ax.set_xlabel("share of total importance"); ax.set_title(title)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_anomaly_separation(percentiles: dict) -> Figure:
    """How well each detector separates fraud: histogram of the percentile score for genuine and fraud rows.

    ``percentiles`` maps detector name -> (genuine_percentiles, fraud_percentiles). A perfect detector piles fraud at 1.0.
    """
    count = max(len(percentiles), 1)
    fig, axes = plt.subplots(1, count, figsize=(4.2 * count, 3.6), squeeze=False)
    bins = np.linspace(0, 1, 21)
    for ax, (name, (genuine, fraud)) in zip(axes[0], percentiles.items()):
        ax.hist(genuine, bins=bins, density=True, color=GENUINE, alpha=0.6, label="genuine")
        ax.hist(fraud, bins=bins, density=True, color=FRAUD, alpha=0.6, label="fraud")
        ax.set_title(name.replace("_", " ")); ax.set_xlabel("anomaly percentile (1 = most unusual)")
        _style(ax)
    axes[0][0].set_ylabel("density"); axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_blend_weights(grid: pd.DataFrame) -> Figure:
    """Validation PR-AUC against the anomaly weight in the hybrid, one line per detector (columns detector, weight, val_pr_auc)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, part in grid.groupby("detector"):
        ax.plot(part["weight"], part["val_pr_auc"], marker="o", label=name.replace("_", " "))
    ax.set_xlabel("weight of the anomaly score (0 = supervised model only)"); ax.set_ylabel("validation PR-AUC")
    ax.set_title("Does adding an anomaly score help?")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_cost_curves(curves: dict, chosen: dict, n_rows: dict) -> Figure:
    """Total cost against the share of transactions flagged, per dataset (``curves`` maps name -> cost_curve table).

    ``chosen`` maps name -> threshold row (flagged count and cost) to mark; ``n_rows`` maps name -> number of transactions.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = {"validation": GENUINE, "test": FRAUD}
    for name, table in curves.items():
        share = table["flagged"].to_numpy() / n_rows[name]
        keep = share > 0
        ax.plot(share[keep], table["cost"].to_numpy()[keep], label=name, color=colours.get(name), lw=1.6)
        if name in chosen:
            ax.scatter([chosen[name]["flagged"] / n_rows[name]], [chosen[name]["cost"]], color=colours.get(name), zorder=3, s=40)
    ax.set_xscale("log"); ax.set_xlabel("share of transactions flagged (log scale)"); ax.set_ylabel("total cost (EUR)")
    ax.set_title("Cost of every possible threshold (dot = chosen)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_tiers(tiers: pd.DataFrame) -> Figure:
    """Transactions and frauds per risk tier (columns tier, rows, frauds), counts on a log axis."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    palette = {"low": GENUINE, "medium": "#e0a030", "high": FRAUD}
    for ax, column, title in zip(axes, ("rows", "frauds"), ("transactions per tier", "frauds per tier")):
        ax.bar(tiers["tier"], tiers[column].clip(lower=0.5), color=[palette.get(t, "grey") for t in tiers["tier"]])
        for i, value in enumerate(tiers[column]):
            ax.text(i, max(value, 0.5), f"{int(value):,}", ha="center", va="bottom", fontsize=9)
        ax.set_yscale("log"); ax.set_title(title)
        _style(ax)
    fig.tight_layout()
    return fig


def plot_threshold_stability(thresholds: Sequence[float], chosen: float) -> Figure:
    """How the cost-optimal threshold moves when the validation set is resampled (log axis), with the chosen (median) value."""
    values = np.asarray([t for t in thresholds if np.isfinite(t) and t > 0])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(np.log10(values), bins=25, color=GENUINE)
    ax.axvline(np.log10(chosen), color=FRAUD, lw=2, label=f"chosen = median ({chosen:.4f})")
    ax.set_xlabel("cost-optimal threshold (log10)"); ax.set_ylabel("bootstrap resamples")
    ax.set_title("How stable is the cost-optimal threshold?")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_shap_importance(table: pd.DataFrame, method: str, top: int = 15) -> Figure:
    """Mean absolute SHAP contribution per feature, over all rows and over frauds only (when the table has ``mean_abs_fraud``)."""
    items = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    height = 0.4
    positions = np.arange(len(items))
    ax.barh(positions + (height / 2 if "mean_abs_fraud" in items else 0), items["mean_abs_all"], height=height, color=GENUINE, label="all rows")
    if "mean_abs_fraud" in items:
        ax.barh(positions - height / 2, items["mean_abs_fraud"], height=height, color=FRAUD, label="fraud rows")
        ax.legend(frameon=False)
    ax.set_yticks(positions, items["feature"])
    ax.set_xlabel("mean |contribution to the score|"); ax.set_title(f"What drives the model ({method})")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_example_explanations(examples: dict) -> Figure:
    """One panel per example transaction: signed contributions of its top features (red raises fraud risk, blue lowers it)."""
    count = max(len(examples), 1)
    fig, axes = plt.subplots(1, count, figsize=(4.6 * count, 3.4), squeeze=False)
    for ax, (kind, info) in zip(axes[0], examples.items()):
        reasons = info["reasons"][::-1]
        ax.barh([r["feature"] for r in reasons], [r["contribution"] for r in reasons], color=[FRAUD if r["contribution"] > 0 else GENUINE for r in reasons])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{kind.replace('_', ' ')} (score {info['score']:.3f})", fontsize=10)
        _style(ax)
    fig.tight_layout()
    return fig


def plot_psi(table: pd.DataFrame, warn: float, alert: float, top: int = 15) -> Figure:
    """PSI of the most drifted features (and the model score) for one window, with the warn and alert lines."""
    items = table.sort_values("psi", ascending=False).head(top).iloc[::-1]
    colours = [FRAUD if p >= alert else "#e0a030" if p >= warn else GENUINE for p in items["psi"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(items["feature"], items["psi"], color=colours)
    ax.axvline(warn, color="grey", ls="--", lw=1); ax.axvline(alert, color="black", ls="--", lw=1)
    ax.set_xlabel("population stability index (dashed: warn, alert)"); ax.set_title("Which inputs have drifted since training?")
    _style(ax)
    fig.tight_layout()
    return fig
