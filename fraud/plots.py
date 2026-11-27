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


def plot_pr_curves(curves: dict, base_rate: float) -> Figure:
    """Precision-recall curves on validation data, one line per strategy (``curves`` maps name -> (precision, recall))."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (precision, recall) in curves.items():
        ax.plot(recall, precision, label=name, lw=1.6)
    ax.axhline(base_rate, color="black", ls=":", lw=1, label="no skill")
    ax.axhline(0.85, color="grey", ls="--", lw=1)
    ax.text(0.01, 0.86, "precision 0.85", fontsize=8, color="grey")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall on validation data")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig
