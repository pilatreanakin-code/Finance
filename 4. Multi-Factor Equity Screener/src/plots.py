from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def plot_quintile_spread(q_returns: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or OUTPUT_DIR / "quintile_spread.png"
    qcols = [c for c in q_returns.columns if c.startswith("Q")]
    means = q_returns[qcols].mean() * 100
    spread = q_returns["spread"].mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(qcols)))
    ax.bar(qcols, means, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean forward return (% per period)")
    ax.set_title(
        f"Forward return by composite quintile - mean Q5-Q1 spread = {spread:.2f}%/period\n"
        "(illustrative, synthetic data)"
    )
    for i, v in enumerate(means):
        ax.text(i, v + 0.02 * np.sign(v), f"{v:.2f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_factor_correlation(corr: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or OUTPUT_DIR / "factor_correlation.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Average cross-sectional factor correlation\n(illustrative, synthetic data)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ic_series(
    ic_ts: pd.DataFrame, weights: pd.DataFrame, path: Path | None = None
) -> Path:
    path = path or OUTPUT_DIR / "ic_and_weights.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for f in ic_ts.columns:
        ax1.plot(ic_ts.index, ic_ts[f], marker="o", ms=3, label=f)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Spearman IC")
    ax1.set_title("Per-date factor ICs and lagged expanding-mean weights - illustrative, synthetic data")
    ax1.legend()

    ax2.stackplot(
        weights.index, [weights[f] for f in weights.columns], labels=list(weights.columns),
        alpha=0.85,
    )
    ax2.set_ylabel("Composite weight")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
