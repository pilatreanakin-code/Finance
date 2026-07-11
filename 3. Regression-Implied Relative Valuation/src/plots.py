from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_multiple import FairMultipleResult

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def plot_actual_vs_fair(
    result: FairMultipleResult, path: Path | None = None
) -> Path:
    path = path or OUTPUT_DIR / f"actual_vs_fair_{result.multiple}.png"
    actual = result.fitted + result.residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        result.fitted, actual, c=result.residual_z, cmap="RdYlGn_r",
        vmin=-2.5, vmax=2.5, s=45, edgecolor="k", linewidth=0.3,
    )
    lims = [min(result.fitted.min(), actual.min()) * 0.95,
            max(result.fitted.max(), actual.max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="Actual = fair (45°)")
    ax.set_xlabel(f"Regression-implied fair {result.multiple.upper()}")
    ax.set_ylabel(f"Actual {result.multiple.upper()}")
    ax.set_title(
        f"Actual vs fair {result.multiple.upper()} (R² = {result.r_squared:.2f}) - "
        "above line = rich, below = cheap\n(illustrative, synthetic data)"
    )
    fig.colorbar(sc, ax=ax, label="Residual z-score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residual_ranking(
    ranking: pd.DataFrame, top_n: int = 10, path: Path | None = None
) -> Path:
    path = path or OUTPUT_DIR / "residual_ranking.png"
    extremes = pd.concat([ranking.head(top_n), ranking.tail(top_n)])
    colors = ["#6acc65" if v < 0 else "#d65f5f" for v in extremes["composite_z"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(extremes.index, extremes["composite_z"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Composite residual z-score (negative = cheap)")
    ax.set_title(
        f"Rich/cheap ranking - {top_n} cheapest and {top_n} richest vs fundamentals\n"
        "(illustrative, synthetic data)"
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_winsorization_effect(
    raw: pd.Series, winsorized: pd.Series, path: Path | None = None
) -> Path:
    path = path or OUTPUT_DIR / "winsorization_effect.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(raw.min(), winsorized.min()),
                       max(raw.max(), winsorized.max()), 50)
    ax.hist(raw, bins=bins, alpha=0.55, label="Raw", color="#d65f5f")
    ax.hist(winsorized, bins=bins, alpha=0.55, label="Winsorized 1/99", color="#4878cf")
    ax.set_xlabel(raw.name or "multiple")
    ax.set_ylabel("Count")
    ax.set_title("Winsorization: outliers capped, not deleted - illustrative, synthetic data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
