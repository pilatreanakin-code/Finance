from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monte_carlo import MCResult

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def plot_fair_value_distribution(
    result: MCResult, ticker: str, path: Path | None = None
) -> Path:
    path = path or OUTPUT_DIR / "fair_value_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(result.values, bins=80, color="#4878cf", alpha=0.85, edgecolor="none")
    ax.axvline(result.market_ev, color="crimson", lw=2, label=f"Market EV = {result.market_ev:,.0f}")
    ax.axvline(
        result.quantiles["p50"],
        color="black",
        lw=1.5,
        ls="--",
        label=f"Median fair value = {result.quantiles['p50']:,.0f}",
    )
    ax.set_title(
        f"{ticker}: Monte-Carlo fair-value distribution "
        f"(P(undervalued) = {result.p_undervalued:.1%}) - illustrative, synthetic data"
    )
    ax.set_xlabel("Enterprise value")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sensitivity_heatmap(
    grid: np.ndarray,
    wacc_range: np.ndarray,
    g_term_range: np.ndarray,
    market_ev: float,
    ticker: str,
    path: Path | None = None,
) -> Path:
    path = path or OUTPUT_DIR / "wacc_growth_sensitivity.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    rel = grid / market_ev - 1.0
    im = ax.imshow(rel, cmap="RdYlGn", aspect="auto", origin="lower", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(wacc_range)), [f"{w:.1%}" for w in wacc_range])
    ax.set_yticks(range(len(g_term_range)), [f"{g:.1%}" for g in g_term_range])
    ax.set_xlabel("WACC")
    ax.set_ylabel("Terminal growth")
    ax.set_title(
        f"{ticker}: DCF value vs market EV across WACC x terminal growth\n"
        "(cell = fair value / market EV - 1) - illustrative, synthetic data"
    )
    for i in range(len(g_term_range)):
        for j in range(len(wacc_range)):
            if np.isfinite(rel[i, j]):
                ax.text(
                    j, i, f"{rel[i, j]:+.0%}", ha="center", va="center", fontsize=8,
                    color="black",
                )
    fig.colorbar(im, ax=ax, label="Premium / (discount) to market EV")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_implied_growth_context(
    hist_growth: np.ndarray, implied: float, ticker: str, path: Path | None = None
) -> Path:
    path = path or OUTPUT_DIR / "implied_vs_historical_growth.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    years = np.arange(len(hist_growth))
    ax.bar(years, hist_growth, color="#4878cf", alpha=0.85, label="Historical revenue growth")
    ax.axhline(implied, color="crimson", lw=2, ls="--", label=f"Market-implied growth = {implied:.1%}")
    ax.set_xticks(years, [f"Y{-len(hist_growth) + 1 + i}" for i in years])
    ax.set_ylabel("Annual revenue growth")
    ax.set_title(f"{ticker}: what the market is pricing vs delivered history - illustrative, synthetic data")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
