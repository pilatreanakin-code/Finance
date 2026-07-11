"""Figures: RI-vs-DCF reconciliation bridge, terminal sensitivity, RI history."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def plot_reconciliation_bridge(
    recon: dict, market_cap: float, ticker: str, path: Path | None = None
) -> Path:
    """Waterfall: book -> +PV explicit RI -> +PV terminal RI -> RI value, then
    the DDM value and the gap decomposition alongside."""
    path = path or OUTPUT_DIR / "reconciliation_bridge.png"
    ri_c = recon["ri_components"]

    labels = ["Book equity", "+PV explicit RI", "+PV terminal RI", "RI value", "DCF (DDM) value"]
    increments = [ri_c["book"], ri_c["pv_explicit_ri"], ri_c["pv_terminal_ri"]]
    cum = np.cumsum(increments)
    bottoms = [0.0, cum[0], cum[1], 0.0, 0.0]
    heights = [increments[0], increments[1], increments[2], recon["ri_value"], recon["ddm_value"]]
    colors = ["#4878cf", "#6acc65", "#6acc65", "#2f4b7c", "#d65f5f"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [3, 2]}
    )
    x = np.arange(len(labels))
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.65)
    ax.axhline(market_cap, color="black", ls="--", lw=1.5, label=f"Market cap = {market_cap:,.0f}")
    for xi, b, h in zip(x, bottoms, heights):
        ax.text(xi, b + h + 0.01 * max(heights), f"{h:,.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"{ticker}: RI valuation bridge vs simple DCF - illustrative, synthetic data")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)

    gd = recon["gap_decomposition"]
    glabels = ["PV(end book)", "PV(terminal RI)", "-PV(terminal div)", "Total gap"]
    gvals = [gd["pv_end_book"], gd["pv_terminal_ri"], gd["minus_pv_terminal_div"], recon["gap"]]
    gcolors = ["#6acc65" if v >= 0 else "#d65f5f" for v in gvals[:-1]] + ["#2f4b7c"]
    ax2.bar(np.arange(4), gvals, color=gcolors, width=0.6)
    ax2.axhline(0, color="black", lw=0.8)
    for xi, v in enumerate(gvals):
        ax2.text(xi, v + np.sign(v) * 0.02 * max(abs(np.array(gvals))), f"{v:,.0f}",
                 ha="center", fontsize=8)
    ax2.set_xticks(np.arange(4), glabels, rotation=20, ha="right", fontsize=8)
    ax2.set_title("Gap decomposition (exact identity):\nRI - DCF = terminal assumptions only", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_terminal_sensitivity(
    sens: dict, ticker: str, path: Path | None = None
) -> Path:
    """RI vs DDM value across terminal growth - RI is the flatter line."""
    path = path or OUTPUT_DIR / "terminal_sensitivity.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    g = sens["g_grid"]
    ax.plot(g, sens["ri"], "o-", color="#2f4b7c", label="Residual income value")
    ax.plot(g, sens["ddm"], "s-", color="#d65f5f", label="Simple DCF (DDM) value")
    ax.set_xlabel("Terminal growth assumption")
    ax.set_ylabel("Equity value")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ri_swing = sens["ri"].max() / sens["ri"].min() - 1
    ddm_swing = sens["ddm"].max() / sens["ddm"].min() - 1
    ax.set_title(
        f"{ticker}: terminal-growth sensitivity - RI swings {ri_swing:.0%} vs DCF {ddm_swing:.0%}\n"
        "(illustrative, synthetic data)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ri_history(
    ri_hist: pd.Series, ticker: str, path: Path | None = None
) -> Path:
    """Historical residual income: is the firm actually earning above r?"""
    path = path or OUTPUT_DIR / "residual_income_history.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#6acc65" if v >= 0 else "#d65f5f" for v in ri_hist]
    ax.bar(ri_hist.index.astype(str), ri_hist.values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{ticker}: historical residual income (NI - r*opening book) - illustrative, synthetic data")
    ax.set_ylabel("Residual income")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
