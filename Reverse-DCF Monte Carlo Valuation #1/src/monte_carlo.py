"""Monte Carlo over DCF assumptions -> fair-value distribution.

Rather than one (WACC, terminal growth, margin) triple, draw all three from
distributions that reflect genuine uncertainty and report the resulting
fair-value distribution and P(undervalued). Truncation enforces the economic
constraint ``wacc > g_term`` on every draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dcf import dcf_value


@dataclass
class MCAssumptions:
    """Sampling distributions for the three uncertain DCF inputs.

    Normals truncated to the given bounds; ``g_term`` is additionally capped
    at ``wacc - min_spread`` draw-by-draw so the Gordon terminal value is
    always defined.
    """

    wacc_mean: float = 0.09
    wacc_std: float = 0.010
    wacc_bounds: tuple[float, float] = (0.05, 0.15)
    g_term_mean: float = 0.025
    g_term_std: float = 0.005
    g_term_bounds: tuple[float, float] = (0.0, 0.04)
    margin_mean: float = 0.12
    margin_std: float = 0.015
    margin_bounds: tuple[float, float] = (0.01, 0.40)
    min_spread: float = 0.01  # enforced wacc - g_term floor


@dataclass
class MCResult:
    """Fair-value distribution and summary statistics."""

    values: np.ndarray
    p_undervalued: float
    market_ev: float
    quantiles: dict[str, float]
    draws: dict[str, np.ndarray]


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    bounds: tuple[float, float],
    size: int,
) -> np.ndarray:
    """Sample a normal and clip to bounds (simple, adequate for mild truncation)."""
    return np.clip(rng.normal(mean, std, size), bounds[0], bounds[1])


def run_monte_carlo(
    revenue0: float,
    growth: float,
    market_ev: float,
    assumptions: MCAssumptions | None = None,
    n_draws: int = 20_000,
    horizon: int = 10,
    seed: int = 42,
) -> MCResult:
    """Monte Carlo the DCF over (wacc, g_term, margin).

    Parameters
    ----------
    revenue0 : float
        Base-year revenue.
    growth : float
        Near-term growth held fixed across draws (typically the analyst's
        central case, or the market-implied growth for a pure assumption-risk
        view).
    market_ev : float
        Observed EV; used for P(undervalued) = P(fair value > market EV).
    assumptions : MCAssumptions
        Sampling distributions; defaults are broad, defensible ranges.
    n_draws : int
        Number of draws (spec requires >= 10,000).
    horizon : int
        DCF explicit horizon.
    seed : int
        RNG seed.

    Returns
    -------
    MCResult
        Values array (length ``n_draws``), P(undervalued), quantiles, and the
        raw parameter draws for diagnostics.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    a = assumptions or MCAssumptions()
    rng = np.random.default_rng(seed)

    wacc = _truncated_normal(rng, a.wacc_mean, a.wacc_std, a.wacc_bounds, n_draws)
    g_term = _truncated_normal(rng, a.g_term_mean, a.g_term_std, a.g_term_bounds, n_draws)
    g_term = np.minimum(g_term, wacc - a.min_spread)
    margin = _truncated_normal(rng, a.margin_mean, a.margin_std, a.margin_bounds, n_draws)

    values = np.array(
        [
            dcf_value(revenue0, growth, w, g, m, horizon)
            for w, g, m in zip(wacc, g_term, margin)
        ]
    )

    qs = np.percentile(values, [5, 25, 50, 75, 95])
    return MCResult(
        values=values,
        p_undervalued=float(np.mean(values > market_ev)),
        market_ev=market_ev,
        quantiles={"p5": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p95": qs[4]},
        draws={"wacc": wacc, "g_term": g_term, "margin": margin},
    )


def sensitivity_grid(
    revenue0: float,
    growth: float,
    margin: float,
    wacc_range: np.ndarray,
    g_term_range: np.ndarray,
    horizon: int = 10,
) -> np.ndarray:
    """WACC x terminal-growth grid of enterprise values.

    Cells where ``wacc <= g_term + 0.5%`` are NaN (terminal value undefined
    or absurd). Rows index ``g_term_range``, columns index ``wacc_range``.
    """
    grid = np.full((len(g_term_range), len(wacc_range)), np.nan)
    for i, g in enumerate(g_term_range):
        for j, w in enumerate(wacc_range):
            if w > g + 0.005:
                grid[i, j] = dcf_value(revenue0, growth, w, g, margin, horizon)
    return grid
