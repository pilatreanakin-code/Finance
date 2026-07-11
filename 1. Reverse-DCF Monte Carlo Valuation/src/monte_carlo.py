from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dcf import dcf_value


@dataclass
class MCAssumptions:

    wacc_mean: float = 0.09
    wacc_std: float = 0.010
    wacc_bounds: tuple[float, float] = (0.05, 0.15)
    g_term_mean: float = 0.025
    g_term_std: float = 0.005
    g_term_bounds: tuple[float, float] = (0.0, 0.04)
    margin_mean: float = 0.12
    margin_std: float = 0.015
    margin_bounds: tuple[float, float] = (0.01, 0.40)
    min_spread: float = 0.01


@dataclass
class MCResult:

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
    grid = np.full((len(g_term_range), len(wacc_range)), np.nan)
    for i, g in enumerate(g_term_range):
        for j, w in enumerate(wacc_range):
            if w > g + 0.005:
                grid[i, j] = dcf_value(revenue0, growth, w, g, margin, horizon)
    return grid
