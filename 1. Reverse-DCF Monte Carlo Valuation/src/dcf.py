from __future__ import annotations

import numpy as np


def dcf_value(
    revenue0: float,
    growth: float,
    wacc: float,
    g_term: float,
    margin: float,
    horizon: int = 10,
    fade: bool = True,
) -> float:
    if wacc <= g_term:
        raise ValueError(f"wacc ({wacc:.4f}) must exceed terminal growth ({g_term:.4f})")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    years = np.arange(1, horizon + 1)
    if fade:
        rates = growth + (g_term - growth) * (years - 1) / max(horizon - 1, 1)
    else:
        rates = np.full(horizon, growth)

    revenue_path = revenue0 * np.cumprod(1.0 + rates)
    fcff_path = margin * revenue_path
    discount = (1.0 + wacc) ** years
    pv_explicit = float(np.sum(fcff_path / discount))

    fcff_terminal = fcff_path[-1] * (1.0 + g_term)
    tv = fcff_terminal / (wacc - g_term)
    pv_terminal = float(tv / discount[-1])

    return pv_explicit + pv_terminal


def terminal_value_share(
    revenue0: float,
    growth: float,
    wacc: float,
    g_term: float,
    margin: float,
    horizon: int = 10,
) -> float:
    total = dcf_value(revenue0, growth, wacc, g_term, margin, horizon)
    no_tv = _explicit_only(revenue0, growth, wacc, g_term, margin, horizon)
    return 1.0 - no_tv / total


def _explicit_only(
    revenue0: float,
    growth: float,
    wacc: float,
    g_term: float,
    margin: float,
    horizon: int,
) -> float:
    years = np.arange(1, horizon + 1)
    rates = growth + (g_term - growth) * (years - 1) / max(horizon - 1, 1)
    revenue_path = revenue0 * np.cumprod(1.0 + rates)
    fcff_path = margin * revenue_path
    return float(np.sum(fcff_path / (1.0 + wacc) ** years))
