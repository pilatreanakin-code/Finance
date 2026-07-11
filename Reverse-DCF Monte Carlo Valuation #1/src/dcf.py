"""Forward DCF: enterprise value as a function of (growth, wacc, g_term, margin).

Revenue grows at ``growth`` for an explicit horizon (with optional linear fade
toward the terminal rate), FCFF is ``margin x revenue``, flows are discounted
at ``wacc``, and a Gordon terminal value is added at the horizon.
"""

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
    """Enterprise value from a margin-driven FCFF DCF.

    Parameters
    ----------
    revenue0 : float
        Base-year revenue (currency units).
    growth : float
        Near-term annual revenue growth (year 1). With ``fade=True`` the
        growth rate declines linearly to ``g_term`` by the horizon year,
        which avoids the discontinuity of growing at 15% in year 10 and 2.5%
        in year 11.
    wacc : float
        Discount rate. Must exceed ``g_term``.
    g_term : float
        Terminal (perpetuity) growth rate.
    margin : float
        FCFF margin applied to revenue in every projection year.
    horizon : int
        Explicit projection years.
    fade : bool
        Linearly fade growth to ``g_term`` over the horizon.

    Returns
    -------
    float
        Enterprise value (PV of explicit FCFF + PV of terminal value).

    Raises
    ------
    ValueError
        If ``wacc <= g_term`` (Gordon terminal value undefined) or
        ``horizon < 1``.
    """
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
    """Fraction of total EV contributed by the terminal value.

    Typically 60-80% - the quantitative reason single-point DCFs are fragile.
    """
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
    """PV of the explicit-horizon FCFF only (no terminal value)."""
    years = np.arange(1, horizon + 1)
    rates = growth + (g_term - growth) * (years - 1) / max(horizon - 1, 1)
    revenue_path = revenue0 * np.cumprod(1.0 + rates)
    fcff_path = margin * revenue_path
    return float(np.sum(fcff_path / (1.0 + wacc) ** years))
