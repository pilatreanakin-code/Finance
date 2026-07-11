"""Reverse DCF: solve for the growth rate the market is already pricing.

Instead of asserting a growth forecast and producing a price, invert the DCF:
given the observed enterprise value, find the near-term growth rate ``g*``
such that ``dcf_value(g*) = market_ev``. That number can be judged against
history and base rates - a far more defensible conversation than defending a
point estimate.
"""

from __future__ import annotations

from scipy.optimize import brentq

from dcf import dcf_value

GROWTH_LO = -0.50
GROWTH_HI = 2.00


def implied_growth(
    market_ev: float,
    revenue0: float,
    wacc: float,
    g_term: float,
    margin: float,
    horizon: int = 10,
    tol: float = 1e-10,
) -> float:
    """Solve for the near-term growth rate that reprices the market EV.

    Uses Brent's method on ``dcf_value(growth) - market_ev`` over a wide
    growth bracket. ``dcf_value`` is strictly increasing in growth, so any
    sign change in the bracket contains exactly one root.

    Parameters
    ----------
    market_ev : float
        Observed enterprise value to match.
    revenue0, wacc, g_term, margin, horizon
        Passed through to :func:`dcf.dcf_value`.
    tol : float
        Absolute solver tolerance on growth.

    Returns
    -------
    float
        Implied near-term annual revenue growth.

    Raises
    ------
    ValueError
        If no root exists in [-50%, +200%] growth - i.e. the market EV cannot
        be justified by any sane growth under the given wacc/margin, which is
        itself diagnostic (check margin and wacc inputs).
    """
    def objective(g: float) -> float:
        return dcf_value(revenue0, g, wacc, g_term, margin, horizon) - market_ev

    f_lo, f_hi = objective(GROWTH_LO), objective(GROWTH_HI)
    if f_lo * f_hi > 0:
        raise ValueError(
            "No implied-growth root in [-50%, 200%]: "
            f"value({GROWTH_LO:.0%})={f_lo + market_ev:,.0f}, "
            f"value({GROWTH_HI:.0%})={f_hi + market_ev:,.0f}, "
            f"target EV={market_ev:,.0f}. Check wacc/margin inputs."
        )
    root = brentq(objective, GROWTH_LO, GROWTH_HI, xtol=tol)
    return float(root)


def repricing_error(
    growth: float,
    market_ev: float,
    revenue0: float,
    wacc: float,
    g_term: float,
    margin: float,
    horizon: int = 10,
) -> float:
    """Relative error between DCF at ``growth`` and the market EV.

    Used by tests to verify the solver actually re-prices the company.
    """
    value = dcf_value(revenue0, growth, wacc, g_term, margin, horizon)
    return abs(value - market_ev) / market_ev
