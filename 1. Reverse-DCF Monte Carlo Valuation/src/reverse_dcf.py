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
    value = dcf_value(revenue0, growth, wacc, g_term, margin, horizon)
    return abs(value - market_ev) / market_ev
