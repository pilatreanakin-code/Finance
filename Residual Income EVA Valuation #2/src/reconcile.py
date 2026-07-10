"""Reconcile the residual-income value against a simple DCF (DDM) and
decompose the gap.

Clean-surplus identity (exact, any horizon T):

    BV0 + Σ PV(RI_t) = Σ PV(Div_t) + PV(BV_T)

so once explicit forecasts agree, **the entire RI-vs-DCF gap is a
terminal-assumption artifact**: PV(BV_T) + PV(terminal RI) − PV(terminal
dividends). Decomposing it shows exactly which terminal claim drives the
disagreement.
"""

from __future__ import annotations

import numpy as np

from residual_income import RIForecast, ddm_value, ri_value


def reconcile(
    book0: float,
    forecast: RIForecast,
    r: float,
    g_term_ri: float = 0.0,
    g_term_ddm: float = 0.03,
) -> dict[str, float]:
    """Value the same forecast both ways and decompose the difference.

    Parameters
    ----------
    book0 : float
        Current book equity.
    forecast : RIForecast
        Shared explicit-horizon forecast.
    r : float
        Cost of equity.
    g_term_ri : float
        Terminal RI growth (0 = excess returns persist flat; negative = fade).
    g_term_ddm : float
        Terminal dividend growth for the DCF/DDM comparator (the usual
        2–3.5% GDP-ish assumption that makes DCFs terminal-heavy).

    Returns
    -------
    dict
        Both values, the gap, the exact gap decomposition, and each method's
        terminal-value share of total value.
    """
    ri = ri_value(book0, forecast, r, g_term_ri)
    ddm = ddm_value(forecast, r, g_term_ddm)

    horizon = len(forecast.dividends)
    disc_T = (1.0 + r) ** horizon
    pv_end_book = float(forecast.book_end[-1] / disc_T)

    gap = ri["value"] - ddm["value"]
    # Exact identity: gap = PV(BV_T) + PV(terminal RI) − PV(terminal dividends)
    gap_decomp = {
        "pv_end_book": pv_end_book,
        "pv_terminal_ri": ri["pv_terminal_ri"],
        "minus_pv_terminal_div": -ddm["pv_terminal_div"],
    }

    return {
        "ri_value": ri["value"],
        "ddm_value": ddm["value"],
        "gap": gap,
        "gap_decomposition": gap_decomp,
        "gap_identity_residual": gap - sum(gap_decomp.values()),
        "ri_terminal_share": ri["pv_terminal_ri"] / ri["value"],
        "ddm_terminal_share": ddm["pv_terminal_div"] / ddm["value"],
        "ri_components": ri,
        "ddm_components": ddm,
    }


def terminal_sensitivity(
    book0: float,
    forecast: RIForecast,
    r: float,
    g_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Value of each method across terminal-growth assumptions.

    The RI value moves far less with ``g`` than the DDM value because most of
    the RI value sits in today's book plus near-dated residual income — the
    quantitative core of the skeptical insight.
    """
    ri_vals = np.array([ri_value(book0, forecast, r, g)["value"] for g in g_grid])
    ddm_vals = np.array([ddm_value(forecast, r, g)["value"] for g in g_grid])
    return {"g_grid": g_grid, "ri": ri_vals, "ddm": ddm_vals}
