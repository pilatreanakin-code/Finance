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
    ri = ri_value(book0, forecast, r, g_term_ri)
    ddm = ddm_value(forecast, r, g_term_ddm)

    horizon = len(forecast.dividends)
    disc_T = (1.0 + r) ** horizon
    pv_end_book = float(forecast.book_end[-1] / disc_T)

    gap = ri["value"] - ddm["value"]
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
    ri_vals = np.array([ri_value(book0, forecast, r, g)["value"] for g in g_grid])
    ddm_vals = np.array([ddm_value(forecast, r, g)["value"] for g in g_grid])
    return {"g_grid": g_grid, "ri": ri_vals, "ddm": ddm_vals}
