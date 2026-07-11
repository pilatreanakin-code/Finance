"""Free cash flow to the firm (FCFF) from statement items.

FCFF = EBIT * (1 - tax) + D&A - CapEx - ΔNWC
"""

from __future__ import annotations

import pandas as pd


def compute_fcff(statements: pd.DataFrame) -> pd.Series:
    """Compute historical FCFF from aligned statement items.

    Parameters
    ----------
    statements : pd.DataFrame
        Indexed by fiscal year (oldest -> newest) with columns
        ``revenue, ebit, tax_rate, dep_amort, capex, nwc``. ``nwc`` is the
        *level* of net working capital; its first difference is used, so the
        first year of FCFF is NaN and dropped.

    Returns
    -------
    pd.Series
        FCFF per year (first year dropped because ΔNWC is undefined there).
    """
    required = {"ebit", "tax_rate", "dep_amort", "capex", "nwc"}
    missing = required - set(statements.columns)
    if missing:
        raise ValueError(f"statements missing columns: {sorted(missing)}")

    nopat = statements["ebit"] * (1.0 - statements["tax_rate"])
    delta_nwc = statements["nwc"].diff()
    fcff = nopat + statements["dep_amort"] - statements["capex"] - delta_nwc
    return fcff.dropna().rename("fcff")


def fcff_margin(statements: pd.DataFrame) -> pd.Series:
    """Historical FCFF as a fraction of revenue (useful MC margin anchor)."""
    fcff = compute_fcff(statements)
    return (fcff / statements.loc[fcff.index, "revenue"]).rename("fcff_margin")


def base_inputs(statements: pd.DataFrame) -> dict[str, float]:
    """Extract the base-year inputs the forward DCF needs.

    Returns
    -------
    dict
        ``revenue0`` (latest revenue), ``fcff_margin`` (median historical
        FCFF margin - median, not mean, to resist one-off working-capital
        swings), and ``fcff0`` (implied base FCFF).
    """
    margin_hist = fcff_margin(statements)
    margin = float(margin_hist.median())
    revenue0 = float(statements["revenue"].iloc[-1])
    return {
        "revenue0": revenue0,
        "fcff_margin": margin,
        "fcff0": revenue0 * margin,
    }
