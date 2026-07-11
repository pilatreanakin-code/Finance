from __future__ import annotations

import pandas as pd


def compute_fcff(statements: pd.DataFrame) -> pd.Series:
    required = {"ebit", "tax_rate", "dep_amort", "capex", "nwc"}
    missing = required - set(statements.columns)
    if missing:
        raise ValueError(f"statements missing columns: {sorted(missing)}")

    nopat = statements["ebit"] * (1.0 - statements["tax_rate"])
    delta_nwc = statements["nwc"].diff()
    fcff = nopat + statements["dep_amort"] - statements["capex"] - delta_nwc
    return fcff.dropna().rename("fcff")


def fcff_margin(statements: pd.DataFrame) -> pd.Series:
    fcff = compute_fcff(statements)
    return (fcff / statements.loc[fcff.index, "revenue"]).rename("fcff_margin")


def base_inputs(statements: pd.DataFrame) -> dict[str, float]:
    margin_hist = fcff_margin(statements)
    margin = float(margin_hist.median())
    revenue0 = float(statements["revenue"].iloc[-1])
    return {
        "revenue0": revenue0,
        "fcff_margin": margin,
        "fcff0": revenue0 * margin,
    }
