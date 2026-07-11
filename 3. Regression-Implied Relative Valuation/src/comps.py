from __future__ import annotations

import pandas as pd

DRIVERS = ["growth", "ebitda_margin", "roic", "leverage"]
MULTIPLES = ["ev_ebitda", "pe"]


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("require 0 <= lower < upper <= 1")
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def prepare_panel(
    panel: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    missing = [c for c in DRIVERS + MULTIPLES if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    out = panel[DRIVERS + MULTIPLES].dropna().copy()
    for col in DRIVERS + MULTIPLES:
        out[col] = winsorize(out[col], lower, upper)
    return out


def summary_stats(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[DRIVERS + MULTIPLES].describe().T[["mean", "std", "25%", "50%", "75%"]]
