from __future__ import annotations

import pandas as pd


def winsorize_cross_section(
    factors: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("require 0 <= lower < upper <= 1")

    def _clip(g: pd.DataFrame) -> pd.DataFrame:
        return g.clip(g.quantile(lower), g.quantile(upper), axis=1)

    return factors.groupby(level="date", group_keys=False).apply(_clip)


def sector_neutral_zscore(
    factors: pd.DataFrame, sectors: pd.Series
) -> pd.DataFrame:
    if not factors.index.equals(sectors.index):
        raise ValueError("factors and sectors must share the same index")

    grouper = [factors.index.get_level_values("date"), sectors]

    def _z(g: pd.DataFrame) -> pd.DataFrame:
        std = g.std(ddof=1)
        z = (g - g.mean()) / std.replace(0.0, pd.NA)
        return z.fillna(0.0)

    z = factors.groupby(grouper, group_keys=False).apply(_z)
    return z.reindex(factors.index)
