from __future__ import annotations

import pandas as pd

FACTOR_NAMES = ["value", "quality", "momentum"]


def compute_raw_factors(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"book_to_price", "roe", "accruals", "ret_12_1"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=panel.index)
    out["value"] = panel["book_to_price"]
    out["quality"] = panel["roe"] - panel["accruals"]
    out["momentum"] = panel["ret_12_1"]
    return out


def factor_correlations(z_scores: pd.DataFrame) -> pd.DataFrame:
    per_date = z_scores.groupby(level="date").corr()
    return per_date.groupby(level=1).mean().loc[FACTOR_NAMES, FACTOR_NAMES]
