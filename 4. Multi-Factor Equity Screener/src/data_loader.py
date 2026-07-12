from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SECTORS = ["Tech", "Financials", "Health", "Energy", "Consumer", "Industrials", "Utilities", "Materials"]

TRUE_IC = {"value": 0.08, "quality": 0.05, "momentum": 0.03}


def generate_synthetic_universe(
    n_firms: int = 200,
    n_periods: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-31", periods=n_periods, freq="ME")
    tickers = [f"SYN{i:03d}" for i in range(n_firms)]
    sector = rng.choice(SECTORS, n_firms)

    rho = 0.95
    expo = {f: np.empty((n_periods, n_firms)) for f in TRUE_IC}
    for f in TRUE_IC:
        expo[f][0] = rng.normal(0, 1, n_firms)
        for t in range(1, n_periods):
            expo[f][t] = rho * expo[f][t - 1] + np.sqrt(1 - rho**2) * rng.normal(0, 1, n_firms)

    sec_idx = pd.Categorical(sector, categories=SECTORS).codes
    sec_offset_value = rng.normal(0, 0.8, len(SECTORS))[sec_idx]
    sec_offset_roe = rng.normal(0, 0.6, len(SECTORS))[sec_idx]

    rows = []
    vol = 0.05
    for t, dt in enumerate(dates):
        zv, zq, zm = expo["value"][t], expo["quality"][t], expo["momentum"][t]
        book_to_price = np.exp(0.3 * (zv + sec_offset_value)) * 0.5
        roe = 0.12 + 0.05 * (zq + sec_offset_roe) + rng.normal(0, 0.01, n_firms)
        accruals = -0.03 * zq + rng.normal(0, 0.02, n_firms)
        ret_12_1 = 0.10 * zm + rng.normal(0, 0.03, n_firms)

        sector_shock = rng.normal(0, 0.015, len(SECTORS))[sec_idx]
        fwd = (
            vol * (TRUE_IC["value"] * zv + TRUE_IC["quality"] * zq + TRUE_IC["momentum"] * zm)
            + sector_shock
            + rng.normal(0, vol, n_firms)
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": dt,
                    "ticker": tickers,
                    "sector": sector,
                    "book_to_price": book_to_price,
                    "roe": roe,
                    "accruals": accruals,
                    "ret_12_1": ret_12_1,
                    "fwd_return": fwd,
                }
            )
        )
    panel = pd.concat(rows).set_index(["date", "ticker"]).sort_index()
    return panel


def load_universe(**synth_kwargs) -> pd.DataFrame:
    fmp_key = os.getenv("FMP_API_KEY", "")
    if fmp_key:
        print(
            "[data_loader] FMP_API_KEY detected, but the live panel loader needs a "
            "ticker list + point-in-time care; see README. Using synthetic universe "
            "for reproducibility."
        )
    return generate_synthetic_universe(**synth_kwargs)
