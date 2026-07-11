from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TRUE_COEF_EV_EBITDA = {
    "const": 6.0,
    "growth": 30.0,
    "ebitda_margin": 10.0,
    "roic": 15.0,
    "leverage": -1.5,
}
TRUE_COEF_PE = {
    "const": 10.0,
    "growth": 60.0,
    "ebitda_margin": 8.0,
    "roic": 30.0,
    "leverage": -1.0,
}


def generate_synthetic_sector(
    n_firms: int = 60,
    n_outliers: int = 4,
    noise_ev: float = 1.2,
    noise_pe: float = 2.5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    growth = np.clip(rng.normal(0.07, 0.05, n_firms), -0.05, 0.30)
    margin = np.clip(rng.normal(0.22, 0.07, n_firms), 0.05, 0.50)
    roic = np.clip(rng.normal(0.12, 0.05, n_firms), 0.01, 0.35)
    leverage = np.clip(rng.normal(1.8, 1.0, n_firms), 0.0, 5.0)

    c = TRUE_COEF_EV_EBITDA
    ev_ebitda = (
        c["const"] + c["growth"] * growth + c["ebitda_margin"] * margin
        + c["roic"] * roic + c["leverage"] * leverage
        + rng.normal(0, noise_ev, n_firms)
    )
    p = TRUE_COEF_PE
    pe = (
        p["const"] + p["growth"] * growth + p["ebitda_margin"] * margin
        + p["roic"] * roic + p["leverage"] * leverage
        + rng.normal(0, noise_pe, n_firms)
    )

    idx = rng.choice(n_firms, size=n_outliers, replace=False)
    ev_ebitda[idx[: n_outliers // 2]] *= 4.0
    pe[idx[n_outliers // 2:]] *= 5.0

    tickers = [f"SYN{i:03d}" for i in range(n_firms)]
    return pd.DataFrame(
        {
            "growth": growth,
            "ebitda_margin": margin,
            "roic": roic,
            "leverage": leverage,
            "ev_ebitda": np.maximum(ev_ebitda, 0.5),
            "pe": np.maximum(pe, 1.0),
        },
        index=pd.Index(tickers, name="ticker"),
    )


def load_sector_panel(tickers: list[str] | None = None, **synth_kwargs) -> pd.DataFrame:
    fmp_key = os.getenv("FMP_API_KEY", "")
    if fmp_key and tickers:
        try:
            return _load_fmp_panel(tickers, fmp_key)
        except Exception as exc:
            print(f"[data_loader] Live FMP load failed ({exc}); using synthetic sector.")
    return generate_synthetic_sector(**synth_kwargs)


def _load_fmp_panel(tickers: list[str], api_key: str) -> pd.DataFrame:
    import requests

    cache = DATA_DIR / "sector_panel_fmp.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    base = "https://financialmodelingprep.com/api/v3"
    rows = []
    for t in tickers:
        km = requests.get(f"{base}/key-metrics-ttm/{t}?apikey={api_key}", timeout=30).json()
        ratios = requests.get(f"{base}/ratios-ttm/{t}?apikey={api_key}", timeout=30).json()
        growth = requests.get(
            f"{base}/financial-growth/{t}?limit=1&apikey={api_key}", timeout=30
        ).json()
        if not (km and ratios and growth):
            continue
        km, ratios, growth = km[0], ratios[0], growth[0]
        rows.append(
            {
                "ticker": t,
                "growth": float(growth.get("revenueGrowth") or np.nan),
                "ebitda_margin": float(ratios.get("ebitdaratioTTM") or np.nan),
                "roic": float(km.get("roicTTM") or np.nan),
                "leverage": float(km.get("netDebtToEBITDATTM") or np.nan),
                "ev_ebitda": float(km.get("enterpriseValueOverEBITDATTM") or np.nan),
                "pe": float(km.get("peRatioTTM") or np.nan),
            }
        )
    panel = pd.DataFrame(rows).set_index("ticker").dropna()
    DATA_DIR.mkdir(exist_ok=True)
    panel.to_parquet(cache)
    return panel
