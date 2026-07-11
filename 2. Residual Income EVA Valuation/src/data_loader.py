from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class EquityFinancials:

    ticker: str
    statements: pd.DataFrame
    market_cap: float
    cost_of_equity: float = 0.09


def generate_synthetic_financials(
    ticker: str = "SYNTH",
    n_years: int = 10,
    book0: float = 5_000.0,
    roe_mean: float = 0.14,
    payout: float = 0.40,
    cost_of_equity: float = 0.09,
    seed: int = 42,
) -> EquityFinancials:
    rng = np.random.default_rng(seed)
    years = pd.RangeIndex(2016, 2016 + n_years, name="year")

    roe = np.empty(n_years)
    roe[0] = roe_mean
    for t in range(1, n_years):
        roe[t] = roe[t - 1] + 0.5 * (roe_mean - roe[t - 1]) + rng.normal(0.0, 0.012)
    roe = np.clip(roe, 0.02, 0.35)

    book = np.empty(n_years + 1)
    book[0] = book0
    ni = np.empty(n_years)
    div = np.empty(n_years)
    for t in range(n_years):
        ni[t] = roe[t] * book[t]
        div[t] = payout * ni[t]
        book[t + 1] = book[t] + ni[t] - div[t]

    statements = pd.DataFrame(
        {"net_income": ni, "dividends": div, "book_equity": book[1:]},
        index=years,
    )

    ri_last = ni[-1] - cost_of_equity * book[-2]
    anchor = book[-1] + ri_last / cost_of_equity * 0.8
    return EquityFinancials(
        ticker=ticker,
        statements=statements,
        market_cap=float(anchor * 1.15),
        cost_of_equity=cost_of_equity,
    )


def load_financials(ticker: str = "SYNTH", **synth_kwargs: float) -> EquityFinancials:
    fmp_key = os.getenv("FMP_API_KEY", "")
    if fmp_key:
        try:
            return _load_fmp(ticker, fmp_key)
        except Exception as exc:
            print(f"[data_loader] Live FMP load failed ({exc}); using synthetic data.")
    return generate_synthetic_financials(ticker=ticker, **synth_kwargs)


def _load_fmp(ticker: str, api_key: str) -> EquityFinancials:
    import requests

    cache = DATA_DIR / f"{ticker}_ri_fmp.parquet"
    if cache.exists():
        statements = pd.read_parquet(cache)
    else:
        base = "https://financialmodelingprep.com/api/v3"
        inc = requests.get(
            f"{base}/income-statement/{ticker}?limit=12&apikey={api_key}", timeout=30
        ).json()
        bs = requests.get(
            f"{base}/balance-sheet-statement/{ticker}?limit=12&apikey={api_key}", timeout=30
        ).json()
        cf = requests.get(
            f"{base}/cash-flow-statement/{ticker}?limit=12&apikey={api_key}", timeout=30
        ).json()
        rows = []
        for i, b, c in zip(reversed(inc), reversed(bs), reversed(cf)):
            rows.append(
                {
                    "year": int(str(i["date"])[:4]),
                    "net_income": float(i["netIncome"]),
                    "dividends": float(abs(c.get("dividendsPaid", 0.0) or 0.0)),
                    "book_equity": float(b["totalStockholdersEquity"]),
                }
            )
        statements = pd.DataFrame(rows).set_index("year")
        DATA_DIR.mkdir(exist_ok=True)
        statements.to_parquet(cache)

    prof = requests.get(
        f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}",
        timeout=30,
    ).json()[0]
    rf = _risk_free_fred()
    beta = float(prof.get("beta") or 1.0)
    return EquityFinancials(
        ticker=ticker,
        statements=statements,
        market_cap=float(prof["mktCap"]),
        cost_of_equity=rf + beta * 0.05,
    )


def _risk_free_fred() -> float:
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        return 0.04
    import requests

    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=DGS10&api_key={fred_key}&file_type=json&sort_order=desc&limit=5"
    )
    obs = requests.get(url, timeout=30).json()["observations"]
    for o in obs:
        if o["value"] not in (".", ""):
            return float(o["value"]) / 100.0
    return 0.04
