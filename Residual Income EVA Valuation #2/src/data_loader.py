"""Data layer: synthetic equity-level financials (default) + guarded FMP/FRED loaders.

Synthetic data obeys the clean-surplus relation by construction:
``BV_t = BV_{t-1} + NI_t − Dividends_t`` — the accounting identity residual
income valuation depends on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class EquityFinancials:
    """Equity-level history plus market data.

    ``statements`` columns: ``net_income, dividends, book_equity`` (book equity
    is end-of-year, clean-surplus consistent), indexed by fiscal year
    oldest → newest.
    """

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
    """Generate a clean-surplus-consistent synthetic equity history.

    ROE follows a noisy mean-reverting path around ``roe_mean``; net income is
    ROE × opening book; dividends are a constant payout; book equity rolls
    forward via clean surplus. Market cap is set at a premium to an RI anchor
    so the demo shows a meaningful valuation gap.

    Parameters
    ----------
    ticker : str
        Label only.
    n_years : int
        Historical years to generate.
    book0 : float
        Opening book equity (currency units).
    roe_mean : float
        Long-run return on equity. Above ``cost_of_equity`` ⇒ positive RI.
    payout : float
        Dividend payout ratio of net income.
    cost_of_equity : float
        Discount rate for residual income.
    seed : int
        RNG seed.
    """
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

    # Market cap anchored to book + RI perpetuity, priced at a 15% premium.
    ri_last = ni[-1] - cost_of_equity * book[-2]
    anchor = book[-1] + ri_last / cost_of_equity * 0.8
    return EquityFinancials(
        ticker=ticker,
        statements=statements,
        market_cap=float(anchor * 1.15),
        cost_of_equity=cost_of_equity,
    )


def load_financials(ticker: str = "SYNTH", **synth_kwargs: float) -> EquityFinancials:
    """FMP live load when ``FMP_API_KEY`` is set; synthetic fallback otherwise."""
    fmp_key = os.getenv("FMP_API_KEY", "")
    if fmp_key:
        try:
            return _load_fmp(ticker, fmp_key)
        except Exception as exc:
            print(f"[data_loader] Live FMP load failed ({exc}); using synthetic data.")
    return generate_synthetic_financials(ticker=ticker, **synth_kwargs)


def _load_fmp(ticker: str, api_key: str) -> EquityFinancials:
    """Income statement + balance sheet + market cap from FMP (cached)."""
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
    # CAPM-style cost of equity with a 5% equity risk premium.
    beta = float(prof.get("beta") or 1.0)
    return EquityFinancials(
        ticker=ticker,
        statements=statements,
        market_cap=float(prof["mktCap"]),
        cost_of_equity=rf + beta * 0.05,
    )


def _risk_free_fred() -> float:
    """DGS10 from FRED if ``FRED_API_KEY`` set, else 4% default."""
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
