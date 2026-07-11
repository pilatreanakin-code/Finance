"""Data layer: synthetic financials (default) and guarded live loaders.

The synthetic generator is the default path so the whole project runs offline
with zero API keys. Live loaders (FMP for fundamentals, FRED for the risk-free
rate) are only attempted when the corresponding environment variables are set,
and results are cached to ``data/*.parquet`` (gitignored).

Free-tier notes encoded here:
- FMP free tier is the reliable free fundamentals source; yfinance fundamentals
  are patchy (ratios should be computed manually from statements).
- FRED is the reliable free source for the risk-free rate (e.g. DGS10).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class CompanyFinancials:
    """Historical statement items needed for FCFF plus current market data.

    All statement arrays are aligned oldest -> newest, in currency units
    (e.g. USD millions).
    """

    ticker: str
    statements: pd.DataFrame  # columns: revenue, ebit, tax_rate, dep_amort, capex, nwc
    market_ev: float  # current enterprise value
    net_debt: float
    shares_outstanding: float
    risk_free_rate: float = 0.04

    @property
    def equity_value(self) -> float:
        """Market equity value implied by EV and net debt."""
        return self.market_ev - self.net_debt


def generate_synthetic_financials(
    ticker: str = "SYNTH",
    n_years: int = 8,
    base_revenue: float = 10_000.0,
    revenue_growth: float = 0.08,
    ebit_margin: float = 0.18,
    tax_rate: float = 0.23,
    seed: int = 42,
) -> CompanyFinancials:
    """Generate a plausible synthetic company history for offline runs.

    Revenue follows a noisy growth path; EBIT margin, D&A, CapEx and net
    working capital scale with revenue plus noise, so the derived FCFF series
    behaves like a real (if idealized) company.

    Parameters
    ----------
    ticker : str
        Label only; no network access happens for synthetic data.
    n_years : int
        Number of historical fiscal years to generate.
    base_revenue : float
        Revenue in the first historical year (currency units).
    revenue_growth : float
        Mean annual revenue growth of the synthetic history.
    ebit_margin : float
        Mean EBIT margin.
    tax_rate : float
        Effective cash tax rate.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    CompanyFinancials
        Synthetic statements plus a market EV set at a modest premium to a
        conservative intrinsic anchor, so the demo has a meaningful
        implied-growth solution.
    """
    rng = np.random.default_rng(seed)
    years = pd.RangeIndex(2018, 2018 + n_years, name="year")

    growth_noise = rng.normal(0.0, 0.02, n_years)
    revenue = base_revenue * np.cumprod(1 + revenue_growth + growth_noise)

    margin = np.clip(ebit_margin + rng.normal(0.0, 0.01, n_years), 0.02, 0.60)
    ebit = revenue * margin
    dep_amort = revenue * np.clip(0.045 + rng.normal(0, 0.003, n_years), 0.01, 0.15)
    capex = revenue * np.clip(0.055 + rng.normal(0, 0.004, n_years), 0.01, 0.20)
    nwc = revenue * np.clip(0.10 + rng.normal(0, 0.005, n_years), 0.02, 0.30)

    statements = pd.DataFrame(
        {
            "revenue": revenue,
            "ebit": ebit,
            "tax_rate": np.full(n_years, tax_rate),
            "dep_amort": dep_amort,
            "capex": capex,
            "nwc": nwc,
        },
        index=years,
    )

    # Market EV anchored to a simple perpetuity of the last FCFF so the
    # reverse-DCF has a sensible root: EV = FCFF1 / (wacc - g) with wacc 9%, g 3%.
    last = statements.iloc[-1]
    fcff_last = (
        last["ebit"] * (1 - last["tax_rate"])
        + last["dep_amort"]
        - last["capex"]
        - (statements["nwc"].iloc[-1] - statements["nwc"].iloc[-2])
    )
    anchor_ev = fcff_last * 1.05 / (0.09 - 0.03)
    market_ev = float(anchor_ev * 1.25)  # priced at a 25% premium to the anchor

    return CompanyFinancials(
        ticker=ticker,
        statements=statements,
        market_ev=market_ev,
        net_debt=float(0.8 * fcff_last * 4),
        shares_outstanding=1_000.0,
        risk_free_rate=0.04,
    )


def load_financials(ticker: str = "SYNTH", **synth_kwargs: float) -> CompanyFinancials:
    """Load company financials: live via FMP/FRED if keys exist, else synthetic.

    The live path requires ``FMP_API_KEY`` (and optionally ``FRED_API_KEY``)
    in the environment or a local ``.env``. Without keys this silently falls
    back to the synthetic generator so the project runs offline.
    """
    fmp_key = os.getenv("FMP_API_KEY", "")
    if fmp_key:
        try:
            return _load_financials_fmp(ticker, fmp_key)
        except Exception as exc:  # network/parse failure -> offline fallback
            print(f"[data_loader] Live FMP load failed ({exc}); using synthetic data.")
    return generate_synthetic_financials(ticker=ticker, **synth_kwargs)


def _load_financials_fmp(ticker: str, api_key: str) -> CompanyFinancials:
    """Fetch income-statement and cash-flow items from FMP (free tier).

    Cached to ``data/<ticker>_fmp.parquet``. Only called when a key is set.
    """
    import requests  # local import: not needed offline

    cache = DATA_DIR / f"{ticker}_fmp.parquet"
    if cache.exists():
        statements = pd.read_parquet(cache)
    else:
        base = "https://financialmodelingprep.com/api/v3"
        inc = requests.get(
            f"{base}/income-statement/{ticker}?limit=10&apikey={api_key}", timeout=30
        ).json()
        cf = requests.get(
            f"{base}/cash-flow-statement/{ticker}?limit=10&apikey={api_key}", timeout=30
        ).json()
        bs = requests.get(
            f"{base}/balance-sheet-statement/{ticker}?limit=10&apikey={api_key}", timeout=30
        ).json()
        rows = []
        for i, c, b in zip(reversed(inc), reversed(cf), reversed(bs)):
            tax = (
                i["incomeTaxExpense"] / i["incomeBeforeTax"]
                if i.get("incomeBeforeTax")
                else 0.21
            )
            rows.append(
                {
                    "year": int(str(i["date"])[:4]),
                    "revenue": float(i["revenue"]),
                    "ebit": float(i["operatingIncome"]),
                    "tax_rate": float(np.clip(tax, 0.0, 0.5)),
                    "dep_amort": float(c["depreciationAndAmortization"]),
                    "capex": float(abs(c["capitalExpenditure"])),
                    "nwc": float(b["totalCurrentAssets"] - b["totalCurrentLiabilities"]),
                }
            )
        statements = pd.DataFrame(rows).set_index("year")
        DATA_DIR.mkdir(exist_ok=True)
        statements.to_parquet(cache)

    profile_ev = _fetch_ev_fmp(ticker, api_key)
    rf = _fetch_risk_free_fred()
    return CompanyFinancials(
        ticker=ticker,
        statements=statements,
        market_ev=profile_ev["ev"],
        net_debt=profile_ev["net_debt"],
        shares_outstanding=profile_ev["shares"],
        risk_free_rate=rf,
    )


def _fetch_ev_fmp(ticker: str, api_key: str) -> dict:
    """Current enterprise value, net debt, and share count from FMP."""
    import requests

    url = (
        "https://financialmodelingprep.com/api/v3/enterprise-values/"
        f"{ticker}?limit=1&apikey={api_key}"
    )
    d = requests.get(url, timeout=30).json()[0]
    net_debt = float(d["addTotalDebt"] - d["minusCashAndCashEquivalents"])
    return {
        "ev": float(d["enterpriseValue"]),
        "net_debt": net_debt,
        "shares": float(d["numberOfShares"]),
    }


def _fetch_risk_free_fred() -> float:
    """10y Treasury (DGS10) from FRED if ``FRED_API_KEY`` set, else 4% default."""
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
