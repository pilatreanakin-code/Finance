"""Run the residual-income vs DCF reconciliation pipeline and save figures.

Offline by default (synthetic clean-surplus financials); set FMP_API_KEY /
FRED_API_KEY in a local .env to value a real company.
"""

from __future__ import annotations

import numpy as np

from data_loader import load_financials
from reconcile import reconcile, terminal_sensitivity
from residual_income import (
    build_forecast,
    check_clean_surplus,
    historical_residual_income,
)
from plots import plot_reconciliation_bridge, plot_ri_history, plot_terminal_sensitivity

# ==== USER INPUTS ====
TICKER = "SYNTH"       # any real ticker if FMP_API_KEY is set; SYNTH = offline demo
HORIZON = 10           # explicit forecast years
ROE_FADE_TO = 0.10     # ROE fades toward this by the horizon (competition)
PAYOUT = 0.40          # dividend payout ratio in the forecast
G_TERM_RI = 0.0        # terminal residual-income growth (0 = flat excess returns)
G_TERM_DDM = 0.03      # terminal dividend growth for the DCF comparator
SEED = 42
# =====================


def run_pipeline() -> dict:
    """Execute history -> forecast -> RI & DCF values -> gap decomposition."""
    np.random.seed(SEED)

    company = load_financials(TICKER)
    s = company.statements
    r = company.cost_of_equity

    cs_gap = check_clean_surplus(s)
    ri_hist = historical_residual_income(s, r)
    print(f"=== {company.ticker} ===  cost of equity r = {r:.1%}")
    print(f"Clean-surplus max violation: {cs_gap.abs().max():.2e}")
    print(f"Historical RI (last 3y): {ri_hist.tail(3).round(0).to_dict()}")

    book0 = float(s["book_equity"].iloc[-1])
    roe_now = float(s["net_income"].iloc[-1] / s["book_equity"].iloc[-2])
    forecast = build_forecast(book0, roe_now, ROE_FADE_TO, PAYOUT, HORIZON)

    recon = reconcile(book0, forecast, r, G_TERM_RI, G_TERM_DDM)
    print(f"RI value:  {recon['ri_value']:,.0f}  (terminal share {recon['ri_terminal_share']:.0%})")
    print(f"DCF value: {recon['ddm_value']:,.0f}  (terminal share {recon['ddm_terminal_share']:.0%})")
    print(f"Gap: {recon['gap']:,.0f}  | identity residual: {recon['gap_identity_residual']:.2e}")
    print(f"Market cap: {company.market_cap:,.0f}")

    g_grid = np.arange(0.0, 0.0451, 0.005)
    g_grid = g_grid[g_grid < r - 0.005]
    sens = terminal_sensitivity(book0, forecast, r, g_grid)

    p1 = plot_reconciliation_bridge(recon, company.market_cap, company.ticker)
    p2 = plot_terminal_sensitivity(sens, company.ticker)
    p3 = plot_ri_history(ri_hist, company.ticker)
    print(f"Figures saved: {p1.name}, {p2.name}, {p3.name}")

    return {
        "company": company,
        "recon": recon,
        "sens": sens,
        "ri_hist": ri_hist,
        "clean_surplus_gap": cs_gap,
    }


if __name__ == "__main__":
    run_pipeline()
