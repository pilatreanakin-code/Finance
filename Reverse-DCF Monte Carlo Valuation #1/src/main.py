"""Run the full reverse-DCF Monte Carlo pipeline and save figures.

Offline by default (synthetic financials); set FMP_API_KEY / FRED_API_KEY in a
local .env to value a real company.
"""

from __future__ import annotations

import numpy as np

from data_loader import load_financials
from dcf import dcf_value, terminal_value_share
from fcff import base_inputs, compute_fcff
from monte_carlo import MCAssumptions, run_monte_carlo, sensitivity_grid
from reverse_dcf import implied_growth, repricing_error
from plots import (
    plot_fair_value_distribution,
    plot_implied_growth_context,
    plot_sensitivity_heatmap,
)

# ==== USER INPUTS ====
TICKER = "SYNTH"        # any ticker if FMP_API_KEY is set; "SYNTH" = offline demo
WACC = 0.09             # central discount-rate assumption
G_TERM = 0.025          # central terminal growth
HORIZON = 10            # explicit DCF years
CENTRAL_GROWTH = 0.08   # analyst central-case near-term growth (for the MC)
N_DRAWS = 20_000        # Monte Carlo draws (>= 10k per spec)
SEED = 42
# =====================


def run_pipeline() -> dict:
    """Execute FCFF -> reverse DCF -> Monte Carlo -> sensitivity, save figures."""
    np.random.seed(SEED)

    company = load_financials(TICKER)
    fcff_hist = compute_fcff(company.statements)
    base = base_inputs(company.statements)

    print(f"=== {company.ticker} ===")
    print(f"Historical FCFF (last 3y): {fcff_hist.tail(3).round(0).to_dict()}")
    print(f"Base revenue: {base['revenue0']:,.0f} | median FCFF margin: {base['fcff_margin']:.1%}")
    print(f"Market EV: {company.market_ev:,.0f}")

    # --- Reverse DCF: what growth is the market pricing? ---
    g_implied = implied_growth(
        company.market_ev, base["revenue0"], WACC, G_TERM, base["fcff_margin"], HORIZON
    )
    err = repricing_error(
        g_implied, company.market_ev, base["revenue0"], WACC, G_TERM,
        base["fcff_margin"], HORIZON,
    )
    print(f"Market-implied near-term growth: {g_implied:.2%} (repricing error {err:.2e})")

    tv_share = terminal_value_share(
        base["revenue0"], g_implied, WACC, G_TERM, base["fcff_margin"], HORIZON
    )
    print(f"Terminal value share of EV at implied growth: {tv_share:.0%}")

    # --- Monte Carlo fair-value distribution at the central growth case ---
    mc = run_monte_carlo(
        base["revenue0"],
        CENTRAL_GROWTH,
        company.market_ev,
        MCAssumptions(wacc_mean=WACC, g_term_mean=G_TERM, margin_mean=base["fcff_margin"]),
        n_draws=N_DRAWS,
        horizon=HORIZON,
        seed=SEED,
    )
    print(
        f"Fair value p5/p50/p95: {mc.quantiles['p5']:,.0f} / "
        f"{mc.quantiles['p50']:,.0f} / {mc.quantiles['p95']:,.0f}"
    )
    print(f"P(undervalued) = {mc.p_undervalued:.1%}")

    # --- WACC x g sensitivity grid ---
    wacc_range = np.arange(0.07, 0.121, 0.005)
    g_range = np.arange(0.005, 0.041, 0.005)
    grid = sensitivity_grid(
        base["revenue0"], CENTRAL_GROWTH, base["fcff_margin"], wacc_range, g_range, HORIZON
    )

    # --- Figures ---
    hist_growth = company.statements["revenue"].pct_change().dropna().to_numpy()
    p1 = plot_fair_value_distribution(mc, company.ticker)
    p2 = plot_sensitivity_heatmap(grid, wacc_range, g_range, company.market_ev, company.ticker)
    p3 = plot_implied_growth_context(hist_growth, g_implied, company.ticker)
    print(f"Figures saved: {p1.name}, {p2.name}, {p3.name}")

    central_value = dcf_value(
        base["revenue0"], CENTRAL_GROWTH, WACC, G_TERM, base["fcff_margin"], HORIZON
    )
    return {
        "company": company,
        "fcff_hist": fcff_hist,
        "implied_growth": g_implied,
        "repricing_error": err,
        "tv_share": tv_share,
        "central_value": central_value,
        "mc": mc,
        "grid": grid,
    }


if __name__ == "__main__":
    run_pipeline()
