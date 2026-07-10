"""Offline tests for the reverse-DCF Monte Carlo pipeline (synthetic data only)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import generate_synthetic_financials
from dcf import dcf_value, terminal_value_share
from fcff import base_inputs, compute_fcff
from monte_carlo import MCAssumptions, run_monte_carlo, sensitivity_grid
from reverse_dcf import implied_growth, repricing_error


@pytest.fixture(scope="module")
def company():
    return generate_synthetic_financials(seed=42)


@pytest.fixture(scope="module")
def base(company):
    return base_inputs(company.statements)


def test_fcff_known_values():
    """FCFF = EBIT(1−τ) + D&A − CapEx − ΔNWC on a hand-checked example."""
    df = pd.DataFrame(
        {
            "revenue": [100.0, 110.0],
            "ebit": [20.0, 22.0],
            "tax_rate": [0.25, 0.25],
            "dep_amort": [5.0, 5.5],
            "capex": [6.0, 6.6],
            "nwc": [10.0, 11.0],
        },
        index=[2023, 2024],
    )
    fcff = compute_fcff(df)
    # 22*0.75 + 5.5 - 6.6 - (11-10) = 16.5 + 5.5 - 6.6 - 1 = 14.4
    assert len(fcff) == 1
    assert fcff.loc[2024] == pytest.approx(14.4)


def test_fcff_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        compute_fcff(pd.DataFrame({"ebit": [1.0]}))


def test_synthetic_financials_sane(company):
    s = company.statements
    assert not s.isna().any().any()
    assert (s["revenue"] > 0).all()
    assert company.market_ev > 0
    assert len(compute_fcff(s)) == len(s) - 1


def test_dcf_monotonicity(base):
    """Value increases in growth and margin, decreases in WACC."""
    kw = dict(revenue0=base["revenue0"], g_term=0.025, margin=base["fcff_margin"])
    v_low_g = dcf_value(growth=0.02, wacc=0.09, **kw)
    v_high_g = dcf_value(growth=0.10, wacc=0.09, **kw)
    assert v_high_g > v_low_g

    v_low_w = dcf_value(growth=0.05, wacc=0.07, **kw)
    v_high_w = dcf_value(growth=0.05, wacc=0.12, **kw)
    assert v_low_w > v_high_w

    v_thin = dcf_value(base["revenue0"], 0.05, 0.09, 0.025, margin=0.08)
    v_fat = dcf_value(base["revenue0"], 0.05, 0.09, 0.025, margin=0.16)
    assert v_fat > v_thin
    assert v_fat == pytest.approx(2 * v_thin)  # value is linear in margin


def test_dcf_rejects_bad_terminal():
    with pytest.raises(ValueError, match="must exceed"):
        dcf_value(1000.0, 0.05, wacc=0.03, g_term=0.04, margin=0.1)


def test_terminal_value_dominates(base):
    """TV should be a large EV share — the fragility the README leads with."""
    share = terminal_value_share(base["revenue0"], 0.08, 0.09, 0.025, base["fcff_margin"])
    assert 0.4 < share < 0.95


def test_implied_growth_reprices_ev(company, base):
    """Acceptance: solver root re-prices market EV within tolerance."""
    g = implied_growth(
        company.market_ev, base["revenue0"], 0.09, 0.025, base["fcff_margin"]
    )
    err = repricing_error(
        g, company.market_ev, base["revenue0"], 0.09, 0.025, base["fcff_margin"]
    )
    assert err < 1e-8
    assert -0.5 < g < 2.0


def test_implied_growth_no_root_raises(base):
    """An EV no sane growth can justify must raise, not return garbage."""
    with pytest.raises(ValueError, match="No implied-growth root"):
        implied_growth(1e15, base["revenue0"], 0.09, 0.025, base["fcff_margin"])


def test_monte_carlo_distribution(company, base):
    """Acceptance: MC produces a ≥10k-draw distribution with sane stats."""
    mc = run_monte_carlo(
        base["revenue0"], 0.08, company.market_ev,
        MCAssumptions(margin_mean=base["fcff_margin"]),
        n_draws=10_000, seed=7,
    )
    assert mc.values.shape == (10_000,)
    assert np.isfinite(mc.values).all()
    assert (mc.values > 0).all()
    assert 0.0 <= mc.p_undervalued <= 1.0
    assert mc.quantiles["p5"] < mc.quantiles["p50"] < mc.quantiles["p95"]
    # wacc > g_term enforced on every draw
    assert (mc.draws["wacc"] - mc.draws["g_term"] >= 0.01 - 1e-12).all()


def test_monte_carlo_reproducible(base):
    a = run_monte_carlo(base["revenue0"], 0.08, 1e5, n_draws=1000, seed=11)
    b = run_monte_carlo(base["revenue0"], 0.08, 1e5, n_draws=1000, seed=11)
    np.testing.assert_array_equal(a.values, b.values)


def test_sensitivity_grid(base):
    wacc_r = np.array([0.07, 0.09, 0.11])
    g_r = np.array([0.01, 0.02, 0.03])
    grid = sensitivity_grid(base["revenue0"], 0.08, base["fcff_margin"], wacc_r, g_r)
    assert grid.shape == (3, 3)
    assert np.isfinite(grid).all()  # all cells valid here (wacc > g + 0.5%)
    # value decreasing left→right (higher wacc), increasing bottom→top rows (higher g)
    assert (np.diff(grid, axis=1) < 0).all()
    assert (np.diff(grid, axis=0) > 0).all()


def test_full_pipeline_and_figures(tmp_path, monkeypatch):
    """End-to-end main.run_pipeline() runs offline and writes the figures."""
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    import main
    import plots

    monkeypatch.setattr(plots, "OUTPUT_DIR", tmp_path)
    res = main.run_pipeline()
    assert res["repricing_error"] < 1e-8
    assert res["mc"].values.shape[0] >= 10_000
    for name in (
        "fair_value_distribution.png",
        "wacc_growth_sensitivity.png",
        "implied_vs_historical_growth.png",
    ):
        assert (tmp_path / name).exists(), name
