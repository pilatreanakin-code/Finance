"""Offline tests for residual-income valuation (synthetic data only)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import generate_synthetic_financials
from reconcile import reconcile, terminal_sensitivity
from residual_income import (
    build_forecast,
    check_clean_surplus,
    ddm_value,
    ddm_value_no_terminal,
    historical_residual_income,
    ri_value,
)


@pytest.fixture(scope="module")
def company():
    return generate_synthetic_financials(seed=42)


def test_synthetic_clean_surplus(company):
    """Synthetic data must satisfy BV_t = BV_{t-1} + NI_t - Div_t exactly."""
    gap = check_clean_surplus(company.statements)
    assert gap.abs().max() < 1e-8


def test_historical_ri_known_example():
    df = pd.DataFrame(
        {
            "net_income": [100.0, 120.0],
            "dividends": [40.0, 48.0],
            "book_equity": [1000.0, 1072.0],
        },
        index=[2023, 2024],
    )
    ri = historical_residual_income(df, r=0.10)
    # 120 - 0.10 x 1000 = 20
    assert len(ri) == 1
    assert ri.loc[2024] == pytest.approx(20.0)


def test_clean_surplus_identity_exact():
    """BV0 + Σ PV(RI) == Σ PV(Div) + PV(BV_T) - exact for any horizon."""
    book0 = 5000.0
    r = 0.09
    fc = build_forecast(book0, roe_start=0.16, roe_fade_to=0.08, payout=0.45, horizon=12)
    ri = ri_value(book0, fc, r, g_term=0.0)
    lhs = ri["book"] + ri["pv_explicit_ri"]  # no terminal
    rhs = ddm_value_no_terminal(fc, r)
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_ri_less_terminal_sensitive_than_ddm(company):
    """The core claim: RI value swings far less with terminal growth."""
    book0 = float(company.statements["book_equity"].iloc[-1])
    fc = build_forecast(book0, 0.14, 0.10, 0.40, 10)
    g = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    sens = terminal_sensitivity(book0, fc, 0.09, g)
    ri_swing = sens["ri"].max() / sens["ri"].min() - 1
    ddm_swing = sens["ddm"].max() / sens["ddm"].min() - 1
    assert ri_swing < ddm_swing
    assert ddm_swing > 2 * ri_swing  # not marginally less - structurally less


def test_gap_decomposition_identity(company):
    """Acceptance: gap decomposition sums exactly to RI - DCF."""
    book0 = float(company.statements["book_equity"].iloc[-1])
    fc = build_forecast(book0, 0.14, 0.10, 0.40, 10)
    recon = reconcile(book0, fc, 0.09, g_term_ri=0.0, g_term_ddm=0.03)
    assert abs(recon["gap_identity_residual"]) < 1e-6 * abs(recon["ri_value"])
    assert recon["ri_value"] > 0 and recon["ddm_value"] > 0


def test_forecast_clean_surplus_internally():
    fc = build_forecast(1000.0, 0.15, 0.10, 0.5, 8)
    implied_end = fc.book_prev + fc.net_income - fc.dividends
    np.testing.assert_allclose(implied_end, fc.book_end, rtol=1e-12)


def test_rejects_bad_terminal():
    fc = build_forecast(1000.0, 0.15, 0.10, 0.5, 5)
    with pytest.raises(ValueError, match="must exceed"):
        ri_value(1000.0, fc, r=0.05, g_term=0.06)
    with pytest.raises(ValueError, match="must exceed"):
        ddm_value(fc, r=0.05, g_term=0.06)


def test_full_pipeline_and_figures(tmp_path, monkeypatch):
    """Acceptance: both values computed and the bridge figure saved."""
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    import main
    import plots

    monkeypatch.setattr(plots, "OUTPUT_DIR", tmp_path)
    res = main.run_pipeline()
    assert res["recon"]["ri_value"] > 0
    assert res["recon"]["ddm_value"] > 0
    for name in (
        "reconciliation_bridge.png",
        "terminal_sensitivity.png",
        "residual_income_history.png",
    ):
        assert (tmp_path / name).exists(), name
