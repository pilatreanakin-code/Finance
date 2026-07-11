import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comps import prepare_panel, winsorize
from data_loader import TRUE_COEF_EV_EBITDA, generate_synthetic_sector
from fair_multiple import fit_fair_multiple, rich_cheap_table


@pytest.fixture(scope="module")
def raw_panel():
    return generate_synthetic_sector(n_firms=120, seed=42)


@pytest.fixture(scope="module")
def panel(raw_panel):
    return prepare_panel(raw_panel)


def test_winsorize_known():
    s = pd.Series(np.arange(101, dtype=float))
    w = winsorize(s, 0.05, 0.95)
    assert w.min() == pytest.approx(5.0)
    assert w.max() == pytest.approx(95.0)
    assert len(w) == len(s)


def test_winsorize_validates_bounds():
    with pytest.raises(ValueError):
        winsorize(pd.Series([1.0, 2.0]), 0.9, 0.1)


def test_panel_no_nans(panel):
    assert not panel.isna().any().any()
    assert (panel["ev_ebitda"] > 0).all()


def test_regression_recovers_true_drivers(panel):
    res = fit_fair_multiple(panel, "ev_ebitda")
    coef = res.coefficients["coef"]
    true = TRUE_COEF_EV_EBITDA
    for k in ("growth", "ebitda_margin", "roic"):
        assert coef[k] > 0, k
    assert coef["leverage"] < 0
    assert coef["growth"] == pytest.approx(true["growth"], rel=0.5)
    assert res.r_squared > 0.3


def test_residual_z_properties(panel):
    res = fit_fair_multiple(panel, "pe")
    z = res.residual_z
    assert abs(z.mean()) < 1e-10
    assert z.std(ddof=1) == pytest.approx(1.0)


def test_ranking_order(panel):
    res = fit_fair_multiple(panel, "ev_ebitda")
    rank = res.ranking()
    assert rank["residual_z"].is_monotonic_increasing
    assert rank["residual"].iloc[0] < 0 < rank["residual"].iloc[-1]


def test_rich_cheap_table(panel):
    results = {m: fit_fair_multiple(panel, m) for m in ("ev_ebitda", "pe")}
    table = rich_cheap_table(results, top_n=5)
    assert (table["flag"] == "CHEAP").sum() == 5
    assert (table["flag"] == "RICH").sum() == 5
    assert table["composite_z"].is_monotonic_increasing


def test_unknown_multiple_raises(panel):
    with pytest.raises(ValueError, match="unknown multiple"):
        fit_fair_multiple(panel, "ev_sales")


def test_full_pipeline_and_figures(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    import main
    import plots

    monkeypatch.setattr(plots, "OUTPUT_DIR", tmp_path)
    res = main.run_pipeline()
    assert res["results"]["ev_ebitda"].r_squared > 0.3
    for name in (
        "actual_vs_fair_ev_ebitda.png",
        "actual_vs_fair_pe.png",
        "residual_ranking.png",
        "winsorization_effect.png",
    ):
        assert (tmp_path / name).exists(), name
