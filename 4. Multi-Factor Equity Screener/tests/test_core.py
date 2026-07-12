import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from composite import (
    composite_score,
    ic_weights,
    information_coefficients,
    quintile_returns,
)
from data_loader import TRUE_IC, generate_synthetic_universe
from factors import compute_raw_factors, factor_correlations
from standardize import sector_neutral_zscore, winsorize_cross_section


@pytest.fixture(scope="module")
def panel():
    return generate_synthetic_universe(n_firms=200, n_periods=24, seed=42)


@pytest.fixture(scope="module")
def z(panel):
    raw = compute_raw_factors(panel)
    w = winsorize_cross_section(raw)
    return sector_neutral_zscore(w, panel["sector"])


def test_panel_shape(panel):
    assert panel.index.get_level_values("ticker").nunique() == 200
    assert panel.index.get_level_values("date").nunique() == 24
    assert not panel[["book_to_price", "roe", "fwd_return"]].isna().any().any()


def test_raw_factors_columns(panel):
    raw = compute_raw_factors(panel)
    assert list(raw.columns) == ["value", "quality", "momentum"]
    with pytest.raises(ValueError, match="missing columns"):
        compute_raw_factors(panel[["sector"]])


def test_winsorize_caps_within_date(panel):
    raw = compute_raw_factors(panel)
    w = winsorize_cross_section(raw, 0.01, 0.99)
    dt = raw.index.get_level_values("date")[0]
    g_raw, g_w = raw.loc[dt], w.loc[dt]
    assert g_w["value"].max() <= g_raw["value"].quantile(0.99) + 1e-12
    assert g_w["value"].min() >= g_raw["value"].quantile(0.01) - 1e-12
    assert len(g_w) == len(g_raw)


def test_sector_neutral_z_means(panel, z):
    grouped = z.groupby(
        [z.index.get_level_values("date"), panel["sector"]]
    ).mean()
    assert grouped.abs().max().max() < 1e-10


def test_ic_recovers_planted_ordering(panel, z):
    ic_ts = information_coefficients(z, panel["fwd_return"])
    means = ic_ts.mean()
    assert means["value"] > means["quality"] > means["momentum"]
    assert (means > 0).all()
    assert means["value"] == pytest.approx(TRUE_IC["value"], abs=0.05)


def test_ic_weights_no_lookahead(panel, z):
    ic_ts = information_coefficients(z, panel["fwd_return"])
    w = ic_weights(ic_ts, min_history=3)
    last = ic_ts.index[-1]
    trailing = ic_ts.loc[: ic_ts.index[-2]].mean().clip(lower=0)
    expected = trailing / trailing.sum()
    pd.testing.assert_series_equal(
        w.loc[last], expected, check_names=False, atol=1e-12
    )
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all().all()


def test_quintile_spread_positive(panel, z):
    ic_ts = information_coefficients(z, panel["fwd_return"])
    w = ic_weights(ic_ts)
    comp = composite_score(z, w)
    q = quintile_returns(comp, panel["fwd_return"])
    assert q["spread"].mean() > 0
    assert q["Q5"].mean() > q["Q1"].mean()


def test_factor_correlations_shape(z):
    corr = factor_correlations(z)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)


def test_full_pipeline_and_figures(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import main
    import plots

    monkeypatch.setattr(plots, "OUTPUT_DIR", tmp_path)
    res = main.run_pipeline()
    assert res["q_ret"]["spread"].mean() > 0
    for name in ("quintile_spread.png", "factor_correlation.png", "ic_and_weights.png"):
        assert (tmp_path / name).exists(), name
