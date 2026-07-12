from __future__ import annotations

import numpy as np

from composite import (
    composite_score,
    ic_weights,
    information_coefficients,
    quintile_returns,
)
from data_loader import load_universe
from factors import compute_raw_factors, factor_correlations
from plots import plot_factor_correlation, plot_ic_series, plot_quintile_spread
from standardize import sector_neutral_zscore, winsorize_cross_section

N_FIRMS = 200
N_PERIODS = 24
WINSOR = (0.01, 0.99)
N_QUANTILES = 5
SEED = 42


def run_pipeline() -> dict:
    np.random.seed(SEED)

    panel = load_universe(n_firms=N_FIRMS, n_periods=N_PERIODS, seed=SEED)
    print(f"Panel: {panel.index.get_level_values('ticker').nunique()} firms x "
          f"{panel.index.get_level_values('date').nunique()} periods")

    raw = compute_raw_factors(panel)
    winsorized = winsorize_cross_section(raw, *WINSOR)
    z = sector_neutral_zscore(winsorized, panel["sector"])

    corr = factor_correlations(z)
    print("\nAvg cross-sectional factor correlation:")
    print(corr.round(2))

    ic_ts = information_coefficients(z, panel["fwd_return"])
    print("\nMean ICs:", ic_ts.mean().round(3).to_dict())

    weights = ic_weights(ic_ts)
    comp = composite_score(z, weights)
    q_ret = quintile_returns(comp, panel["fwd_return"], N_QUANTILES)
    print(f"\nMean quintile returns (%/period): "
          f"{(q_ret[[c for c in q_ret.columns if c.startswith('Q')]].mean() * 100).round(3).to_dict()}")
    print(f"Mean Q5-Q1 spread: {q_ret['spread'].mean():.3%}/period "
          f"(t-stat {q_ret['spread'].mean() / q_ret['spread'].sem():.2f})")

    p1 = plot_quintile_spread(q_ret)
    p2 = plot_factor_correlation(corr)
    p3 = plot_ic_series(ic_ts, weights)
    print(f"Figures saved: {p1.name}, {p2.name}, {p3.name}")

    return {
        "panel": panel,
        "z": z,
        "ic_ts": ic_ts,
        "weights": weights,
        "composite": comp,
        "q_ret": q_ret,
        "corr": corr,
    }


if __name__ == "__main__":
    run_pipeline()
