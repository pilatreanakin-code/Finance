from __future__ import annotations

import numpy as np

from comps import prepare_panel, summary_stats
from data_loader import load_sector_panel
from fair_multiple import fit_fair_multiple, rich_cheap_table
from plots import plot_actual_vs_fair, plot_residual_ranking, plot_winsorization_effect

TICKERS: list[str] | None = None
MULTIPLES = ["ev_ebitda", "pe"]
WINSOR = (0.01, 0.99)
TOP_N = 5
SEED = 42


def run_pipeline() -> dict:
    np.random.seed(SEED)

    raw = load_sector_panel(TICKERS, seed=SEED)
    panel = prepare_panel(raw, *WINSOR)
    print(f"Universe: {len(panel)} firms")
    print(summary_stats(panel).round(2))

    results = {}
    for m in MULTIPLES:
        res = fit_fair_multiple(panel, m)
        results[m] = res
        print(f"\n--- {m.upper()} fair-multiple regression (R² = {res.r_squared:.2f}) ---")
        print(res.coefficients.round(3))

    ranking = rich_cheap_table(results, top_n=TOP_N)
    print("\nCheapest 5 (composite residual z):")
    print(ranking.head(5).round(2))
    print("Richest 5:")
    print(ranking.tail(5).round(2))

    p1 = plot_actual_vs_fair(results["ev_ebitda"])
    p2 = plot_actual_vs_fair(results["pe"])
    p3 = plot_residual_ranking(ranking, top_n=10)
    p4 = plot_winsorization_effect(
        raw["ev_ebitda"].rename("EV/EBITDA"), panel["ev_ebitda"]
    )
    print(f"\nFigures saved: {p1.name}, {p2.name}, {p3.name}, {p4.name}")

    return {"panel": panel, "raw": raw, "results": results, "ranking": ranking}


if __name__ == "__main__":
    run_pipeline()
