<!-- suggested repo name: reverse-dcf-monte-carlo-valuation -->
# Reverse-DCF Monte Carlo Valuation

Invert the DCF to reveal the growth the market is pricing, then Monte-Carlo the assumptions into a fair-value distribution.

## The skeptical insight

A single-point DCF is the least defensible object in finance - WACC and terminal growth alone swing it 100%+. This inverts the DCF to ask what growth the market is *already* pricing, and Monte-Carlos the assumptions into a fair-value distribution instead of one false-precision number.

## What it does

The pipeline builds historical free cash flow to the firm (FCFF = EBIT(1-τ) + D&A - CapEx - ΔNWC) from statement data, then runs a margin-driven forward DCF expressed as `value(growth, wacc, g_term, margin)`. Instead of guessing growth, it uses Brent root-finding to solve for the near-term growth rate that exactly reproduces the observed enterprise value - the market-implied growth you can then judge against delivered history. A Monte Carlo engine (20,000 draws) samples WACC, terminal growth, and FCFF margin from truncated distributions to produce a fair-value distribution and P(undervalued), and a WACC x terminal-growth grid quantifies exactly how fragile any single point estimate is.

## Methodology

- **FCFF construction** from income-statement and balance-sheet items, with the median historical FCFF margin (not the mean) as the base-rate anchor.
- **Forward DCF** with a linear growth fade to the terminal rate over a 10-year horizon and a Gordon terminal value, which is rejected outright when `wacc <= g_term`.
- **Reverse DCF**: `scipy.optimize.brentq` on `value(growth) - market EV` over a [-50%, +200%] growth bracket; the DCF is strictly increasing in growth so the root is unique.
- **Monte Carlo**: truncated-normal draws of (WACC, g_term, margin) with `wacc - g_term >= 1%` enforced draw-by-draw; outputs quantiles and P(fair value > market EV).
- **Sensitivity grid**: enterprise value across WACC x terminal growth, reported as premium/discount to the market EV.

## How to run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pilatreanakin-code/Finance/blob/main/1.%20Reverse-DCF%20Monte%20Carlo%20Valuation/notebooks/demo.ipynb)

```
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on Unix)
pip install numpy>=1.26 pandas>=2.0 scipy>=1.11 matplotlib>=3.8 pytest>=8.0 requests>=2.31
pytest                        # offline, synthetic data
jupyter notebook notebooks/demo.ipynb
```

Or run the full pipeline directly: `python src/main.py`.

## Data & API keys

Runs on built-in synthetic data with no keys; live data requires the keys below.

- `FMP_API_KEY` - Financial Modeling Prep, for statements and enterprise value (free tier is sufficient; yfinance fundamentals are patchy, so ratios are computed manually from statements).
- `FRED_API_KEY` - FRED, for the 10y Treasury risk-free rate (`DGS10`).

Copy `.env.example` to `.env` and fill in values. Pulled data is cached to `data/*.parquet` (gitignored).

## Results

All figures below are produced by actually running `src/main.py` on the built-in synthetic company - **illustrative, synthetic data**, not a live-market claim.

- `outputs/fair_value_distribution.png` - 20,000-draw fair-value histogram vs the market EV. On the synthetic demo the p5/p95 range is roughly 30,000-61,500 against a market EV of ~42,200 (P(undervalued) ~ 49%): assumption uncertainty alone spans ~2x in value.
- `outputs/wacc_growth_sensitivity.png` - WACC x terminal-growth heatmap of premium/discount to market EV, making the point-estimate fragility explicit.
- `outputs/implied_vs_historical_growth.png` - the market-implied near-term growth (~8.1% in the demo, solver repricing error < 1e-10) plotted against delivered revenue growth.

At the implied growth, the terminal value contributes ~56% of enterprise value in the demo - the quantitative reason single-point DCFs are fragile.

## Limitations

- Free fundamentals data (FMP free tier) can restate, lag, or misclassify line items; ΔNWC from balance-sheet snapshots is noisy, which is why the margin anchor uses the median.
- The Monte Carlo samples assumptions independently; in reality WACC, growth, and margins are correlated (e.g. high-growth regimes coincide with higher discount rates).
- A constant FCFF margin over the horizon is a simplification; the reverse-DCF answer is conditional on that margin and the chosen WACC.
- Synthetic demo data is idealized - it validates the machinery, not any specific company view.

## References

- Damodaran, A. - *Investment Valuation* (Wiley), on DCF construction and implied-growth analysis.
- Koller, T., Goedhart, M., Wessels, D. - *Valuation: Measuring and Managing the Value of Companies* (McKinsey/Wiley).
- Mauboussin, M., Rappaport, A. - *Expectations Investing* (the reverse-DCF/price-implied-expectations framework).
- Glasserman, P. - *Monte Carlo Methods in Financial Engineering* (Springer).
