<!-- suggested repo name: regression-implied-relative-valuation -->
# Regression-Implied Relative Valuation

Replace "the peer median multiple" with a cross-sectional regression that says what multiple each firm *should* trade at - and rank the residuals.

## The skeptical insight

Naïve comps assume peers are identical. A cross-sectional regression controls for *why* multiples differ; the residual is the actual mispricing signal.

## What it does

Given a sector cross-section of firms with fundamentals (revenue growth, EBITDA margin, ROIC, leverage) and trading multiples (EV/EBITDA, P/E), the pipeline winsorizes everything at the 1st/99th percentiles, regresses each multiple on the drivers with heteroskedasticity-robust (HC3) standard errors, and produces a fundamentals-implied "fair" multiple per firm. The residual - actual minus fair - is z-scored and combined across multiples into a composite rich/cheap ranking: a firm trading below the multiple its own growth, profitability, and leverage justify is flagged CHEAP, and vice versa. The synthetic fallback panel is generated from *known* coefficients (plus planted data-error outliers), so the tests verify the regression genuinely recovers the drivers rather than just running.

## Methodology

- **Winsorization 1/99** on both multiples and drivers: outliers (distressed denominators, data errors) are capped, not deleted, so they stay in the cross-section without dominating the fit.
- **Fair-multiple OLS**: `multiple ~ growth + ebitda_margin + roic + leverage` with HC3 robust standard errors; coefficient signs are economically interpretable (growth/margin/ROIC positive, leverage negative).
- **Residual z-scores** per multiple, averaged into a composite so a firm must look cheap on more than one denominator to top the ranking.
- **Known-truth validation**: the synthetic generator plants true coefficients and outliers; tests assert sign recovery and magnitude within tolerance.

## How to run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pilatreanakin-code/Finance/blob/main/3.%20Regression-Implied%20Relative%20Valuation/notebooks/demo.ipynb)

```
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on Unix)
pip install numpy>=1.26 pandas>=2.0 matplotlib>=3.8 statsmodels>=0.14 pytest>=8.0 requests>=2.31
pytest                        # offline, synthetic data
jupyter notebook notebooks/demo.ipynb
```

Or run the full pipeline directly: `python src/main.py`.

## Data & API keys

Runs on built-in synthetic data with no keys; live data requires the key below.

- `FMP_API_KEY` - Financial Modeling Prep, for TTM key metrics, ratios, and growth per ticker (free tier works; pass a ticker list in the `USER INPUTS` block).

Copy `.env.example` to `.env`. Pulled data is cached to `data/*.parquet` (gitignored).

## Results

All figures produced by running `src/main.py` on the synthetic sector - **illustrative, synthetic data**.

- `outputs/actual_vs_fair_ev_ebitda.png` and `outputs/actual_vs_fair_pe.png` - actual vs regression-implied fair multiple with the 45° line; points above are rich, below are cheap (demo R² ~ 0.36 and 0.20 respectively - realistic for cross-sectional comps).
- `outputs/residual_ranking.png` - the composite rich/cheap ranking (10 cheapest and 10 richest).
- `outputs/winsorization_effect.png` - the raw vs winsorized EV/EBITDA distribution, showing exactly what the 1/99 clip caps.

## Limitations

- A regression only reallocates value *within* the peer set - if the whole sector is mispriced, every residual is contaminated (relative, not absolute, valuation).
- Modest cross-sections (dozens of firms) mean noisy coefficients; robust SEs help inference but not estimation variance.
- TTM fundamentals from free endpoints can lag or restate; denominator quirks (near-zero EBITDA or EPS) survive winsorization only partially.
- Linear drivers are a first-order approximation; multiples are convex in growth for near-zero-rate regimes.

## References

- Damodaran, A. - *Investment Valuation* (Wiley), ch. on relative valuation and sector regressions.
- Koller, T., Goedhart, M., Wessels, D. - *Valuation* (McKinsey/Wiley), on what drives multiples.
- Liu, J., Nissim, D., Thomas, J. (2002) - "Equity Valuation Using Multiples", *Journal of Accounting Research*.
