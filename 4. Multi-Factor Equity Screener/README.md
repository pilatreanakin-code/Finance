<!-- suggested repo name: multi-factor-equity-screener -->
# Multi-Factor Equity Screener

Value, quality, and momentum combined the way practitioners actually do it: winsorized, sector-neutral, IC-weighted - with no look-ahead in the weights.

## The skeptical insight

Factor *hygiene* - winsorization, sector-neutral z-scoring, IC weighting - separates people who use factors from people who understand them.

## What it does

The screener takes a multi-period equity panel (raw fundamentals + forward returns), computes three canonical raw factors (value = book-to-price, quality = ROE minus accruals, momentum = 12-1 return), then applies the hygiene layer: per-date 1/99 winsorization, z-scoring *within* (date, sector) groups so no factor smuggles in a sector bet, and per-date Spearman information coefficients whose expanding mean - lagged one period - sets the composite weights. Negative-IC factors get weight zero rather than a sign flip (flipping on noisy ICs is overfitting). The composite is evaluated the standard way: quintile portfolios and the Q5-Q1 forward-return spread. The synthetic universe plants known ICs (value 0.08 > quality 0.05 > momentum 0.03), so tests assert the pipeline *recovers* the planted ordering and produces a positive spread - not merely that it runs.

## Methodology

- **Winsorization per date** (1st/99th pct) so outliers are capped within each cross-section.
- **Sector-neutral z-scores**: demean/scale within (date, sector); singleton groups get 0, not NaN.
- **IC weighting without look-ahead**: weights at month t use ICs through t-1 only (expanding mean, floored at 0, renormalized; equal weights until 3 months of history exist). A test recomputes the last date's weights by hand to prove no leakage.
- **Quintile evaluation**: rank-based `qcut` per date; mean return per quintile; Q5-Q1 spread with its t-stat.

## How to run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pilatreanakin-code/Finance/blob/main/4.%20Multi-Factor%20Equity%20Screener/notebooks/demo.ipynb)

```
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on Unix)
pip install numpy>=1.26 pandas>=2.0 scipy>=1.11 matplotlib>=3.8 pytest>=8.0
pytest                        # offline, synthetic data
jupyter notebook notebooks/demo.ipynb
```

Or run the full pipeline directly: `python src/main.py`.

## Data & API keys

Runs on built-in synthetic data with no keys; live data requires the keys below.

- `FMP_API_KEY` - Financial Modeling Prep for fundamentals (free tier; yfinance fundamentals are patchy, so compute ratios from statements).
- `SEC_USER_AGENT` - SEC EDGAR requires a `Name <email>` User-Agent header for filings-based factors.

A real deployment needs point-in-time fundamentals (as-reported, lagged to availability date) - see Limitations.

## Results

All figures produced by running `src/main.py` on the synthetic universe - **illustrative, synthetic data**.

- `outputs/quintile_spread.png` - mean forward return by composite quintile; the demo shows a monotone Q1->Q5 pattern with a mean Q5-Q1 spread of ~1.27%/period (t ~ 6.5) - consistent with the planted signal, as it should be.
- `outputs/factor_correlation.png` - average cross-sectional correlations (near zero by construction here; on real data value/momentum are typically negatively correlated).
- `outputs/ic_and_weights.png` - per-date ICs and the resulting lagged expanding-mean weights.

## Limitations

- The synthetic universe validates machinery, not markets: real ICs are noisier, regime-dependent, and factor crowding changes them.
- Free fundamentals are **not point-in-time** - using current statements for past dates injects restatement and reporting-lag look-ahead; a production screener needs as-reported data with availability lags.
- No transaction costs or turnover control in the quintile evaluation (this is a screener, not a backtest - see project #12 for the costed version).
- Sector neutralization uses coarse labels; industry-level neutralization and size control matter on real universes.

## References

- Sloan, R. (1996) - "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows about Future Earnings?", *The Accounting Review*.
- Asness, C., Moskowitz, T., Pedersen, L.H. (2013) - "Value and Momentum Everywhere", *Journal of Finance*.
- Grinold, R., Kahn, R. - *Active Portfolio Management* (McGraw-Hill), on ICs and the fundamental law.
- Jegadeesh, N., Titman, S. (1993) - "Returns to Buying Winners and Selling Losers", *Journal of Finance*.
