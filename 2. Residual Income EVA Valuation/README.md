<!-- suggested repo name: residual-income-eva-valuation -->
# Residual Income EVA Valuation

Value equity as book value plus the present value of excess returns - and reconcile the answer against a DCF instead of picking one.

## The skeptical insight

RI anchors on book value and is far less terminal-value-sensitive than DCF - so a DCF/RI disagreement is diagnostic, not noise. Serious analysts triangulate; this shows both and explains the gap.

## What it does

The pipeline computes historical residual income (RI = NI - r*opening book equity) under the clean-surplus relation, verifies that relation holds in the data, then builds a single explicit forecast (ROE fading toward the cost of equity as competition erodes excess returns) and values it two ways: residual income (V = BV₀ + PV of RI) and a simple dividend-discount DCF. Because clean surplus makes the two algebraically identical over any explicit horizon, the entire disagreement is a terminal-assumption artifact - and the code decomposes the gap *exactly* into PV(end book) + PV(terminal RI) - PV(terminal dividends), with the identity residual printed at machine precision. A terminal-growth sensitivity sweep then shows how much less the RI value moves than the DCF value.

## Methodology

- **Clean surplus check**: BVₜ = BVₜ₋₁ + NIₜ - Divₜ verified on the input data (the identity RI valuation depends on).
- **Historical RI**: NIₜ - r*BVₜ₋₁, with a CAPM-style cost of equity (FRED risk-free + beta x 5% ERP on live data; fixed 9% in the offline demo).
- **Forecast**: ROE fades linearly to a terminal level over the horizon - the Ohlson-style persistence idea in its simplest deterministic form.
- **Two valuations, one forecast**: RI value with a flat/decaying terminal RI perpetuity vs a Gordon-terminal DDM. Both reject `r <= g` inputs.
- **Exact gap decomposition** from the clean-surplus identity: BV₀ + Σ PV(RIₜ) = Σ PV(Divₜ) + PV(BV_T); the tests assert both this identity and the decomposition at ~1e-12 relative error.

## How to run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pilatreanakin-code/Finance/blob/main/2.%20Residual%20Income%20EVA%20Valuation/notebooks/demo.ipynb)

```
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on Unix)
pip install numpy>=1.26 pandas>=2.0 matplotlib>=3.8 pytest>=8.0 requests>=2.31
pytest                        # offline, synthetic data
jupyter notebook notebooks/demo.ipynb
```

Or run the full pipeline directly: `python src/main.py`.

## Data & API keys

Runs on built-in synthetic data with no keys; live data requires the keys below.

- `FMP_API_KEY` - Financial Modeling Prep, for statements, market cap, and beta.
- `FRED_API_KEY` - FRED, for the risk-free rate (`DGS10`).

Copy `.env.example` to `.env`. Pulled data is cached to `data/*.parquet` (gitignored).

## Results

All figures produced by actually running `src/main.py` on the built-in synthetic company - **illustrative, synthetic data**.

- `outputs/reconciliation_bridge.png` - the RI waterfall (book -> +PV explicit RI -> +PV terminal RI) next to the DCF value, plus the exact gap decomposition. In the demo the RI value's terminal share is ~7% versus ~57% for the DCF - same forecast, radically different fragility.
- `outputs/terminal_sensitivity.png` - value vs terminal growth for both methods; the RI line is nearly flat while the DCF line climbs steeply.
- `outputs/residual_income_history.png` - historical RI, i.e. whether the firm actually earns above its cost of equity.

The gap decomposition identity residual in the demo run is ~8e-12 - the reconciliation is exact, not approximate.

## Limitations

- Clean surplus fails in real filings (OCI, buybacks at premiums to book, currency translation); live data will show violations the synthetic data doesn't, and the check quantifies them rather than hiding them.
- Book value is an accounting construct - intangibles-heavy firms understate the RI anchor and shift value into the RI stream.
- A deterministic linear ROE fade is a modeling choice; Ohlson persistence is richer (AR(1) in RI).
- The DDM comparator is deliberately simple ("a simple DCF"); an FCFF DCF would differ in leverage treatment, not in the terminal-sensitivity conclusion.

## References

- Ohlson, J. (1995) - "Earnings, Book Values, and Dividends in Equity Valuation", *Contemporary Accounting Research*.
- Penman, S. - *Financial Statement Analysis and Security Valuation* (McGraw-Hill).
- Stewart, G.B. - *The Quest for Value* (the EVA framework, Stern Stewart & Co).
- Damodaran, A. - *Investment Valuation* (Wiley).
