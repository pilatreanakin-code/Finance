"""Residual income (EVA-style) equity valuation under clean surplus.

RI_t = NI_t − r·BV_{t−1}. Under the clean-surplus relation the dividend
discount model is algebraically identical to:

    V0 = BV0 + Σ PV(RI_t)

so any DCF/RI disagreement in practice comes from *inconsistent assumptions*
(mostly the terminal value) — which makes the gap diagnostic, not noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RIForecast:
    """Explicit-horizon forecast produced by :func:`build_forecast`.

    All arrays length ``horizon``, year 1 = first forecast year.
    ``book_prev[t]`` is the opening book for year t (clean surplus).
    """

    net_income: np.ndarray
    dividends: np.ndarray
    book_prev: np.ndarray
    book_end: np.ndarray
    residual_income: np.ndarray


def historical_residual_income(statements: pd.DataFrame, r: float) -> pd.Series:
    """Historical RI series: NI_t − r × opening book equity.

    The first year is dropped (no opening book in the sample).
    """
    book_prev = statements["book_equity"].shift(1)
    ri = statements["net_income"] - r * book_prev
    return ri.dropna().rename("residual_income")


def check_clean_surplus(statements: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
    """Violation of BV_t − (BV_{t−1} + NI_t − Div_t) per year (≈0 if clean)."""
    implied = statements["book_equity"].shift(1) + statements["net_income"] - statements["dividends"]
    return (statements["book_equity"] - implied).dropna().rename("clean_surplus_gap")


def build_forecast(
    book0: float,
    roe_start: float,
    roe_fade_to: float,
    payout: float,
    horizon: int,
) -> RIForecast:
    """Project NI/dividends/book with ROE fading linearly to ``roe_fade_to``.

    The fade encodes the economic prior that excess returns (ROE > r) are
    competed away over time — the standard Ohlson-style persistence idea in
    its simplest deterministic form.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    roe_path = np.linspace(roe_start, roe_fade_to, horizon)
    ni = np.empty(horizon)
    div = np.empty(horizon)
    book_prev = np.empty(horizon)
    book_end = np.empty(horizon)
    b = book0
    for t in range(horizon):
        book_prev[t] = b
        ni[t] = roe_path[t] * b
        div[t] = payout * ni[t]
        b = b + ni[t] - div[t]
        book_end[t] = b
    return RIForecast(ni, div, book_prev, book_end, residual_income=np.empty(0))


def ri_value(
    book0: float,
    forecast: RIForecast,
    r: float,
    g_term: float = 0.0,
) -> dict[str, float]:
    """Residual income valuation: V0 = BV0 + PV(explicit RI) + PV(terminal RI).

    Parameters
    ----------
    book0 : float
        Current book equity (the valuation anchor).
    forecast : RIForecast
        Explicit-horizon projections from :func:`build_forecast`.
    r : float
        Cost of equity. Must exceed ``g_term``.
    g_term : float
        Perpetuity growth of RI beyond the horizon. ``0`` (flat) or negative
        (decaying excess returns) are the defensible choices; large positive
        terminal RI growth asserts a permanent moat.

    Returns
    -------
    dict
        ``value`` plus the additive pieces ``book``, ``pv_explicit_ri``,
        ``pv_terminal_ri`` (the reconciliation bridge components).
    """
    if r <= g_term:
        raise ValueError(f"cost of equity ({r:.4f}) must exceed terminal growth ({g_term:.4f})")
    horizon = len(forecast.net_income)
    years = np.arange(1, horizon + 1)
    ri = forecast.net_income - r * forecast.book_prev
    disc = (1.0 + r) ** years
    pv_explicit = float(np.sum(ri / disc))
    ri_next = ri[-1] * (1.0 + g_term)
    pv_terminal = float(ri_next / (r - g_term) / disc[-1])
    return {
        "value": book0 + pv_explicit + pv_terminal,
        "book": book0,
        "pv_explicit_ri": pv_explicit,
        "pv_terminal_ri": pv_terminal,
    }


def ddm_value(forecast: RIForecast, r: float, g_term: float = 0.0) -> dict[str, float]:
    """Dividend-discount value on the *same* forecast (the DCF comparator).

    Terminal value uses a Gordon perpetuity on the year-after-horizon
    dividend. With ``g_term`` consistent between RI and DDM the two values
    coincide only if the terminal assumptions are economically equivalent —
    which they generally are not, and that gap is the diagnostic.
    """
    if r <= g_term:
        raise ValueError(f"cost of equity ({r:.4f}) must exceed terminal growth ({g_term:.4f})")
    horizon = len(forecast.dividends)
    years = np.arange(1, horizon + 1)
    disc = (1.0 + r) ** years
    pv_explicit = float(np.sum(forecast.dividends / disc))
    div_next = forecast.dividends[-1] * (1.0 + g_term)
    pv_terminal = float(div_next / (r - g_term) / disc[-1])
    return {
        "value": pv_explicit + pv_terminal,
        "pv_explicit_div": pv_explicit,
        "pv_terminal_div": pv_terminal,
    }


def ddm_value_no_terminal(forecast: RIForecast, r: float) -> float:
    """PV of explicit dividends plus discounted end book (long-horizon check).

    As the horizon grows this converges to the RI value — the identity used
    by the tests to verify both implementations.
    """
    horizon = len(forecast.dividends)
    years = np.arange(1, horizon + 1)
    disc = (1.0 + r) ** years
    return float(np.sum(forecast.dividends / disc) + forecast.book_end[-1] / disc[-1])
