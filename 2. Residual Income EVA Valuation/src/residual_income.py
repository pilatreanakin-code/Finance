from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RIForecast:

    net_income: np.ndarray
    dividends: np.ndarray
    book_prev: np.ndarray
    book_end: np.ndarray
    residual_income: np.ndarray


def historical_residual_income(statements: pd.DataFrame, r: float) -> pd.Series:
    book_prev = statements["book_equity"].shift(1)
    ri = statements["net_income"] - r * book_prev
    return ri.dropna().rename("residual_income")


def check_clean_surplus(statements: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
    implied = statements["book_equity"].shift(1) + statements["net_income"] - statements["dividends"]
    return (statements["book_equity"] - implied).dropna().rename("clean_surplus_gap")


def build_forecast(
    book0: float,
    roe_start: float,
    roe_fade_to: float,
    payout: float,
    horizon: int,
) -> RIForecast:
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
    horizon = len(forecast.dividends)
    years = np.arange(1, horizon + 1)
    disc = (1.0 + r) ** years
    return float(np.sum(forecast.dividends / disc) + forecast.book_end[-1] / disc[-1])
