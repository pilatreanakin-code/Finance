from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def information_coefficients(
    z_scores: pd.DataFrame, fwd_returns: pd.Series
) -> pd.DataFrame:
    rows = {}
    for dt, g in z_scores.groupby(level="date"):
        fwd = fwd_returns.loc[dt]
        ics = {}
        for f in z_scores.columns:
            ic, _ = spearmanr(g[f].droplevel("date"), fwd)
            ics[f] = ic
        rows[dt] = ics
    return pd.DataFrame(rows).T.rename_axis("date")


def ic_weights(ic_ts: pd.DataFrame, min_history: int = 3) -> pd.DataFrame:
    trailing = ic_ts.expanding(min_periods=min_history).mean().shift(1)
    w = trailing.clip(lower=0.0)
    row_sums = w.sum(axis=1)
    w = w.div(row_sums.replace(0.0, np.nan), axis=0)
    equal = pd.DataFrame(
        1.0 / ic_ts.shape[1], index=ic_ts.index, columns=ic_ts.columns
    )
    w = w.where(row_sums > 0, equal)
    w = w.fillna(equal)
    return w


def composite_score(z_scores: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    scores = []
    for dt, g in z_scores.groupby(level="date"):
        w = weights.loc[dt]
        scores.append((g * w).sum(axis=1))
    return pd.concat(scores).rename("composite")


def quintile_returns(
    composite: pd.Series, fwd_returns: pd.Series, n_quantiles: int = 5
) -> pd.DataFrame:
    df = pd.DataFrame({"score": composite, "fwd": fwd_returns}).dropna()

    def _one_date(g: pd.DataFrame) -> pd.Series:
        q = pd.qcut(g["score"].rank(method="first"), n_quantiles, labels=False) + 1
        means = g.groupby(q)["fwd"].mean()
        means.index = [f"Q{int(i)}" for i in means.index]
        return means

    out = df.groupby(level="date").apply(_one_date)
    out["spread"] = out[f"Q{n_quantiles}"] - out["Q1"]
    return out
