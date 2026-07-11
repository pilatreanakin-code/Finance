from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from comps import DRIVERS


@dataclass
class FairMultipleResult:

    multiple: str
    model: object
    fitted: pd.Series
    residuals: pd.Series
    residual_z: pd.Series
    r_squared: float
    coefficients: pd.DataFrame

    def ranking(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "actual": self.fitted + self.residuals,
                "fair": self.fitted,
                "residual": self.residuals,
                "residual_z": self.residual_z,
            }
        )
        return df.sort_values("residual_z")


def fit_fair_multiple(
    panel: pd.DataFrame,
    multiple: str = "ev_ebitda",
    robust_se: bool = True,
) -> FairMultipleResult:
    if multiple not in panel.columns:
        raise ValueError(f"unknown multiple: {multiple}")
    y = panel[multiple]
    X = sm.add_constant(panel[DRIVERS])
    res = sm.OLS(y, X).fit(cov_type="HC3" if robust_se else "nonrobust")

    fitted = pd.Series(res.fittedvalues, index=panel.index, name="fair")
    resid = pd.Series(res.resid, index=panel.index, name="residual")
    z = ((resid - resid.mean()) / resid.std(ddof=1)).rename("residual_z")

    coef = pd.DataFrame(
        {"coef": res.params, "t_stat": res.tvalues, "p_value": res.pvalues}
    )
    return FairMultipleResult(
        multiple=multiple,
        model=res,
        fitted=fitted,
        residuals=resid,
        residual_z=z,
        r_squared=float(res.rsquared),
        coefficients=coef,
    )


def rich_cheap_table(
    results: dict[str, FairMultipleResult], top_n: int = 5
) -> pd.DataFrame:
    z = pd.DataFrame({m: r.residual_z for m, r in results.items()})
    z["composite_z"] = z.mean(axis=1)
    z = z.sort_values("composite_z")
    z["flag"] = ""
    z.iloc[:top_n, z.columns.get_loc("flag")] = "CHEAP"
    z.iloc[-top_n:, z.columns.get_loc("flag")] = "RICH"
    return z
