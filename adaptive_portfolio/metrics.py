"""Performance and risk metrics (daily return input)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def drawdown(r: pd.Series) -> pd.Series:
    w = (1 + r).cumprod()
    return w / w.cummax() - 1


def cvar(r, beta=0.95):
    r = np.asarray(r)
    q = np.quantile(r, 1 - beta)
    return -r[r <= q].mean()


def summarize(r: pd.Series, rf: pd.Series | None = None, turnover: pd.Series | None = None,
              periods_per_year: float | None = None) -> dict:
    r = r.dropna()
    ex = r - (0 if rf is None else rf.reindex(r.index).fillna(0))
    yrs = len(r) / ANN
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ANN)
    sharpe = ex.mean() / ex.std() * np.sqrt(ANN)
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(ANN)
    dd = drawdown(r)
    mdd = dd.min()
    out = {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": ex.mean() * ANN / downside if downside > 0 else np.nan,
        "MaxDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "CVaR95(d)": cvar(r, 0.95),
        "Skew": stats.skew(r),
        "ExKurt": stats.kurtosis(r),
        "Worst month": (1 + r).resample("ME").prod().min() - 1,
    }
    if turnover is not None and periods_per_year:
        # first rebalance builds the portfolio from cash; exclude it
        out["Turnover/yr"] = turnover.iloc[1:].mean() * periods_per_year
    return out


def table(results: dict, rf=None, periods_per_year=52, start=None, end=None) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        r = res["daily"].loc[start:end]
        to = res.get("turnover")
        to = None if to is None else to.loc[start:end]
        rows[name] = summarize(r, rf, to, periods_per_year)
    return pd.DataFrame(rows).T


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    pct = ["CAGR", "Vol", "MaxDD", "CVaR95(d)", "Worst month", "Turnover/yr"]
    out = df.copy().astype(object)
    for c in df.columns:
        if c in pct:
            out[c] = df[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
        else:
            out[c] = df[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    return out
