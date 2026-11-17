"""
Observable market-state features for the contextual gate (no predefined regimes).
Every feature at rebalance t uses only returns available at t, and is standardised with an
expanding (past-only) mean / std, then clipped to [-3, 3].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def raw_features(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns
    ew = r.mean(axis=1)
    f = pd.DataFrame(index=r.index)
    vol21 = ew.rolling(21).std() * np.sqrt(252)
    vol252 = ew.rolling(252).std() * np.sqrt(252)
    f["log_vol21"] = np.log(vol21)
    f["vol_ratio"] = np.log(vol21 / vol252)
    if "SPY" in r:
        f["spy_mom126"] = (1 + r["SPY"]).rolling(126).apply(np.prod, raw=True) - 1
    if {"SPY", "IEF"} <= set(r):
        f["stock_bond_corr63"] = r["SPY"].rolling(63).corr(r["IEF"])
    # average pairwise correlation (63d)
    n = r.shape[1]
    var = r.rolling(63).var()
    port_var = ew.rolling(63).var()
    avg_var = var.mean(axis=1)
    f["avg_corr63"] = ((port_var * n * n / n - avg_var) / (avg_var * (n - 1))).clip(-1, 1)
    if {"DBC", "IEF"} <= set(r):
        cum = (1 + r[["DBC", "IEF"]]).rolling(126).apply(np.prod, raw=True)
        f["infl_tilt126"] = cum["DBC"] - cum["IEF"]
    wealth = (1 + ew).cumprod()
    f["ew_drawdown"] = wealth / wealth.rolling(252, min_periods=1).max() - 1
    return f


def features_at(returns: pd.DataFrame, rebal_idx, lag=1, burn_in=26) -> pd.DataFrame:
    """Feature matrix (n_periods x d), past-only standardised."""
    F = raw_features(returns)
    rows = [max(s - lag - 1, 0) for s in rebal_idx]  # last row available at decision time
    X = F.iloc[rows].reset_index(drop=True)
    Z = np.zeros(X.shape)
    vals = X.values
    for p in range(len(X)):
        hist = vals[: p + 1]
        hist = hist[~np.isnan(hist).any(axis=1)]
        if len(hist) < burn_in or np.isnan(vals[p]).any():
            continue
        m, s = hist.mean(0), hist.std(0) + 1e-12
        Z[p] = np.clip((vals[p] - m) / s, -3, 3)
    return pd.DataFrame(Z, columns=X.columns, index=returns.index[list(rebal_idx)])
