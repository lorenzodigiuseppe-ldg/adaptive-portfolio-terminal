"""
Compute layer for the interactive terminal (no UI imports, fully testable).

Everything the app shows is produced here from the research engine:
market snapshot, expert back-tests, the online learner, benchmarks, risk tables,
statistical tests and the *current recommended allocation*.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import metrics as M
from . import stats_tests as ST
from .backtest import Backtester, best_constant_mix, grid_search
from .data import TICKERS, load_prices, load_risk_free
from .experts import MeanVariance, MinCVaR, RiskParity, risk_contributions
from .estimators import ledoit_wolf_cc
from .features import features_at
from .meta import (ONS, AdaptiveBOA, ContextualGate, FollowTheLeader, Hedge, Hierarchical,
                   StaticMix)

LEARNERS = ["AdaptiveBOA", "BOA+Context", "Hedge/FixedShare", "ONS (log-wealth)",
            "ContextGate", "FollowTheLeader", "1/3 static mix"]
EXPERT_NAMES = ["MVO", "RiskParity", "MinCVaR"]
BENCHMARKS = ["1/3 static mix", "MVO", "RiskParity", "MinCVaR", "1/N", "60/40 SPY/IEF",
              "Oracle const. mix*"]


# ----------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class EngineSettings:
    rebal_every: int = 5
    expert_refit: int = 4
    lag: int = 1
    tc_bps: float = 5.0
    warmup: int = 504
    start: str = "2006-01-01"


@dataclass(frozen=True)
class ExpertSettings:
    mvo_delta: float = 4.0
    mvo_lb_mu: int = 504
    mvo_lb_cov: int = 252
    mvo_cap: float = 0.40
    rp_lb: int = 126
    cvar_beta: float = 0.95
    cvar_lb: int = 504
    cvar_cap: float = 0.40


@dataclass(frozen=True)
class LearnerSettings:
    name: str = "AdaptiveBOA"
    gamma: float = 5.0
    rho: float = 1.0
    alpha: float = 0.01
    eta: float = 0.25
    ons_beta: float = 1.0
    gate_lr: float = 0.05
    gate_l2: float = 0.01
    ftl_window: int = 52


# ----------------------------------------------------------------------------- data
def load_market(csv_path: str, start="2006-01-01"):
    prices = load_prices(TICKERS, start=start, source="csv", csv_path=csv_path, backfill_proxies=True)
    returns = prices.pct_change().dropna()
    rf = load_risk_free(returns.index, csv_path=csv_path)
    return prices, returns, rf


def market_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    """Last price and trailing performance per asset (monitor page)."""
    last = prices.iloc[-1]
    r = prices.pct_change()

    def chg(days):
        return prices.iloc[-1] / prices.iloc[-1 - days] - 1

    ytd_base = prices.loc[: f"{prices.index[-1].year - 1}-12-31"].iloc[-1]
    out = pd.DataFrame({
        "Last": last,
        "1D": chg(1), "1W": chg(5), "1M": chg(21), "3M": chg(63),
        "YTD": last / ytd_base - 1, "1Y": chg(252),
        "Vol 1M": r.iloc[-21:].std() * np.sqrt(252),
        "Vol 1Y": r.iloc[-252:].std() * np.sqrt(252),
        "DD from 1Y high": last / prices.iloc[-252:].max() - 1,
    })
    return out


def asset_stats(returns: pd.DataFrame, rf: pd.Series, start=None, end=None) -> pd.DataFrame:
    rows = {c: M.summarize(returns[c].loc[start:end], rf) for c in returns}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- engine
def make_backtester(returns, rf, es: EngineSettings) -> Backtester:
    return Backtester(returns, rebal_every=es.rebal_every, lag=es.lag, tc_bps=es.tc_bps,
                      warmup=es.warmup, rf=rf, expert_refit_every=es.expert_refit)


def make_experts(xs: ExpertSettings):
    return [
        MeanVariance(risk_aversion=xs.mvo_delta, lookback_mu=xs.mvo_lb_mu,
                     lookback_cov=xs.mvo_lb_cov, w_max=xs.mvo_cap),
        RiskParity(lookback=xs.rp_lb),
        MinCVaR(beta=xs.cvar_beta, lookback=xs.cvar_lb, w_max=xs.cvar_cap),
    ]


def make_learner(ls: LearnerSettings):
    n = ls.name
    if n == "AdaptiveBOA":
        return AdaptiveBOA(gamma=ls.gamma, rho=ls.rho, alpha=ls.alpha, name=n)
    if n == "BOA+Context":
        return Hierarchical(AdaptiveBOA(gamma=ls.gamma, rho=ls.rho, alpha=ls.alpha),
                            ContextualGate(gamma=ls.gamma, lr=ls.gate_lr, l2=ls.gate_l2), name=n)
    if n == "Hedge/FixedShare":
        return Hedge(eta=ls.eta, alpha=ls.alpha, gamma=ls.gamma, name=n)
    if n == "ONS (log-wealth)":
        return ONS(beta=ls.ons_beta, rho=ls.rho, name=n)
    if n == "ContextGate":
        return ContextualGate(gamma=ls.gamma, lr=ls.gate_lr, l2=ls.gate_l2, name=n)
    if n == "FollowTheLeader":
        return FollowTheLeader(window=ls.ftl_window, gamma=ls.gamma, name=n)
    return StaticMix(name="1/3 static mix")


def expert_weights(bt: Backtester, xs: ExpertSettings) -> dict:
    return bt.expert_weights(make_experts(xs))


def run_strategy(bt: Backtester, Wd: dict, ls: LearnerSettings, features=None):
    """Run the selected learner. Returns (result dict, fitted learner)."""
    learner = make_learner(ls)
    needs_x = ls.name in ("ContextGate", "BOA+Context")
    res = bt.run_meta(learner, Wd, features=features if needs_x else None)
    return res, learner


def run_benchmarks(bt: Backtester, Wd: dict, gamma=5.0) -> dict:
    out = {"1/3 static mix": bt.run_meta(StaticMix(name="1/3 static mix"), Wd)}
    for n, W in Wd.items():
        out[n] = bt.run_static(n, W)
    P = len(bt.dates)
    out["1/N"] = bt.run_static("1/N", np.tile(np.full(bt.N, 1 / bt.N), (P, 1)))
    w = np.array([0.6 if a == "SPY" else 0.4 if a == "IEF" else 0.0 for a in bt.assets])
    out["60/40 SPY/IEF"] = bt.run_static("60/40", np.tile(w, (P, 1)))
    Rk = bt.expert_period_returns(Wd)
    pi_star = best_constant_mix(Rk, gamma=gamma)
    out["Oracle const. mix*"] = bt.run_static("oracle", bt.combine(np.tile(pi_star, (P, 1)), Wd))
    out["Oracle const. mix*"]["pi_star"] = pi_star
    return out


def recommend_now(bt: Backtester, xs: ExpertSettings, learner, features=None):
    """Allocation to hold from the next rebalance, using ALL data available today."""
    R = bt.R
    experts = make_experts(xs)
    W_now = np.vstack([e.weights(R) for e in experts])            # K x N
    x = None if features is None else features.values[-1]
    pi_now = learner.weights(x)                                    # learner already updated
    w_now = pi_now @ W_now
    S = ledoit_wolf_cc(R[-252:]) * 252
    rc = risk_contributions(w_now, S)
    vol = float(np.sqrt(w_now @ S @ w_now))
    return {
        "pi": pd.Series(pi_now, index=EXPERT_NAMES),
        "expert_weights": pd.DataFrame(W_now, index=EXPERT_NAMES, columns=bt.assets),
        "weights": pd.Series(w_now, index=bt.assets),
        "risk_contrib": pd.Series(rc, index=bt.assets),
        "ex_ante_vol": vol,
        "as_of": bt.returns.index[-1],
    }


def kpis(res: dict, rf, ppy, start=None, end=None) -> dict:
    to = res.get("turnover")
    return M.summarize(res["daily"].loc[start:end], rf,
                       None if to is None else to.loc[start:end], ppy)


def comparison_table(results: dict, rf, ppy, start=None, end=None) -> pd.DataFrame:
    return M.table(results, rf, ppy, start=start, end=end)


def period_returns_series(bt: Backtester, res: dict) -> pd.Series:
    return pd.Series(bt.period_returns(res["weights"].values) - bt.rf_period, index=bt.dates)


def utility_gap(bt, res, ref, gamma):
    a, b = period_returns_series(bt, res), period_returns_series(bt, ref)
    ua, ub = a - 0.5 * gamma * a ** 2, b - 0.5 * gamma * b ** 2
    return (ua - ub).cumsum()


def learning_rates(res: dict) -> pd.DataFrame | None:
    d = res.get("diagnostics") or []
    if not d or "eta" not in d[0]:
        return None
    return pd.DataFrame([x["eta"] for x in d], index=res["pi"].index, columns=res["pi"].columns)


def conditional_pi(res: dict, features: pd.DataFrame) -> dict:
    pi = res["pi"]
    F = features.reindex(pi.index)
    out = {}
    if "log_vol21" in F:
        q = pd.qcut(F["log_vol21"].rank(method="first"), 3, labels=["Low vol", "Mid vol", "High vol"])
        out["Volatility state"] = pi.groupby(q, observed=False).mean()
    if "stock_bond_corr63" in F:
        s = np.where(F["stock_bond_corr63"] > 0, "Stock-bond corr > 0", "Stock-bond corr < 0")
        out["Stock-bond correlation"] = pi.groupby(s).mean()
    return out


def calendar_returns(daily: pd.Series) -> pd.Series:
    return (1 + daily).groupby(daily.index.year).prod() - 1


def monthly_table(daily: pd.Series) -> pd.DataFrame:
    m = (1 + daily).resample("ME").prod() - 1
    t = pd.DataFrame({"y": m.index.year, "m": m.index.month, "r": m.values})
    tab = t.pivot(index="y", columns="m", values="r")
    tab.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: tab.shape[1]]
    tab["Year"] = calendar_returns(daily).reindex(tab.index).values
    return tab


def worst_drawdowns(daily: pd.Series, n=5) -> pd.DataFrame:
    w = (1 + daily).cumprod()
    peak = w.cummax()
    dd = w / peak - 1
    rows, in_dd, start = [], False, None
    for t, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, start = True, t
        elif v == 0 and in_dd:
            seg = dd.loc[start:t]
            rows.append((start, seg.idxmin(), t, seg.min()))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        rows.append((start, seg.idxmin(), pd.NaT, seg.min()))
    df = pd.DataFrame(rows, columns=["Start", "Trough", "Recovered", "Depth"])
    df = df.sort_values("Depth").head(n).reset_index(drop=True)
    df["Days to trough"] = [(b - a).days for a, b in zip(df.Start, df.Trough)]
    return df


def pairwise_tests(results: dict, bt, rf, learner_name, benches, start, gamma):
    rows = []
    a_d = results[learner_name]["daily"]
    ex_a = (a_d - rf.reindex(a_d.index).fillna(0)).loc[start:]
    wk_a = period_returns_series(bt, results[learner_name]).loc[start:]
    for b in benches:
        if b == learner_name or b not in results:
            continue
        b_d = results[b]["daily"]
        ex_b = (b_d - rf.reindex(b_d.index).fillna(0)).loc[start:]
        wk_b = period_returns_series(bt, results[b]).loc[start:]
        h = ST.sharpe_diff_hac(ex_a, ex_b)
        bs = ST.sharpe_diff_bootstrap(ex_a, ex_b, n_boot=500)
        dm = ST.utility_dm_test(wk_a, wk_b, gamma=gamma)
        rows.append({"vs": b, "ΔSharpe": h["dSR_ann"], "HAC p": h["p_value"],
                     "CI low": bs["ci95"][0], "CI high": bs["ci95"][1],
                     "ΔUtility bp/wk": 1e4 * dm["mean_diff"], "DM p": dm["p_value"]})
    return pd.DataFrame(rows).set_index("vs") if rows else pd.DataFrame()


def learnability(bt, Wd, end=None) -> pd.DataFrame:
    Rk = bt.expert_period_returns(Wd).loc[:end]
    C = Rk.corr()
    names = list(Wd)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"Pair": f"{a} / {b}", "Corr": C.loc[a, b],
                         "Yrs to detect ΔSR 0.2": ST.min_track_record_years(0.2, C.loc[a, b]),
                         "Yrs to detect ΔSR 0.5": ST.min_track_record_years(0.5, C.loc[a, b])})
    return pd.DataFrame(rows).set_index("Pair")


def tune_boa(bt, Wd, is_end, rf):
    grid = [dict(gamma=g, rho=r, alpha=a) for g in [2, 5, 10] for r in [0.98, 0.99, 0.995, 1.0]
            for a in [0.0, 0.01, 0.05]]
    return grid_search(bt, Wd, lambda gamma, rho, alpha: AdaptiveBOA(gamma=gamma, rho=rho, alpha=alpha),
                       grid, is_end, rf=rf)


def settings_dict(*objs):
    d = {}
    for o in objs:
        d.update(asdict(o))
    return d
