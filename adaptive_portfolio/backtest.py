"""
Walk-forward back-test engine.

Timing convention (no look-ahead):
    returns row i  = close(i-1) -> close(i)
    a rebalance at row s holds new weights over rows s .. s_next-1
    the weights are computed from rows < s - lag   (lag = 1 => one full day of slack
    between the last observation and the trade, i.e. trade at the next close)
Within a holding period every portfolio drifts with prices (buy-and-hold); turnover is
measured against the drifted weights and charged at `tc_bps` per unit of one-way turnover.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .meta import MetaLearner


@dataclass
class Backtester:
    returns: pd.DataFrame
    rebal_every: int = 5
    lag: int = 1
    tc_bps: float = 5.0
    warmup: int = 504
    rf: pd.Series | None = None
    expert_refit_every: int = 4     # re-optimise experts every k rebalances (4 x weekly ~ monthly)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.R = self.returns.values.astype(float)
        self.T, self.N = self.R.shape
        self.assets = list(self.returns.columns)
        start = self.warmup + self.lag
        self.rebal_idx = np.arange(start, self.T, self.rebal_every)
        self.ends = np.append(self.rebal_idx[1:], self.T)
        self.dates = self.returns.index[self.rebal_idx]
        # asset cumulative growth inside each period
        self.G = [np.cumprod(1 + self.R[s:e], axis=0) for s, e in zip(self.rebal_idx, self.ends)]
        rf = np.zeros(self.T) if self.rf is None else self.rf.reindex(self.returns.index).fillna(0).values
        self.rf_daily = rf
        self.rf_period = np.array([np.prod(1 + rf[s:e]) - 1 for s, e in zip(self.rebal_idx, self.ends)])
        self.bt_index = self.returns.index[self.rebal_idx[0]:]

    # ---------------------------------------------------------------- experts
    def expert_weights(self, experts, verbose=False) -> dict:
        out = {}
        for e in experts:
            key = (e.name, repr(e))
            if key not in self._cache:
                W = np.empty((len(self.rebal_idx), self.N))
                for p, s in enumerate(self.rebal_idx):
                    if p % self.expert_refit_every == 0:
                        w = e.weights(self.R[: s - self.lag])
                    W[p] = w
                self._cache[key] = W
                if verbose:
                    print(f"  computed {e.name}: {len(W)} rebalances")
            out[e.name] = self._cache[key]
        return out

    def period_returns(self, W: np.ndarray) -> np.ndarray:
        """Gross period return of a weight path W (P x N)."""
        return np.array([G[-1] @ w - 1 for G, w in zip(self.G, W)])

    def expert_period_returns(self, Wd: dict, excess=True) -> pd.DataFrame:
        M = np.column_stack([self.period_returns(W) for W in Wd.values()])
        if excess:
            M = M - self.rf_period[:, None]
        return pd.DataFrame(M, index=self.dates, columns=list(Wd))

    # ---------------------------------------------------------------- simulation
    def simulate(self, W: np.ndarray, tc_bps: float | None = None):
        """Daily net returns and per-rebalance turnover of a target-weight path."""
        c = (self.tc_bps if tc_bps is None else tc_bps) / 1e4
        daily, turnover = [], []
        prev = np.zeros(self.N)  # start from cash: first trade counted
        for G, w in zip(self.G, W):
            to = np.abs(w - prev).sum()
            V = G @ w
            r = np.empty(len(V))
            r[0] = V[0] - 1
            r[1:] = V[1:] / V[:-1] - 1
            r[0] = (1 + r[0]) * (1 - c * to) - 1
            daily.append(r)
            turnover.append(to)
            prev = w * G[-1] / V[-1]
        return pd.Series(np.concatenate(daily), index=self.bt_index), np.array(turnover)

    def combine(self, pi: np.ndarray, Wd: dict) -> np.ndarray:
        stack = np.stack(list(Wd.values()), axis=1)  # P x K x N
        return np.einsum("pk,pkn->pn", pi, stack)

    def run_meta(self, learner: MetaLearner, Wd: dict, features: pd.DataFrame | None = None,
                 periods: slice | None = None):
        """Run an online learner sequentially. Returns dict with pi, weights, daily returns."""
        Rk = self.expert_period_returns(Wd).values
        X = None if features is None else features.values
        K = Rk.shape[1]
        learner.reset(K, 0 if X is None else X.shape[1])
        P = len(Rk)
        pi = np.empty((P, K))
        diag = []
        for p in range(P):
            x = None if X is None else X[p]
            pi[p] = learner.weights(x)
            learner.update(Rk[p], x)
            diag.append(learner.diagnostics())
        W = self.combine(pi, Wd)
        daily, to = self.simulate(W)
        return {
            "name": learner.name,
            "pi": pd.DataFrame(pi, index=self.dates, columns=list(Wd)),
            "weights": pd.DataFrame(W, index=self.dates, columns=self.assets),
            "daily": daily,
            "turnover": pd.Series(to, index=self.dates),
            "diagnostics": diag,
        }

    def run_static(self, name: str, W: np.ndarray):
        daily, to = self.simulate(W)
        return {
            "name": name,
            "weights": pd.DataFrame(W, index=self.dates, columns=self.assets),
            "daily": daily,
            "turnover": pd.Series(to, index=self.dates),
        }


def best_constant_mix(Rk: pd.DataFrame, gamma=5.0) -> np.ndarray:
    """Oracle: the fixed convex combination maximising realised quadratic utility (hindsight)."""
    from scipy.optimize import minimize

    M = Rk.values
    K = M.shape[1]

    def f(p):
        r = M @ p
        return -1e4 * (r - 0.5 * gamma * r ** 2).mean()

    res = minimize(f, np.full(K, 1 / K), method="SLSQP", bounds=[(0, 1)] * K,
                   constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}])
    return np.clip(res.x, 0, 1) / np.clip(res.x, 0, 1).sum()


def grid_search(bt: Backtester, Wd: dict, make_learner, grid: list[dict], is_end,
                features=None, rf=None):
    """Evaluate every hyper-parameter set on the IN-SAMPLE window only.

    Returns a DataFrame (one row per config) with the in-sample Sharpe and the daily
    in-sample Sharpe used later by the Deflated Sharpe Ratio.
    """
    import pandas as pd
    from .metrics import summarize

    rows = []
    for cfg in grid:
        res = bt.run_meta(make_learner(**cfg), Wd, features=features)
        r = res["daily"].loc[:is_end]
        ex = r - (0 if rf is None else rf.reindex(r.index).fillna(0))
        s = summarize(r, rf)
        rows.append({**cfg, "IS_Sharpe": s["Sharpe"], "IS_CAGR": s["CAGR"], "IS_MaxDD": s["MaxDD"],
                     "IS_SR_daily": ex.mean() / ex.std()})
    return pd.DataFrame(rows).sort_values("IS_Sharpe", ascending=False).reset_index(drop=True)
