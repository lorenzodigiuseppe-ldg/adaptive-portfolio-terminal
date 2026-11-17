"""
Base strategies ("experts"). Each maps a T x N window of past daily returns to long-only
weights summing to one. Every expert is re-estimated walk-forward at each rebalance date
using only data available at that date.

    MeanVariance : max  mu'w - (delta/2) w'Sigma w      (Bayes-Stein mu, LW-CC Sigma)
    RiskParity   : equal risk contribution (Spinu 2013 convex formulation, solved by
                   cyclical coordinate descent, Griveau-Billion et al. 2013)
    MinCVaR      : min CVaR_beta of the empirical loss distribution
                   (Rockafellar & Uryasev 2000 linear program)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize

from .estimators import bayes_stein_mean, ewma_cov, ledoit_wolf_cc

ANN = 252


class Expert:
    name = "expert"
    lookback = 252

    def weights(self, R: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class MeanVariance(Expert):
    risk_aversion: float = 4.0       # annualised delta
    lookback_mu: int = 504
    lookback_cov: int = 252
    w_max: float = 0.40
    shrink_mean: bool = True
    name: str = "MVO"

    @property
    def lookback(self):
        return max(self.lookback_mu, self.lookback_cov)

    def weights(self, R):
        n = R.shape[1]
        S = ledoit_wolf_cc(R[-self.lookback_cov:]) * ANN
        Rm = R[-self.lookback_mu:]
        mu = (bayes_stein_mean(Rm, ledoit_wolf_cc(Rm)) if self.shrink_mean else Rm.mean(0)) * ANN
        d = self.risk_aversion

        def f(w):
            return -(mu @ w - 0.5 * d * w @ S @ w)

        def g(w):
            return -(mu - d * S @ w)

        res = minimize(
            f, np.full(n, 1 / n), jac=g, method="SLSQP",
            bounds=[(0, self.w_max)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        return _clean(res.x)


@dataclass
class RiskParity(Expert):
    lookback: int = 126
    use_ewma: bool = False
    halflife: float = 63.0
    name: str = "RiskParity"

    def weights(self, R):
        X = R[-self.lookback:]
        S = ewma_cov(X, self.halflife) if self.use_ewma else ledoit_wolf_cc(X)
        return erc_weights(S)


def erc_weights(S, b=None, tol=1e-10, max_iter=10_000):
    """Equal (or budgeted) risk contribution via cyclical coordinate descent."""
    n = S.shape[0]
    b = np.full(n, 1 / n) if b is None else np.asarray(b)
    y = 1 / np.sqrt(np.diag(S))
    y /= y.sum()
    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            c = S[i] @ y - S[i, i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4 * S[i, i] * b[i])) / (2 * S[i, i])
        if np.max(np.abs(y - y_old)) < tol:
            break
    return y / y.sum()


@dataclass
class MinCVaR(Expert):
    beta: float = 0.95
    lookback: int = 504
    w_max: float = 0.40
    name: str = "MinCVaR"

    def weights(self, R):
        X = R[-self.lookback:]
        T, n = X.shape
        # variables: w (n), zeta (1), u (T)
        c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1 / ((1 - self.beta) * T))])
        # -X w - zeta - u <= 0
        A_ub = np.hstack([-X, -np.ones((T, 1)), -np.eye(T)])
        b_ub = np.zeros(T)
        A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)])[None, :]
        bounds = [(0, self.w_max)] * n + [(None, None)] + [(0, None)] * T
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
        if not res.success:
            return np.full(n, 1 / n)
        return _clean(res.x[:n])


@dataclass
class EqualWeight(Expert):
    lookback: int = 1
    name: str = "EqualWeight"

    def weights(self, R):
        n = R.shape[1]
        return np.full(n, 1 / n)


@dataclass
class FixedWeights(Expert):
    w: np.ndarray = None
    lookback: int = 1
    name: str = "Fixed"

    def weights(self, R):
        return np.asarray(self.w, dtype=float)


def _clean(w, eps=1e-6):
    w = np.where(w < eps, 0.0, w)
    return w / w.sum()


def risk_contributions(w, S):
    m = S @ w
    return w * m / (w @ m)
