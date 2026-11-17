"""
Online meta-learners that combine K expert portfolios.

Setting (full-information online convex optimisation over the simplex Delta_K):
    at rebalance t the learner outputs pi_t in Delta_K, the investor holds
        w_t = sum_k pi_{t,k} w_t^{(k)}
    and then observes the period (excess) return of EVERY expert, R_t in R^K.
    Because buy-and-hold is linear, the combined gross period return is exactly pi_t' R_t.

Loss: negative quadratic utility of the combined period return
        l_t(pi) = -( pi'R_t - (gamma/2) (pi'R_t)^2 )
  -> convex in pi; gamma is the usual annualised risk-aversion coefficient (period mean and
     variance scale identically with horizon, so gamma does not depend on the rebalancing
     frequency). Gradient: g_t = -(1 - gamma * pi_t'R_t) R_t.
  Feeding the learner the *linearised* losses  l~_{t,k} = g_{t,k}  (the "gradient trick")
  makes it compete with the best fixed CONVEX COMBINATION of experts, not merely the best
  single expert.

Learners
  StaticMix          fixed pi (e.g. 1/3 each) - the natural null hypothesis
  FollowTheLeader    100% in the expert with best trailing utility (hard switching)
  Hedge              exponential weights; alpha>0 gives Fixed-Share (Herbster & Warmuth 1998),
                     rho<1 gives discounting
  AdaptiveBOA        Bernstein Online Aggregation (Wintenberger 2017) with per-expert
                     adaptive learning rates (second-order, scale-free), discounting rho and
                     fixed-share alpha for non-stationarity.  <- main proposal
  ONS                Online Newton Step on log-wealth (Hazan, Agarwal & Kale 2007) - growth-
                     optimal (Kelly) alternative, O(log T) regret for exp-concave losses.
  ContextualGate     pi_t = softmax(Theta x_t + b), x_t = observable market-state features,
                     trained online with AdaGrad on the utility loss. Learns *conditional*
                     preferences without predefined regimes.
  Hierarchical       AdaptiveBOA over {experts..., ContextualGate}: the gate is admitted only
                     to the extent it earns its keep out of sample.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class MetaLearner:
    name = "meta"

    def reset(self, K: int, n_features: int = 0):
        self.K = K
        self.n_features = n_features
        self.t = 0

    def weights(self, x=None) -> np.ndarray:
        raise NotImplementedError

    def update(self, R: np.ndarray, x=None):
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}


# ----------------------------------------------------------------------------------------
class StaticMix(MetaLearner):
    def __init__(self, pi=None, name="StaticMix"):
        self.pi0, self.name = pi, name

    def reset(self, K, n_features=0):
        super().reset(K, n_features)
        self.pi = np.full(K, 1 / K) if self.pi0 is None else np.asarray(self.pi0, float)

    def weights(self, x=None):
        return self.pi.copy()

    def update(self, R, x=None):
        self.t += 1


class FollowTheLeader(MetaLearner):
    def __init__(self, window=52, gamma=5.0, name="FollowTheLeader"):
        self.window, self.gamma, self.name = window, gamma, name

    def reset(self, K, n_features=0):
        super().reset(K, n_features)
        self.hist = []

    def weights(self, x=None):
        if not self.hist:
            return np.full(self.K, 1 / self.K)
        H = np.array(self.hist[-self.window:])
        u = (H - 0.5 * self.gamma * H ** 2).mean(0)
        pi = np.zeros(self.K)
        pi[np.argmax(u)] = 1.0
        return pi

    def update(self, R, x=None):
        self.hist.append(np.asarray(R, float))
        self.t += 1


class _UtilityLossMixin:
    """Linearised quadratic-utility regrets, scaled by a running bound B_t."""

    gamma: float

    def _regrets(self, pi, R):
        rp = pi @ R
        g = -(1.0 - self.gamma * rp) * R          # gradient of the loss at pi
        r = pi @ g - g                            # instantaneous regret vs each expert
        self.B = max(self.B, np.max(np.abs(r)), 1e-12)
        return r / self.B


class Hedge(_UtilityLossMixin, MetaLearner):
    """Exponentially weighted average forecaster (+ Fixed-Share, + discounting)."""

    def __init__(self, eta=0.5, gamma=5.0, alpha=0.0, name=None):
        self.eta, self.gamma, self.alpha = eta, gamma, alpha
        self.name = name or ("FixedShare" if alpha > 0 else "Hedge")

    def reset(self, K, n_features=0):
        super().reset(K, n_features)
        self.pi = np.full(K, 1 / K)
        self.B = 0.0

    def weights(self, x=None):
        return self.pi.copy()

    def update(self, R, x=None):
        r = self._regrets(self.pi, np.asarray(R, float))
        v = self.pi * np.exp(self.eta * r)
        v /= v.sum()
        self.pi = (1 - self.alpha) * v + self.alpha / self.K
        self.t += 1


class AdaptiveBOA(_UtilityLossMixin, MetaLearner):
    """Bernstein Online Aggregation with per-expert adaptive learning rates.

        r~_{k,t}   = scaled instantaneous regret of expert k (in [-1, 1])
        Rcum_k    <- rho * Rcum_k + r~_k - eta_k r~_k^2          (second-order correction)
        V_k       <- rho * V_k + r~_k^2
        eta_k      = min(1/2, sqrt(log K / V_k))
        pi_k      ∝ pi0_k * eta_k * exp(eta_k * Rcum_k);   pi <- (1-alpha) pi + alpha/K

    rho = 1, alpha = 0 is the stationary algorithm with a second-order (variance-adaptive)
    regret bound against the best convex combination. rho < 1 bounds the effective memory
    to ~1/(1-rho) periods; alpha > 0 keeps every expert alive so the learner can switch.
    """

    def __init__(self, gamma=5.0, rho=1.0, alpha=0.0, name="AdaptiveBOA"):
        self.gamma, self.rho, self.alpha, self.name = gamma, rho, alpha, name

    def reset(self, K, n_features=0):
        super().reset(K, n_features)
        self.Rcum = np.zeros(K)
        self.V = np.zeros(K)
        self.eta = np.full(K, 0.5)
        self.B = 0.0
        self.pi = np.full(K, 1 / K)

    def weights(self, x=None):
        return self.pi.copy()

    def update(self, R, x=None):
        r = self._regrets(self.pi, np.asarray(R, float))
        self.Rcum = self.rho * self.Rcum + r - self.eta * r ** 2
        self.V = self.rho * self.V + r ** 2
        self.eta = np.minimum(0.5, np.sqrt(np.log(self.K) / np.maximum(self.V, 1e-12)))
        logw = np.log(self.eta) + self.eta * self.Rcum
        p = _softmax(logw)
        self.pi = (1 - self.alpha) * p + self.alpha / self.K
        self.t += 1

    def diagnostics(self):
        return {"eta": self.eta.copy()}


class ONS(MetaLearner):
    """Online Newton Step on the log-wealth loss  l_t(pi) = -log(1 + pi'R_t)."""

    def __init__(self, beta=1.0, delta=0.5, rho=1.0, name="ONS"):
        self.beta, self.delta, self.rho, self.name = beta, delta, rho, name

    def reset(self, K, n_features=0):
        super().reset(K, n_features)
        self.A = self.delta * np.eye(K)
        self.pi = np.full(K, 1 / K)

    def weights(self, x=None):
        return self.pi.copy()

    def update(self, R, x=None):
        R = np.asarray(R, float)
        g = -R / (1.0 + self.pi @ R)
        self.A = self.rho * self.A + (1 - self.rho) * self.delta * np.eye(self.K) + np.outer(g, g)
        y = self.pi - np.linalg.solve(self.A, g) / self.beta
        self.pi = _project_simplex_A(y, self.A, self.pi)
        self.t += 1


def _project_simplex_A(y, A, x0):
    K = len(y)
    res = minimize(
        lambda x: (x - y) @ A @ (x - y), x0, jac=lambda x: 2 * A @ (x - y), method="SLSQP",
        bounds=[(0, 1)] * K, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
        options={"ftol": 1e-14, "maxiter": 200},
    )
    x = np.clip(res.x, 0, None)
    return x / x.sum()


class ContextualGate(MetaLearner):
    """Softmax gating network pi_t = softmax(Theta x_t + b), trained online (AdaGrad)."""

    def __init__(self, gamma=5.0, lr=0.1, l2=1e-2, name="ContextGate"):
        self.gamma, self.lr, self.l2, self.name = gamma, lr, l2, name

    def reset(self, K, n_features=0):
        super().reset(K, n_features)
        self.Theta = np.zeros((K, n_features + 1))  # last column = bias
        self.G2 = np.full_like(self.Theta, 1e-8)
        self.B = 0.0
        self._last = None

    def _z(self, x):
        xb = np.append(np.zeros(self.n_features) if x is None else np.asarray(x, float), 1.0)
        return xb, self.Theta @ xb

    def weights(self, x=None):
        xb, z = self._z(x)
        pi = _softmax(z)
        self._last = (xb, pi)
        return pi

    def update(self, R, x=None):
        R = np.asarray(R, float)
        xb, pi = self._last if self._last is not None else (self._z(x)[0], _softmax(self._z(x)[1]))
        rp = pi @ R
        g = -(1.0 - self.gamma * rp) * R
        self.B = max(self.B, np.max(np.abs(g - pi @ g)), 1e-12)
        g = g / self.B
        dz = pi * (g - pi @ g)                      # softmax Jacobian times g
        grad = np.outer(dz, xb) + self.l2 * self.Theta
        self.G2 += grad ** 2
        self.Theta -= self.lr * grad / np.sqrt(self.G2)
        self.t += 1


class Hierarchical(MetaLearner):
    """Top-level learner over the K experts plus the contextual gate as an extra expert."""

    def __init__(self, top: MetaLearner, gate: ContextualGate, name="BOA+Context"):
        self.top, self.gate, self.name = top, gate, name

    def reset(self, K, n_features=0):
        super().reset(K, n_features)
        self.top.reset(K + 1)
        self.gate.reset(K, n_features)

    def weights(self, x=None):
        self._pg = self.gate.weights(x)
        th = self.top.weights()
        return th[: self.K] + th[self.K] * self._pg

    def update(self, R, x=None):
        R = np.asarray(R, float)
        self.top.update(np.append(R, self._pg @ R))
        self.gate.update(R, x)
        self.t += 1

    def diagnostics(self):
        return {"top_weights": self.top.weights()}
