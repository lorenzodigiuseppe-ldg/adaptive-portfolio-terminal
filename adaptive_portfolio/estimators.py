"""
Parameter estimators (the "calibration" layer).

* ledoit_wolf_cc : Ledoit & Wolf (2004, "Honey, I shrunk the sample covariance matrix")
                   shrinkage toward the constant-correlation target. Better than the scaled
                   identity target when asset volatilities differ widely (IEF vs IEMG).
* bayes_stein_mean : Jorion (1986) Bayes-Stein shrinkage of the mean vector toward the
                   mean of the global-minimum-variance portfolio.
* ewma_cov        : RiskMetrics-style exponentially weighted covariance (optional).
"""
from __future__ import annotations

import numpy as np


def ledoit_wolf_cc(X: np.ndarray, return_shrinkage: bool = False):
    """Ledoit-Wolf constant-correlation shrinkage. X is T x N (rows = observations)."""
    X = np.asarray(X, dtype=float)
    t, n = X.shape
    y = X - X.mean(axis=0)
    sample = y.T @ y / t
    var = np.diag(sample).copy()
    sd = np.sqrt(var)
    corr = sample / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1))
    prior = rbar * np.outer(sd, sd)
    np.fill_diagonal(prior, var)

    y2 = y ** 2
    phi_mat = y2.T @ y2 / t - sample ** 2
    phi = phi_mat.sum()

    term1 = (y ** 3).T @ y / t
    theta = term1 - var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    rho = np.trace(phi_mat) + rbar * (np.outer(1 / sd, sd) * theta).sum()

    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    kappa = (phi - rho) / gamma if gamma > 0 else 0.0
    delta = float(np.clip(kappa / t, 0.0, 1.0))
    sigma = delta * prior + (1 - delta) * sample
    return (sigma, delta) if return_shrinkage else sigma


def bayes_stein_mean(X: np.ndarray, sigma: np.ndarray | None = None, return_phi: bool = False):
    """Jorion (1986) Bayes-Stein estimator of expected returns."""
    X = np.asarray(X, dtype=float)
    t, n = X.shape
    mu = X.mean(axis=0)
    if sigma is None:
        sigma = ledoit_wolf_cc(X)
    inv = np.linalg.inv(sigma)
    ones = np.ones(n)
    w_mv = inv @ ones / (ones @ inv @ ones)
    mu0 = w_mv @ mu
    d = mu - mu0
    phi = (n + 2) / ((n + 2) + t * d @ inv @ d)
    phi = float(np.clip(phi, 0.0, 1.0))
    mu_bs = (1 - phi) * mu + phi * mu0
    return (mu_bs, phi) if return_phi else mu_bs


def ewma_cov(X: np.ndarray, halflife: float = 63.0) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    t = X.shape[0]
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(t - 1, -1, -1)
    w /= w.sum()
    m = w @ X
    y = X - m
    return (y * w[:, None]).T @ y
