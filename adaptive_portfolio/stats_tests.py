"""
Statistical inference for back-tests.

* sharpe_diff_hac       Ledoit & Wolf (2008) HAC delta-method test of H0: SR_a = SR_b
                        (robust to fat tails, autocorrelation and the strong correlation
                        between the two strategies - the Jobson-Korkie/Memmel test is not).
* sharpe_diff_bootstrap Circular block bootstrap CI for SR_a - SR_b.
* utility_dm_test       Diebold-Mariano-type HAC t-test on per-period quadratic-utility
                        differences: tests the learner's own objective directly.
* probabilistic_sharpe  Bailey & Lopez de Prado (2012) PSR.
* deflated_sharpe       Bailey & Lopez de Prado (2014) DSR - corrects for the number of
                        configurations tried during in-sample tuning (selection bias).
* min_track_record      years needed to detect a given Sharpe difference.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

EULER = 0.5772156649


def _nw_cov(Y: np.ndarray, lags: int | None = None) -> np.ndarray:
    """Newey-West HAC covariance of the columns of Y (already centred)."""
    T = Y.shape[0]
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    S = Y.T @ Y / T
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Y[l:].T @ Y[:-l] / T
        S += w * (G + G.T)
    return S


def sharpe_diff_hac(a, b, ann=252):
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = len(a)
    mu1, mu2 = a.mean(), b.mean()
    g1, g2 = (a ** 2).mean(), (b ** 2).mean()
    s1, s2 = np.sqrt(g1 - mu1 ** 2), np.sqrt(g2 - mu2 ** 2)
    diff = mu1 / s1 - mu2 / s2
    grad = np.array([g1 / s1 ** 3, -g2 / s2 ** 3, -mu1 / (2 * s1 ** 3), mu2 / (2 * s2 ** 3)])
    Y = np.column_stack([a - mu1, b - mu2, a ** 2 - g1, b ** 2 - g2])
    Psi = _nw_cov(Y)
    se = np.sqrt(grad @ Psi @ grad / T)
    z = diff / se
    return {"dSR_ann": diff * np.sqrt(ann), "se_ann": se * np.sqrt(ann), "z": z,
            "p_value": 2 * (1 - stats.norm.cdf(abs(z)))}


def sharpe_diff_bootstrap(a, b, block=21, n_boot=2000, ann=252, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T] % T
    A, B = a[idx], b[idx]
    d = (A.mean(1) / A.std(1) - B.mean(1) / B.std(1)) * np.sqrt(ann)
    point = (a.mean() / a.std() - b.mean() / b.std()) * np.sqrt(ann)
    lo, hi = np.quantile(d, [0.025, 0.975])
    # two-sided p-value from the centred bootstrap distribution
    p = np.mean(np.abs(d - d.mean()) >= abs(point))
    return {"dSR_ann": point, "ci95": (lo, hi), "p_value": p}


def utility_dm_test(ra, rb, gamma=5.0):
    """HAC t-test of mean[u(ra) - u(rb)], u(r) = r - gamma/2 r^2 (period returns)."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    d = (ra - 0.5 * gamma * ra ** 2) - (rb - 0.5 * gamma * rb ** 2)
    T = len(d)
    var = _nw_cov((d - d.mean())[:, None])[0, 0]
    t = d.mean() / np.sqrt(var / T)
    return {"mean_diff": d.mean(), "t": t, "p_value": 2 * (1 - stats.norm.cdf(abs(t)))}


def probabilistic_sharpe(r, sr_benchmark=0.0):
    """P(true SR > benchmark). r: per-period returns; SRs are per-period (not annualised)."""
    r = np.asarray(r, float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    sk = stats.skew(r)
    ku = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return stats.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom)


def expected_max_sharpe(n_trials, var_sr):
    """Expected maximum of n_trials SR estimates under H0 (true SR = 0)."""
    if n_trials <= 1:
        return 0.0
    return np.sqrt(var_sr) * ((1 - EULER) * stats.norm.ppf(1 - 1 / n_trials)
                              + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(r, n_trials, trial_srs):
    """DSR: PSR against the SR one would expect from the best of `n_trials` useless configs.

    trial_srs: per-period Sharpe ratios of all configurations tried (same frequency as r).
    """
    sr0 = expected_max_sharpe(n_trials, np.var(trial_srs, ddof=1))
    return probabilistic_sharpe(r, sr0), sr0


def min_track_record_years(d_sr_ann, corr, alpha=0.05, power=0.8):
    """Years of data needed to detect an annualised Sharpe difference d_sr between two
    strategies with return correlation `corr` (iid normal approximation:
    Var(SR_a - SR_b) ~ 2(1 - corr)/T_years)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z ** 2 * 2 * (1 - corr) / d_sr_ann ** 2
