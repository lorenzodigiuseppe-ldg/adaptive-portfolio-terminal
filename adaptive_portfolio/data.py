"""
Data layer.

* load_prices()      -> real adjusted-close data (yfinance first, stooq fallback, or local CSV)
                        with optional proxy back-fill for ETFs with short histories.
* load_risk_free()   -> daily risk-free rate from the 13-week T-bill (^IRX), 0 if unavailable.
* simulate_market()  -> a 3-regime, fat-tailed synthetic market calibrated to the stylised
                        behaviour of the 7 ETFs. Used ONLY to unit-test the pipeline and to
                        check that the learners can track a known regime structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICKERS = ["DBC", "GLD", "IEF", "TLT", "SPY", "VEA", "IEMG"]

# Proxies used to extend history back before an ETF's inception (spliced on returns).
#   IEMG (inception Oct-2012)  <- EEM  (iShares MSCI Emerging Markets, 2003)
#   VEA  (inception Jul-2007)  <- EFA  (iShares MSCI EAFE, 2001)
PROXIES = {"IEMG": "EEM", "VEA": "EFA"}


# ----------------------------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------------------------
def _download_yf(tickers, start, end):
    import yfinance as yf  # imported lazily so the library works without it

    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px


def _download_stooq(tickers, start, end):
    frames = {}
    for t in tickers:
        url = f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        frames[t] = df["Close"]
    px = pd.DataFrame(frames)
    return px.loc[start:end] if end else px.loc[start:]


def load_prices(
    tickers=TICKERS,
    start="2004-01-01",
    end=None,
    source="yfinance",
    backfill_proxies=True,
    csv_path=None,
):
    """Return a DataFrame of daily adjusted close prices (columns = tickers).

    With backfill_proxies=True, missing early history of IEMG / VEA is reconstructed from
    the daily returns of EEM / EFA and chained to the first real price. This lets the
    back-test include 2008 (GFC) and is disclosed in the results.
    """
    tickers = list(tickers)
    need = tickers + ([PROXIES[t] for t in tickers if t in PROXIES] if backfill_proxies else [])
    if source == "csv":
        raw = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
        raw = raw.loc[start:end] if end else raw.loc[start:]
        raw = raw[[c for c in need if c in raw.columns]]
    else:
        loader = {"yfinance": _download_yf, "stooq": _download_stooq}[source]
        raw = loader(need, start, end).sort_index()

    px = raw[tickers].copy()
    if backfill_proxies:
        for t, p in PROXIES.items():
            if t not in px or p not in raw:
                continue
            first = px[t].first_valid_index()
            proxy_ret = raw[p].pct_change().dropna()
            link = proxy_ret.loc[proxy_ret.index <= first]
            if len(link) < 2:
                continue
            back = _chain_back(px.loc[first, t], link)
            px.loc[back.index, t] = back
    return px.dropna()


def _chain_back(first_price, link_returns: pd.Series) -> pd.Series:
    """Reconstruct prices strictly before the first real date.

    link_returns: proxy daily returns for dates d_0 < ... < d_m = first.
    P_{d_{i-1}} = P_{d_i} / (1 + r_{d_i}), starting from the observed P_first.
    """
    r = link_returns.values
    dates = link_returns.index
    out = np.empty(len(r) - 1)
    p = first_price
    for i in range(len(r) - 1, 0, -1):
        p = p / (1 + r[i])
        out[i - 1] = p
    return pd.Series(out, index=dates[:-1])


def load_risk_free(index, start="2004-01-01", end=None, csv_path=None):
    """Daily risk-free return aligned to `index` (13-week T-bill yield / 252). 0 on failure."""
    if csv_path is not None:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if "IRX" in df:
            irx = df["IRX"].reindex(index).ffill().bfill()
            return (irx / 100.0 / 252.0).rename("rf")
    try:
        import yfinance as yf

        irx = yf.download("^IRX", start=start, end=end, auto_adjust=False, progress=False)["Close"]
        irx = irx.squeeze().reindex(index).ffill().bfill()
        return (irx / 100.0 / 252.0).rename("rf")
    except Exception:  # noqa: BLE001
        return pd.Series(0.0, index=index, name="rf")


# ----------------------------------------------------------------------------------------
# Synthetic market (validation only)
# ----------------------------------------------------------------------------------------
def _corr_matrix(tickers, eq_eq, eq_bond, bond_bond, gold_bond, gold_eq, dbc_eq, dbc_gold, dbc_bond):
    groups = {"SPY": "eq", "VEA": "eq", "IEMG": "eq", "IEF": "bd", "TLT": "bd", "GLD": "au", "DBC": "cm"}
    tab = {
        ("eq", "eq"): eq_eq, ("eq", "bd"): eq_bond, ("bd", "bd"): bond_bond,
        ("au", "bd"): gold_bond, ("au", "eq"): gold_eq, ("cm", "eq"): dbc_eq,
        ("cm", "au"): dbc_gold, ("cm", "bd"): dbc_bond,
    }
    n = len(tickers)
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = groups[tickers[i]], groups[tickers[j]]
            v = tab.get((a, b), tab.get((b, a)))
            C[i, j] = C[j, i] = v
    w, V = np.linalg.eigh(C)  # make PSD
    C = V @ np.diag(np.clip(w, 1e-4, None)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def simulate_market(tickers=TICKERS, start="2006-02-06", end="2026-11-13", seed=7, df=5):
    """3-regime Markov-switching multivariate-t returns. Returns (prices, regimes)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T, n = len(idx), len(tickers)
    order = {t: i for i, t in enumerate(tickers)}

    def vec(d):
        return np.array([d[t] for t in tickers])

    regimes = {
        # name: (annual mean, annual vol, correlation matrix)
        "calm": (
            vec(dict(SPY=.14, VEA=.09, IEMG=.10, IEF=.02, TLT=.03, GLD=.04, DBC=.05)),
            vec(dict(SPY=.13, VEA=.14, IEMG=.18, IEF=.06, TLT=.12, GLD=.15, DBC=.16)),
            _corr_matrix(tickers, .85, -.30, .90, .30, .05, .40, .30, -.10),
        ),
        "crisis": (
            vec(dict(SPY=-.35, VEA=-.40, IEMG=-.45, IEF=.08, TLT=.20, GLD=.10, DBC=-.35)),
            vec(dict(SPY=.35, VEA=.38, IEMG=.42, IEF=.09, TLT=.20, GLD=.22, DBC=.30)),
            _corr_matrix(tickers, .92, -.50, .92, .35, -.05, .60, .20, -.30),
        ),
        "inflation": (
            vec(dict(SPY=-.15, VEA=-.15, IEMG=-.18, IEF=-.10, TLT=-.25, GLD=.08, DBC=.25)),
            vec(dict(SPY=.22, VEA=.22, IEMG=.25, IEF=.08, TLT=.17, GLD=.18, DBC=.22)),
            _corr_matrix(tickers, .88, .50, .92, .20, .10, .30, .35, -.20),
        ),
    }
    names = list(regimes)
    P = np.array([
        [0.995, 0.003, 0.002],
        [0.013, 0.985, 0.002],
        [0.008, 0.002, 0.990],
    ])
    s = np.empty(T, dtype=int)
    s[0] = 0
    for t in range(1, T):
        s[t] = rng.choice(3, p=P[s[t - 1]])

    chol = {k: np.linalg.cholesky(v[2]) for k, v in regimes.items()}
    scale_t = np.sqrt((df - 2) / df)
    R = np.empty((T, n))
    for k, name in enumerate(names):
        m, v, _ = regimes[name]
        mask = s == k
        z = rng.standard_normal((mask.sum(), n)) @ chol[name].T
        chi = rng.chisquare(df, size=(mask.sum(), 1))
        eps = z / np.sqrt(chi / df) * scale_t  # unit-variance multivariate t
        R[mask] = m / 252 + eps * v / np.sqrt(252)
    prices = pd.DataFrame(100 * np.cumprod(1 + R, axis=0), index=idx, columns=tickers)
    reg = pd.Series(np.array(names)[s], index=idx, name="regime")
    _ = order
    return prices, reg


# ----------------------------------------------------------------------------------------
# Yahoo Finance refresh using only the standard library (no yfinance needed)
# ----------------------------------------------------------------------------------------
FETCH_TICKERS = ["DBC", "GLD", "IEF", "TLT", "SPY", "VEA", "IEMG", "EEM", "EFA", "^IRX"]


def fetch_yahoo_csv(out_path, tickers=FETCH_TICKERS, start_epoch=1072915200, progress=None):
    """Download daily adjusted closes from Yahoo's chart API and write a wide CSV.

    Columns: Date, one per ticker ('^IRX' saved as 'IRX'). Returns the DataFrame.
    """
    import json
    import time
    import urllib.parse
    import urllib.request

    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/130 Safari/537.36"}
    cols = {}
    for i, t in enumerate(tickers):
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(t)}"
               f"?period1={start_epoch}&period2={int(time.time())}&interval=1d"
               f"&events=div%2Csplits&includeAdjustedClose=true")
        with urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=30) as r:
            res = json.load(r)["chart"]["result"][0]
        ind = res["indicators"]
        adj = ind["adjclose"][0]["adjclose"] if "adjclose" in ind else ind["quote"][0]["close"]
        idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
        s = pd.Series(adj, index=idx, dtype=float).dropna()
        cols[t.replace("^", "")] = s[~s.index.duplicated(keep="last")]
        if progress:
            progress((i + 1) / len(tickers), t)
        time.sleep(0.3)
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "Date"
    df.to_csv(out_path)
    return df
