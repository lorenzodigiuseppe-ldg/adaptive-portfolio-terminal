"""Matplotlib figures (validated colour-blind-safe categorical palette, thin marks)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"


def setup():
    plt.rcParams.update({
        "figure.figsize": (11, 4.2), "figure.dpi": 110, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#c9c8c3", "axes.grid": True,
        "grid.color": "#e8e7e3", "grid.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
        "lines.linewidth": 2.0, "font.size": 10,
    })


# colour follows the entity, never its position in a given chart
ENTITY_COLORS = {
    "AdaptiveBOA": "#2a78d6", "MVO": "#6250d6", "RiskParity": "#eb6834", "MinCVaR": "#1baf7a",
    "1/3 static mix": NEUTRAL, "StaticMix": NEUTRAL, "1/N": "#e87ba4", "60/40 SPY/IEF": "#008300",
    "BOA+Context": "#e34948", "ONS (log-wealth)": "#eda100", "Hedge/FixedShare": "#1baf7a",
    "ContextGate": "#6250d6", "FollowTheLeader": "#eb6834",
}


def _colors(names, highlight=None):
    cols, spare = {}, iter(p for p in PALETTE if p not in [ENTITY_COLORS.get(n) for n in names])
    for n in names:
        cols[n] = ENTITY_COLORS.get(n) or next(spare)
    return cols


def wealth(results, names, start=None, end=None, ax=None, title="Growth of $1 (net of costs)",
           split=None, log=True):
    ax = ax or plt.subplots()[1]
    cols = _colors(names)
    for n in names:
        r = results[n]["daily"].loc[start:end]
        w = (1 + r).cumprod()
        ax.plot(w.index, w.values, color=cols[n], label=n,
                lw=2.4 if n == names[0] else 1.6, zorder=3 if n == names[0] else 2)
        ax.annotate(n, (w.index[-1], w.values[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    if split is not None:
        ax.axvline(pd.Timestamp(split), color=INK2, lw=1, ls="--")
        ax.text(pd.Timestamp(split), 0.02, " out-of-sample →", va="bottom", fontsize=9, color=INK2,
                transform=ax.get_xaxis_transform())
    if log:
        ax.set_yscale("log")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncol=min(4, len(names)))
    return ax


def drawdowns(results, names, start=None, end=None, ax=None, title="Drawdown", legend=True):
    ax = ax or plt.subplots(figsize=(11, 3.2))[1]
    cols = _colors(names)
    for n in names:
        dd = drawdown(results[n]["daily"].loc[start:end])
        ax.plot(dd.index, dd.values, color=cols[n], label=n, lw=2.0 if n == names[0] else 1.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, loc="left")
    if legend:
        ax.legend(loc="lower left", ncol=min(4, len(names)))
    return ax


def stacked(df: pd.DataFrame, ax=None, title="", colors=None):
    ax = ax or plt.subplots(figsize=(11, 3.4))[1]
    if colors is None:
        colors = ([ENTITY_COLORS[c] for c in df.columns] if all(c in ENTITY_COLORS for c in df.columns)
                  else PALETTE)
    ax.stackplot(df.index, df.T.values, labels=df.columns, colors=colors[: df.shape[1]],
                 edgecolor="white", linewidth=0.3, alpha=0.95)
    ax.set_ylim(0, 1)
    ax.set_xlim(df.index[0], df.index[-1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(False)
    return ax


def rolling_sharpe(results, names, window=252, ax=None, rf=None, title="Rolling 1-year Sharpe ratio"):
    ax = ax or plt.subplots(figsize=(11, 3.4))[1]
    cols = _colors(names)
    for n in names:
        r = results[n]["daily"]
        ex = r - (0 if rf is None else rf.reindex(r.index).fillna(0))
        rs = ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(252)
        ax.plot(rs.index, rs.values, color=cols[n], label=n, lw=1.8 if n == names[0] else 1.2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", ncol=min(4, len(names)))
    return ax


def cumulative_utility_gap(pi_returns: dict, reference: pd.Series, gamma=5.0, ax=None,
                           title="Cumulative utility vs 1/3 static mix (above 0 = learner ahead)"):
    ax = ax or plt.subplots(figsize=(11, 3.4))[1]
    u_ref = reference - 0.5 * gamma * reference ** 2
    for i, (n, r) in enumerate(pi_returns.items()):
        u = r - 0.5 * gamma * r ** 2
        ax.plot(u.index, (u - u_ref).cumsum().values, color=_colors([n])[n], label=n, lw=1.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncol=3)
    return ax
