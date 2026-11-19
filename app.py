"""
ADAPTIVE PORTFOLIO TERMINAL
Interactive front-end for the online expert-aggregation portfolio engine.

Run:   streamlit run app.py        (or double-click run_app.bat on Windows)
"""
from __future__ import annotations

import inspect
import os
import sys
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from adaptive_portfolio import app_core as C  # noqa: E402
from adaptive_portfolio import stats_tests as ST  # noqa: E402
from adaptive_portfolio.data import fetch_yahoo_csv  # noqa: E402
from adaptive_portfolio.features import features_at  # noqa: E402

DATA_DIR = os.environ.get("APT_DATA_DIR", HERE)      # desktop build stores data in the user's profile
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "etf_prices_yahoo.csv")

# ======================================================================================
# Look & feel
# ======================================================================================
AMBER, AMBER_DIM = "#ff9e1b", "#a8650f"
BG, PANEL, GRID, BORDER = "#000000", "#070707", "#1c1c1c", "#2a2a2a"
INK, INK2, INK3 = "#e8e8e8", "#a9a9a9", "#6f6f6f"
UP, DOWN = "#0ca30c", "#e66767"
# dark-mode categorical palette (validated: CVD + contrast on #0d0d0d)
COL = {
    "AdaptiveBOA": "#3987e5", "BOA+Context": "#e66767", "Hedge/FixedShare": "#199e70",
    "ONS (log-wealth)": "#c98500", "ContextGate": "#9085e9", "FollowTheLeader": "#d95926",
    "1/3 static mix": "#8a8985", "MVO": "#9085e9", "RiskParity": "#d95926", "MinCVaR": "#199e70",
    "1/N": "#d55181", "60/40 SPY/IEF": "#0ca30c", "Oracle const. mix*": "#c98500",
    "Momentum": "#c98500", "Core": "#3987e5", "Balanced 10%": "#199e70", "Growth 12%": "#c98500",
    "Aggressive 15%": "#e66767",
}
ASSET_COL = {"DBC": "#3987e5", "GLD": "#c98500", "IEF": "#199e70", "TLT": "#0ca30c",
             "SPY": "#d55181", "VEA": "#9085e9", "IEMG": "#d95926"}
NAMES = {"DBC": "Commodities", "GLD": "Gold", "IEF": "UST 7-10Y", "TLT": "UST 20Y+",
         "SPY": "S&P 500", "VEA": "Developed ex-US", "IEMG": "Emerging Mkts"}
PAGES = {
    "HOME": "Product Overview",
    "MON": "Market Monitor", "PORT": "Strategy Backtest", "ALLOC": "Recommended Allocation",
    "EXPT": "Expert Analytics", "LRN": "Learner Diagnostics", "RISK": "Risk Analytics",
    "STAT": "Statistical Validation", "DATA": "Data Management", "PNL": "Daily P&L",
    "STRS": "Stress Test & Downside Protection",
}
PAGE_KEYS = list(PAGES)

st.set_page_config(page_title="Adaptive Portfolio Terminal", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, .stText, input, textarea, select, button, label {{
  font-family: 'IBM Plex Mono', Consolas, 'Courier New', monospace !important; }}
.stApp {{ background: {BG}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; height: 2rem; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 2rem; max-width: 100%; }}
section[data-testid="stSidebar"] {{ background: #050505; border-right: 1px solid {AMBER_DIM}; }}
section[data-testid="stSidebar"] * {{ font-size: 12px; }}
h1, h2, h3, h4 {{ color: {AMBER} !important; letter-spacing: .04em; }}
.topbar {{ display:flex; align-items:center; justify-content:space-between; background:#0b0b0b;
  border:1px solid {BORDER}; border-left:4px solid {AMBER}; padding:6px 12px; margin-bottom:6px; }}
.brand {{ color:{AMBER}; font-weight:600; font-size:15px; letter-spacing:.12em; }}
.brand small {{ color:{INK2}; font-weight:400; letter-spacing:.05em; margin-left:10px; }}
.clock {{ color:{INK2}; font-size:12px; text-align:right; }}
.clock b {{ color:{INK}; }}
.tape {{ overflow:hidden; white-space:nowrap; border-top:1px solid {BORDER}; border-bottom:1px solid {BORDER};
  background:#050505; padding:4px 0; margin-bottom:8px; }}
.tape-inner {{ display:inline-block; padding-left:100%; animation: scroll 45s linear infinite; }}
.tape span.t {{ margin-right:34px; font-size:12px; }}
@keyframes scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-100%); }} }}
.up {{ color:{UP}; }} .down {{ color:{DOWN}; }} .flat {{ color:{INK2}; }}
.amber {{ color:{AMBER}; }}
.ptitle {{ color:{AMBER}; font-size:12px; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  border-bottom:1px solid {AMBER_DIM}; padding:2px 0 3px 0; margin:10px 0 6px 0; }}
.ptitle span {{ color:{INK3}; font-weight:400; text-transform:none; letter-spacing:0; margin-left:8px; }}
.kpi {{ background:{PANEL}; border:1px solid {BORDER}; padding:8px 10px 7px 10px; height:100%; }}
.kpi .l {{ color:{AMBER}; font-size:10.5px; letter-spacing:.06em; text-transform:uppercase;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.kpi .v {{ color:{INK}; font-size:clamp(15px, 1.55vw, 21px); font-weight:500; line-height:1.3;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.kpi .s {{ font-size:11px; color:{INK2}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.kpi-cap {{ color:{INK3}; font-size:11px; margin:-2px 0 4px 0; }}
.tile {{ background:{PANEL}; border:1px solid {BORDER}; padding:7px 9px; }}
.tile .tk {{ color:{AMBER}; font-weight:600; font-size:13px; }}
.tile .nm {{ color:{INK3}; font-size:10px; }}
.tile .px {{ color:{INK}; font-size:18px; }}
.tile .row {{ display:flex; justify-content:space-between; font-size:10.5px; color:{INK2}; }}
.note {{ color:{INK3}; font-size:11px; border-left:2px solid {AMBER_DIM}; padding-left:8px; margin:6px 0; }}
.fkey {{ color:#000; background:{AMBER}; padding:0 5px; font-weight:600; margin-right:4px; }}
div[role="radiogroup"] {{ gap: 4px; flex-wrap: wrap; }}
div[role="radiogroup"] > label {{ background:#0e0e0e; border:1px solid {BORDER}; padding:3px 10px !important;
  margin-right:0 !important; border-radius:0; }}
div[role="radiogroup"] > label:hover {{ border-color:{AMBER}; }}
div[role="radiogroup"] > label:has(input:checked) {{ background:{AMBER}; border-color:{AMBER}; }}
div[role="radiogroup"] > label:has(input:checked) p {{ color:#000 !important; font-weight:600; }}
div[role="radiogroup"] > label > div:first-child,
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{ display:none !important; }}
[data-testid="stRadio"] label[data-baseweb="radio"] {{ background:#0e0e0e; border:1px solid {BORDER};
  padding:3px 10px !important; margin-right:4px !important; border-radius:0; }}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{ background:{AMBER}; border-color:{AMBER}; }}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) * {{ color:#000 !important; font-weight:600; }}
.stButton > button, .stDownloadButton > button {{ background:#0e0e0e; color:{AMBER}; border:1px solid {AMBER};
  border-radius:0; font-weight:600; letter-spacing:.06em; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:{AMBER}; color:#000; }}
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stDateInput input {{
  background:#0b0b0b !important; border-radius:0 !important; border-color:{BORDER} !important; }}
.stTextInput input {{ color:{AMBER} !important; }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; }}
[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:0; }}
hr {{ border-color:{BORDER}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s: str):
    """Render HTML (lines stripped so markdown never treats indentation as code)."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def title(t: str, sub: str = ""):
    html(f'<div class="ptitle">{t}{f"<span>{sub}</span>" if sub else ""}</div>')


def pct(v, d=1, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:+.{d}%}" if sign else f"{v:.{d}%}"


def updown(v, d=1):
    if v is None or np.isnan(v):
        return '<span class="flat">–</span>'
    cls, arr = ("up", "▲") if v > 0 else ("down", "▼") if v < 0 else ("flat", "■")
    return f'<span class="{cls}">{arr} {abs(v):.{d}%}</span>'


def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def kpi_row(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        with c:
            html(kpi(*it))


def sparkline(series: pd.Series, w=150, h=30):
    y = series.values.astype(float)
    if len(y) < 2:
        return ""
    lo, hi = np.nanmin(y), np.nanmax(y)
    rng = hi - lo or 1
    xs = np.linspace(0, w, len(y))
    ys = h - (y - lo) / rng * (h - 4) - 2
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    col = UP if y[-1] >= y[0] else DOWN
    return (f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="{col}" stroke-width="1.5" points="{pts}"/></svg>')


# ---- version-tolerant Streamlit helpers ---------------------------------------------
def _width_kw(fn):
    p = inspect.signature(fn).parameters
    if "width" in p and p["width"].default in ("stretch", "content"):
        return {"width": "stretch"}
    if "use_container_width" in p:
        return {"use_container_width": True}
    return {}


PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def show(fig, key=None):
    kw = {"config": PLOT_CFG, **_width_kw(st.plotly_chart)}
    if key:
        kw["key"] = key
    st.plotly_chart(fig, **kw)


def table(df, height=None, hide_index=False):
    kw = dict(_width_kw(st.dataframe))
    if height:
        kw["height"] = height
    if hide_index and "hide_index" in inspect.signature(st.dataframe).parameters:
        kw["hide_index"] = True
    st.dataframe(df, **kw)


def rerun():
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun"))()


def styler_map(sty, fn, subset=None):
    return (sty.map if hasattr(sty, "map") else sty.applymap)(fn, subset=subset)


def color_sign(v):
    if not isinstance(v, (int, float)) or pd.isna(v):
        return ""
    return f"color: {UP}" if v > 0 else f"color: {DOWN}" if v < 0 else ""


PCT_COLS = ["CAGR", "Vol", "MaxDD", "CVaR95(d)", "Worst month", "Turnover/yr"]


def fmt_perf(df: pd.DataFrame):
    f = {c: "{:.1%}" for c in df.columns if c in PCT_COLS}
    f.update({c: "{:.2f}" for c in df.columns if c not in PCT_COLS})
    sty = df.style.format(f, na_rep="–")
    return styler_map(sty, color_sign, subset=[c for c in ["CAGR", "Sharpe"] if c in df.columns])


# ---- plotly theme --------------------------------------------------------------------
def base_layout(fig, height=340, title_text=None, legend=True, yfmt=None, log=False):
    fig.update_layout(
        height=height, paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="IBM Plex Mono, Consolas, monospace", size=11, color=INK2),
        margin=dict(l=52, r=18, t=40 if title_text else 30, b=44),
        title=dict(text=title_text or "", font=dict(color=AMBER, size=12), x=0, xanchor="left"),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10.5),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111111", bordercolor=AMBER, font=dict(family="IBM Plex Mono", color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=INK2), automargin=True,
                     showspikes=True, spikecolor=AMBER_DIM, spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#333333", linecolor=BORDER, tickfont=dict(color=INK2), automargin=True,
                     tickformat=yfmt)
    if log:
        fig.update_yaxes(type="log")
    return fig


def mark_oos(fig, split):
    fig.add_vline(x=pd.Timestamp(split), line=dict(color=AMBER_DIM, width=1, dash="dash"))
    fig.add_annotation(x=pd.Timestamp(split), y=1, yref="paper", text=" OUT-OF-SAMPLE ▸", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=AMBER_DIM, size=10))


def fig_lines(series: dict, height=340, log=False, yfmt=None, title_text=None, split=None, widths=None):
    fig = go.Figure()
    for i, (n, s) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=n, mode="lines",
                                 line=dict(color=COL.get(n, ASSET_COL.get(n, INK2)),
                                           width=(widths or {}).get(n, 1.6))))
    base_layout(fig, height, title_text, yfmt=yfmt, log=log)
    if split is not None:
        mark_oos(fig, split)
    return fig


def fig_stacked(df: pd.DataFrame, colors: dict, height=300, title_text=None, split=None):
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c].values, name=c, mode="lines", stackgroup="one",
                                 line=dict(width=0.4, color="#000"), fillcolor=colors.get(c, INK2),
                                 hovertemplate="%{y:.1%}"))
    base_layout(fig, height, title_text, yfmt=".0%")
    fig.update_yaxes(range=[0, 1])
    if split is not None:
        mark_oos(fig, split)
    return fig


def fig_hbar(s: pd.Series, colors=None, height=260, title_text=None, fmt=".1%"):
    s = s.sort_values()
    cols = [(colors or {}).get(i, AMBER) for i in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=[f"{i}" for i in s.index], orientation="h", marker_color=cols,
                           text=[format(v, fmt) for v in s.values], textposition="outside",
                           textfont=dict(color=INK), hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    base_layout(fig, height, title_text, legend=False, yfmt=None)
    fig.update_xaxes(tickformat=fmt, range=[min(0, s.min() * 1.25), max(s.max() * 1.25, 0.05)])
    fig.update_layout(hovermode="closest", bargap=0.35)
    return fig


def fig_heat(df: pd.DataFrame, height=330, title_text=None, zmid=0, fmt=".2f", zmin=None, zmax=None,
             scale="div"):
    if scale == "div":
        cs = [[0, "#d95926"], [0.5, "#262624"], [1, "#3987e5"]]
    else:
        cs = [[0, "#0b1a2e"], [1, "#3987e5"]]
    fig = go.Figure(go.Heatmap(z=df.values, x=[str(c) for c in df.columns], y=[str(i) for i in df.index],
                               colorscale=cs, zmid=zmid, zmin=zmin, zmax=zmax, texttemplate="%{z:" + fmt + "}",
                               textfont=dict(size=10, color=INK), hoverongaps=False, xgap=2, ygap=2,
                               colorbar=dict(thickness=8, tickfont=dict(color=INK2)),
                               hovertemplate="%{y} · %{x}: %{z:" + fmt + "}<extra></extra>"))
    base_layout(fig, height, title_text, legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


# ======================================================================================
# Daily auto-refresh of market data
# ======================================================================================
CHECK_FILE = os.path.join(DATA_DIR, ".last_data_check")


def _now_new_york():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    except Exception:  # noqa: BLE001  (no tz database on some Windows installs)
        return datetime.utcnow() - pd.Timedelta(hours=5)


def expected_last_close() -> pd.Timestamp:
    """Most recent US trading session whose close should be available (holidays ignored)."""
    ny = _now_new_york()
    d = pd.Timestamp(ny.date())
    if ny.hour * 60 + ny.minute < 16 * 60 + 30:      # before 4:30pm ET -> previous session
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _last_csv_date():
    try:
        with open(DATA_PATH, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode(errors="ignore").strip().splitlines()
        return pd.Timestamp(tail[-1].split(",")[0])
    except Exception:  # noqa: BLE001
        return None


def auto_refresh(min_hours_between_checks=1.0):
    """Download new closes if the file is behind the latest session. Returns a status string."""
    last = _last_csv_date()
    target = expected_last_close()
    if last is not None and last >= target:
        return "ok"
    try:
        if os.path.exists(CHECK_FILE) and time.time() - os.path.getmtime(CHECK_FILE) < min_hours_between_checks * 3600:
            return "checked"       # tried recently (e.g. market holiday): don't hammer Yahoo
        with open(CHECK_FILE, "w") as f:
            f.write(datetime.now().isoformat())
        fetch_yahoo_csv(DATA_PATH)
        return "updated"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {exc}"


if "refresh_status" not in st.session_state:
    with st.spinner("Checking for new market data…"):
        st.session_state["refresh_status"] = auto_refresh()


# ======================================================================================
# Cached engine
# ======================================================================================
def data_mtime():
    return os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0


@st.cache_data(show_spinner=False)
def get_market(mtime):
    return C.load_market(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_bt(mtime, es: C.EngineSettings):
    _, returns, rf = get_market(mtime)
    return C.make_backtester(returns, rf, es)


@st.cache_resource(show_spinner=False)
def get_experts(mtime, es: C.EngineSettings, xs: C.ExpertSettings):
    return C.expert_weights(get_bt(mtime, es), xs)


@st.cache_data(show_spinner=False)
def get_features(mtime, es: C.EngineSettings):
    bt = get_bt(mtime, es)
    return features_at(bt.returns, bt.rebal_idx, lag=es.lag)


@st.cache_resource(show_spinner=False)
def get_bench(mtime, es, xs, gamma):
    return C.run_benchmarks(get_bt(mtime, es), get_experts(mtime, es, xs), gamma=gamma)


# ======================================================================================
# Session defaults & sidebar
# ======================================================================================
DEFAULTS = dict(
    nav="HOME", profile="Core", max_lev=2.0, universe="Core 3", learner="AdaptiveBOA", gamma=5.0, rho=1.0, alpha=0.01, eta=0.25, ons_beta=1.0,
    gate_lr=0.05, gate_l2=0.01, ftl_window=52, rebal="Weekly", refit=4, lag=1, tc=5.0,
    split=date(2016, 12, 31), mvo_delta=4.0, mvo_cap=0.40, mvo_lb_mu=504, mvo_lb_cov=252, rp_lb=126,
    cvar_beta=0.95, cvar_lb=504, cvar_cap=0.40,
    bench=["1/3 static mix", "1/N", "60/40 SPY/IEF"], base="1/3 static mix", cmd="",
)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_defaults():
    for k, v in DEFAULTS.items():
        if k not in ("nav", "cmd"):
            st.session_state[k] = v


def on_command():
    c = st.session_state.get("cmd", "").strip().upper().replace("<GO>", "").strip()
    alias = {"HELP": "DATA", "START": "HOME", "PRODUCT": "HOME", "BT": "PORT", "ALLOCATION": "ALLOC", "W": "ALLOC", "EXP": "EXPT",
             "LEARN": "LRN", "VAR": "RISK", "TEST": "STAT", "P&L": "PNL", "DAY": "PNL", "PL": "PNL",
             "STRESS": "STRS", "PROOF": "STRS", "CRISIS": "STRS"}
    c = alias.get(c, c)
    if c in PAGES:
        st.session_state["nav"] = c
    st.session_state["cmd"] = ""


RHO_OPTS = [0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
ALPHA_OPTS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
REBAL = {"Weekly": 5, "Bi-weekly": 10, "Monthly": 21}

with st.sidebar:
    html(f'<div class="brand" style="font-size:13px">◆ CONTROL PANEL</div>')
    title("Product profile")
    st.selectbox("Risk profile", list(C.PROFILES), key="profile",
                 help="Core = the portfolio as built. The others scale it to a volatility target with leverage "
                      "(borrowing at T-bills + 0.5%). More return potential, deeper drawdowns.")
    if st.session_state.profile != "Core":
        st.select_slider("Maximum leverage", [1.0, 1.5, 2.0, 3.0], key="max_lev")
    st.selectbox("Strategy set", ["Core 3", "Extended 6"], key="universe",
                 help="Core 3: MVO, Risk Parity, Min-CVaR. Extended 6 adds Momentum, 1/N and 60/40 as "
                      "strategies the learner can choose.")
    title("Meta-learner")
    st.selectbox("Algorithm", C.LEARNERS, key="learner")
    L = st.session_state.learner
    if L != "1/3 static mix":
        st.slider("Risk aversion γ", 1.0, 20.0, step=0.5, key="gamma",
                  help="Curvature of the quadratic-utility loss the learner minimises.")
    if L in ("AdaptiveBOA", "BOA+Context", "ONS (log-wealth)"):
        st.select_slider("Memory ρ (1 = never forget)", RHO_OPTS, key="rho",
                         help="Effective memory ≈ 1/(1-ρ) rebalances.")
    if L in ("AdaptiveBOA", "BOA+Context", "Hedge/FixedShare"):
        st.select_slider("Fixed-share α", ALPHA_OPTS, key="alpha",
                         help="Minimum weight kept on every expert so the learner can switch.")
    if L == "Hedge/FixedShare":
        st.select_slider("Learning rate η", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0], key="eta")
    if L == "ONS (log-wealth)":
        st.select_slider("ONS β", [0.25, 0.5, 1.0, 2.0, 4.0], key="ons_beta")
    if L in ("ContextGate", "BOA+Context"):
        st.select_slider("Gate learning rate", [0.01, 0.02, 0.05, 0.1, 0.2], key="gate_lr")
        st.select_slider("Gate L2 penalty", [0.001, 0.01, 0.1], key="gate_l2")
    if L == "FollowTheLeader":
        st.select_slider("Look-back (rebalances)", [13, 26, 52, 104], key="ftl_window")

    title("Execution")
    st.selectbox("Rebalance", list(REBAL), key="rebal")
    st.select_slider("Re-optimise experts every N rebalances", [1, 2, 4, 8], key="refit")
    st.select_slider("Execution lag (days)", [0, 1, 2], key="lag")
    st.number_input("Transaction cost (bp, one-way)", 0.0, 100.0, step=1.0, key="tc")
    st.date_input("In-sample ends", key="split", min_value=date(2010, 1, 1), max_value=date(2024, 12, 31))

    with st.expander("EXPERT PARAMETERS"):
        st.slider("MVO risk aversion δ", 1.0, 15.0, step=0.5, key="mvo_delta")
        st.select_slider("MVO mean window (d)", [252, 504, 756], key="mvo_lb_mu")
        st.select_slider("MVO cov window (d)", [126, 252, 504], key="mvo_lb_cov")
        st.select_slider("MVO max weight", [0.25, 0.3, 0.4, 0.5, 0.75, 1.0], key="mvo_cap")
        st.select_slider("Risk-parity cov window (d)", [63, 126, 252, 504], key="rp_lb")
        st.select_slider("CVaR confidence β", [0.90, 0.95, 0.975, 0.99], key="cvar_beta")
        st.select_slider("CVaR scenario window (d)", [252, 504, 756], key="cvar_lb")
        st.select_slider("CVaR max weight", [0.25, 0.3, 0.4, 0.5, 0.75, 1.0], key="cvar_cap")

    title("Comparison")
    st.multiselect("Benchmarks on charts", C.BENCHMARKS, key="bench")
    st.selectbox("Delta vs", C.BENCHMARKS, key="base")
    st.button("RESET DEFAULTS", on_click=reset_defaults)

S = st.session_state
es = C.EngineSettings(rebal_every=REBAL[S.rebal], expert_refit=int(S.refit), lag=int(S.lag), tc_bps=float(S.tc))
xs = C.ExpertSettings(mvo_delta=float(S.mvo_delta), mvo_lb_mu=int(S.mvo_lb_mu), mvo_lb_cov=int(S.mvo_lb_cov),
                      mvo_cap=float(S.mvo_cap), rp_lb=int(S.rp_lb), cvar_beta=float(S.cvar_beta),
                      cvar_lb=int(S.cvar_lb), cvar_cap=float(S.cvar_cap),
                      universe="extended" if S.universe == "Extended 6" else "core")
rs = C.RiskSettings(target=C.PROFILES[S.profile], max_lev=float(S.max_lev))
LEVERED = rs.target is not None
PROFILE_TXT = S.profile + (f" · max {rs.max_lev:g}x" if LEVERED else "") + f" · {S.universe}"
ls = C.LearnerSettings(name=S.learner, gamma=float(S.gamma), rho=float(S.rho), alpha=float(S.alpha),
                       eta=float(S.eta), ons_beta=float(S.ons_beta), gate_lr=float(S.gate_lr),
                       gate_l2=float(S.gate_l2), ftl_window=int(S.ftl_window))
SPLIT = pd.Timestamp(S.split).strftime("%Y-%m-%d")
OOS_START = (pd.Timestamp(SPLIT) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
PPY = 252 / es.rebal_every

# ======================================================================================
# Header
# ======================================================================================
mt = data_mtime()
have_data = mt > 0
if have_data:
    prices, returns, rf = get_market(mt)
    snap = C.market_snapshot(prices)

now = datetime.now()
asof = prices.index[-1].strftime("%d-%b-%Y").upper() if have_data else "NO DATA"
_rs = st.session_state.get("refresh_status", "")
feed = ('<span class="up">● LIVE</span>' if _rs in ("ok", "updated", "checked")
        else '<span class="down">● OFFLINE</span>')
html(f"""
<div class="topbar">
  <div class="brand">◆ APT <small>ADAPTIVE PORTFOLIO TERMINAL · online aggregation of MVO / ERC / Min-CVaR</small></div>
  <div class="clock"><span class="amber">{PROFILE_TXT.upper()}</span> · {feed} <b>{now.strftime('%d-%b-%Y %H:%M').upper()}</b> local · close of <b>{asof}</b></div>
</div>""")

if have_data:
    items = "".join(
        f'<span class="t"><span class="amber">{t}</span> {snap.loc[t, "Last"]:.2f} {updown(snap.loc[t, "1D"], 2)}</span>'
        for t in snap.index)
    html(f'<div class="tape"><div class="tape-inner">{items}{items}</div></div>')

c1, c2 = st.columns([1, 3.2])
with c1:
    st.text_input("command", key="cmd", on_change=on_command, placeholder="FUNCTION <GO>  e.g. ALLOC",
                  label_visibility="collapsed")
with c2:
    st.radio("nav", PAGE_KEYS, key="nav", horizontal=True, label_visibility="collapsed",
             format_func=lambda k: f"{PAGE_KEYS.index(k) + 1} {k}")
page = S.nav
html(f'<div class="ptitle" style="font-size:14px;margin-top:2px"><span class="fkey">{page}</span>{PAGES[page]}</div>')

if not have_data and page != "DATA":
    html('<div class="note">No price file found. Go to <b>DATA</b> and press UPDATE FROM YAHOO FINANCE.</div>')
    page = "DATA"

# ======================================================================================
# Engine run (shared by pages)
# ======================================================================================
if have_data and page != "DATA" and page != "MON":
    with st.spinner("Optimising experts…"):
        bt = get_bt(mt, es)
        Wd = get_experts(mt, es, xs)
        F = get_features(mt, es)
        bench = get_bench(mt, es, xs, ls.gamma)
    res_core, learner = C.run_strategy(bt, Wd, ls, F)
    res = C.apply_risk_target(bt, res_core, rs)
    EXP = list(Wd)
    allres = {ls.name: res, **{k: v for k, v in bench.items() if k != ls.name}}
    rf_ = bt.rf if bt.rf is not None else None


def period_bounds(choice):
    return {"FULL": (None, None), "IN-SAMPLE": (None, SPLIT), "OUT-OF-SAMPLE": (OOS_START, None)}[choice]


def perf_kpis(r, base_r, start, end):
    k = C.kpis(r, rf_, PPY, start, end)
    b = C.kpis(base_r, rf_, PPY, start, end)

    def d(key, p=True, inv=False):
        dv = k[key] - b[key]
        good = dv < 0 if inv else dv > 0
        cls = "up" if good else "down" if dv != 0 else "flat"
        txt = f"{dv:+.1%}" if p else f"{dv:+.2f}"
        return f'<span class="{cls}">{txt}</span>'

    return [
        ("CAGR", pct(k["CAGR"]), d("CAGR")),
        ("Vol", pct(k["Vol"]), d("Vol", inv=True)),
        ("Sharpe", f'{k["Sharpe"]:.2f}', d("Sharpe", p=False)),
        ("Sortino", f'{k["Sortino"]:.2f}', d("Sortino", p=False)),
        ("Max DD", pct(k["MaxDD"]), d("MaxDD")),
        ("CVaR95 1D", pct(k["CVaR95(d)"], 2), d("CVaR95(d)", inv=True)),
        ("Turnover", pct(k.get("Turnover/yr", np.nan), 0), d("Turnover/yr", inv=True)),
    ]


# ======================================================================================
# Pages
# ======================================================================================
if page == "HOME":
    rec = C.recommend_now(bt, xs, learner, F, res=res, rs=rs)
    k = C.kpis(res, rf_, PPY, OOS_START, None)
    grown = 10000 * float((1 + res["daily"].loc[OOS_START:]).prod())
    top = rec["weights"].sort_values(ascending=False).head(3)
    html(f'''<div class="kpi" style="border-left:4px solid {AMBER};padding:14px 16px;margin-bottom:10px">
    <div class="l">Adaptive Portfolio Terminal</div>
    <div class="v" style="font-size:24px;white-space:normal">One portfolio that learns which strategy to trust.</div>
    <div class="s" style="white-space:normal;font-size:12.5px;margin-top:4px">Seven liquid ETFs across stocks, bonds, gold and
    commodities. {len(EXP)} portfolio strategies compete; an online-learning algorithm reallocates between them every
    week from live market data. You choose the risk level. Every number is explained and stress-tested.</div></div>''')
    title("Your product today", f"{PROFILE_TXT} · results since {OOS_START}, years the model never saw in tuning")
    kpi_row([
        ("Return / yr", pct(k["CAGR"]), "net of costs"),
        ("Volatility", pct(k["Vol"]), "annual"),
        ("Sharpe", f'{k["Sharpe"]:.2f}', "return per unit of risk"),
        ("Max drawdown", pct(k["MaxDD"]), "worst peak-to-trough"),
        ("$10,000 became", f"${grown:,.0f}", f"since {OOS_START[:4]}"),
        ("Leverage now", f'{rec["leverage"]:.2f}x', S.profile),
    ])
    title("Choose your risk profile", "same engine, four risk levels, next to two simple reference portfolios")
    ptab, curves = C.profile_table(bt, res_core, rf_, PPY, OOS_START, max_lev=rs.max_lev,
                                   extra={"1/N": bench["1/N"], "60/40 SPY/IEF": bench["60/40 SPY/IEF"]})
    a, b = st.columns([1.25, 1])
    with a:
        money = [c for c in ptab.columns if c.startswith("$")][0]
        sty = ptab.style.format({"Return / yr": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
                                 "Max drawdown": "{:.1%}", "Worst year": "{:+.1%}", "Avg leverage": "{:.2f}x",
                                 "Max leverage": "{:.2f}x", money: "${:,.0f}"})
        table(styler_map(sty, color_sign, subset=["Worst year"]))
        html('<div class="note">Higher profiles borrow to scale the same portfolio up (cost: T-bills + 0.5%). '
             'Return rises, and so do volatility and drawdowns; the Sharpe ratio does not improve. '
             'Leverage requires a margin account and can lose more than an unlevered portfolio.</div>')
    with b:
        show(fig_lines(curves, height=330, log=True, widths={S.profile: 2.8},
                       title_text="GROWTH OF $1 BY PROFILE"), key="home_prof")
    a, b, c = st.columns(3)
    with a:
        title("What it holds today")
        html("".join(f'<div class="tile" style="margin-bottom:4px"><div class="row"><span class="tk">{t}</span>'
                     f'<span class="nm">{NAMES[t]}</span></div><div class="px">{v:.0%}</div></div>'
                     for t, v in top.items()))
        html('<div class="note">Full portfolio and order ticket: <b>ALLOC</b>.</div>')
    with b:
        title("Which strategy it trusts")
        html("".join(f'<div class="tile" style="margin-bottom:4px"><div class="row"><span class="tk">{n}</span></div>'
                     f'<div class="px">{v:.0%}</div></div>'
                     for n, v in rec["pi"].sort_values(ascending=False).head(3).items()))
        html('<div class="note">How the trust evolved: <b>LRN</b>.</div>')
    with c:
        title("How it works")
        html('<div class="tile"><div class="row"><span class="tk">1 · COMPETE</span></div>'
             '<div class="nm" style="font-size:11.5px">Each strategy builds its own portfolio from the seven ETFs.</div></div>'
             '<div class="tile" style="margin-top:4px"><div class="row"><span class="tk">2 · LEARN</span></div>'
             '<div class="nm" style="font-size:11.5px">Every week the learner scores them and shifts money to what works.</div></div>'
             '<div class="tile" style="margin-top:4px"><div class="row"><span class="tk">3 · SCALE</span></div>'
             '<div class="nm" style="font-size:11.5px">Your risk profile sets how much of it you hold.</div></div>')
    title("Where to look next")
    html('<div class="note"><b>PORT</b> full track record vs benchmarks · <b>RISK</b> worst days, months and drawdowns · '
         '<b>STRS</b> behaviour in six market stress periods · <b>PNL</b> any single day explained · '
         '<b>STAT</b> statistical tests of whether the edge is real.</div>')
    html('<div class="note">Figures are backtested, net of trading costs, on daily closes from Yahoo Finance. '
         'The tests on STAT show no statistically significant advantage over simple portfolios; the product\'s '
         'value is a transparent, rule-based, risk-controlled allocation.</div>')

elif page == "MON":
    cols = st.columns(7)
    for c, t in zip(cols, snap.index):
        s = snap.loc[t]
        with c:
            html(f"""<div class="tile"><div class="row"><span class="tk">{t}</span><span class="nm">{NAMES[t]}</span></div>
            <div class="px">{s['Last']:.2f}</div><div class="row"><span>1D</span>{updown(s['1D'], 2)}</div>
            {sparkline(prices[t].iloc[-63:])}
            <div class="row"><span>1M</span>{updown(s['1M'])}</div><div class="row"><span>YTD</span>{updown(s['YTD'])}</div>
            <div class="row"><span>1Y</span>{updown(s['1Y'])}</div><div class="row"><span>VOL 1M</span><span>{s['Vol 1M']:.1%}</span></div></div>""")

    left, right = st.columns([1.8, 1])
    with left:
        title("Relative performance", "rebased to 100")
        a, b = st.columns([2, 2])
        with a:
            per = st.radio("Period", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"], index=4,
                           horizontal=True, key="mon_per", label_visibility="collapsed")
        with b:
            sel = st.multiselect("Assets", list(prices.columns), default=list(prices.columns), key="mon_assets",
                                 label_visibility="collapsed")
        days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}
        if per == "YTD":
            p = prices.loc[f"{prices.index[-1].year}-01-01":]
        elif per == "MAX":
            p = prices
        else:
            p = prices.iloc[-days[per]:]
        reb = 100 * p / p.iloc[0]
        show(fig_lines({t: reb[t] for t in sel}, height=360), key="mon_perf")
    with right:
        title("Correlation", "daily returns")
        win = st.radio("Window", [63, 126, 252, 756], index=2, horizontal=True, key="mon_corr",
                       format_func=lambda d: f"{d}D", label_visibility="collapsed")
        show(fig_heat(returns.iloc[-win:].corr().round(2), height=360, zmin=-1, zmax=1), key="mon_heat")

    title("Asset monitor")
    t = snap.copy()
    t.insert(0, "Name", [NAMES[i] for i in t.index])
    pc = [c for c in t.columns if c not in ("Name", "Last")]
    sty = t.style.format({**{c: "{:+.2%}" for c in pc}, "Last": "{:.2f}",
                          "Vol 1M": "{:.1%}", "Vol 1Y": "{:.1%}"})
    sty = styler_map(sty, color_sign, subset=["1D", "1W", "1M", "3M", "YTD", "1Y", "DD from 1Y high"])
    table(sty)

    title("Long-run statistics", f"{returns.index[0].year}–{returns.index[-1].year}, excess of T-bills")
    table(fmt_perf(C.asset_stats(returns, rf)))

elif page == "PORT":
    choice = st.radio("Window", ["FULL", "IN-SAMPLE", "OUT-OF-SAMPLE"], index=2, horizontal=True,
                      key="port_win", label_visibility="collapsed")
    start, end = period_bounds(choice)
    html(f'<div class="kpi-cap">{ls.name} · {PROFILE_TXT} · coloured figures = difference vs {S.base} '
         f'(green = better) · turnover per year</div>')
    kpi_row(perf_kpis(res, allres[S.base], start, end))

    title("Growth of $1", f"{ls.name} vs benchmarks · net of {es.tc_bps:g}bp costs")
    logy = st.checkbox("Log scale", value=True, key="port_log")
    names = [ls.name] + [b for b in S.bench if b != ls.name]
    curves = {n: (1 + allres[n]["daily"].loc[start:end]).cumprod() for n in names}
    show(fig_lines(curves, height=380, log=logy, split=SPLIT if choice == "FULL" else None,
                   widths={ls.name: 2.6}), key="port_w")

    title("Drawdown")
    dds = {n: c / c.cummax() - 1 for n, c in curves.items()}
    fig = fig_lines(dds, height=220, yfmt=".0%", widths={ls.name: 2.2})
    fig.update_layout(showlegend=False)
    show(fig, key="port_dd")

    title("Strategy comparison", choice.lower())
    tab = C.comparison_table(allres, rf_, PPY, start, end)
    order = [ls.name] + [n for n in tab.index if n != ls.name]
    table(fmt_perf(tab.loc[order]))
    html('<div class="note">* Oracle = best fixed mix of the three experts chosen with hindsight over the whole '
         'sample. Not investable; it is the ceiling for any constant combination.</div>')

elif page == "ALLOC":
    rec = C.recommend_now(bt, xs, learner, F, res=res, rs=rs)
    w = rec["weights"]
    lev_now = rec["leverage"]
    eff_n = 1 / (rec["base_weights"] ** 2).sum()
    if LEVERED:
        html(f'<div class="note">Risk profile <b>{S.profile}</b>: positions are scaled {lev_now:.2f}x to target '
             f'{rs.target:.0%} volatility. ' + (f'Weights add up to {lev_now:.0%}; the extra {lev_now - 1:.0%} is '
             f'borrowed (margin account required).' if lev_now > 1 else
             f'Weights add up to {lev_now:.0%}; the remaining {1 - lev_now:.0%} stays in cash.') + '</div>')
    kpi_row([
        ("As of close", rec["as_of"].strftime("%d-%b-%Y").upper(), "hold from next rebalance"),
        ("Leverage", f"{lev_now:.2f}x", S.profile),
        ("Ex-ante vol", pct(rec["ex_ante_vol"]), "LW-shrunk 1Y covariance"),
        ("Largest position", f"{w.idxmax()} {w.max():.1%}", NAMES[w.idxmax()]),
        ("Effective # assets", f"{eff_n:.1f}", "1 / Σw²"),
        ("Top risk contributor", f"{rec['risk_contrib'].idxmax()} {rec['risk_contrib'].max():.0%}", "share of variance"),
    ])
    a, b = st.columns([1.25, 1])
    with a:
        title("Target asset weights")
        show(fig_hbar(w, ASSET_COL, height=300), key="al_w")
    with b:
        title("Weight on each expert (π)", ls.name)
        show(fig_hbar(rec["pi"], COL, height=300), key="al_pi")

    a, b = st.columns([1.25, 1])
    with a:
        title("Weight vs risk contribution")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=w.index, x=w.values, name="Weight", orientation="h", marker_color="#3987e5"))
        fig.add_trace(go.Bar(y=w.index, x=rec["risk_contrib"].values, name="Risk contribution",
                             orientation="h", marker_color="#d95926"))
        base_layout(fig, 300, yfmt=None)
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(barmode="group", hovermode="y unified", bargap=0.25)
        show(fig, key="al_rc")
    with b:
        title("Expert portfolios today")
        ew = rec["expert_weights"].copy()
        ew.loc[f"► {ls.name}"] = w
        show(fig_heat(ew, height=300, fmt=".0%", zmin=0, zmax=max(0.4, float(ew.values.max())), scale="seq"),
             key="al_heat")

    title("Order ticket", "notional sizing at last close (no orders are sent)")
    c1, c2 = st.columns([1, 3])
    with c1:
        notional = st.number_input("Portfolio value ($)", 1000.0, 1e10, 100000.0, step=1000.0, key="notional")
    last = prices.iloc[-1]
    ticket = pd.DataFrame({"Name": [NAMES[t] for t in w.index], "Weight": w.values,
                           "Target $": w.values * notional, "Last": last[w.index].values})
    ticket["Shares"] = np.floor(ticket["Target $"] / ticket["Last"]).astype(int)
    ticket.index = w.index
    if LEVERED and abs(lev_now - 1) > 1e-9:
        html(f'<div class="note">{"Borrowing" if lev_now > 1 else "Cash"}: '
             f'${abs(lev_now - 1) * notional:,.0f} ({abs(lev_now - 1):.0%} of the portfolio value).</div>')
    with c2:
        table(ticket.style.format({"Weight": "{:.1%}", "Target $": "${:,.0f}", "Last": "{:.2f}", "Shares": "{:,}"}))
    st.download_button("DOWNLOAD TICKET (CSV)", ticket.to_csv().encode(), "apt_order_ticket.csv", "text/csv")

    title("Allocation history", ls.name)
    show(fig_stacked(res.get("base_weights", res["weights"]), ASSET_COL, height=300, split=SPLIT), key="al_hist")
    if LEVERED:
        title("Leverage history", f"scaling applied to reach the {rs.target:.0%} volatility target")
        fig = fig_lines({ls.name: res["leverage"]}, height=220, split=SPLIT)
        fig.add_hline(y=1, line=dict(color=INK3, width=1))
        fig.update_layout(showlegend=False)
        show(fig, key="al_lev")

elif page == "EXPT":
    choice = st.radio("Window", ["FULL", "IN-SAMPLE", "OUT-OF-SAMPLE"], index=2, horizontal=True,
                      key="ex_win", label_visibility="collapsed")
    start, end = period_bounds(choice)
    exres = {n: bench[n] for n in EXP}
    exres[ls.name] = res
    title("Expert performance", choice.lower())
    table(fmt_perf(C.comparison_table(exres, rf_, PPY, start, end)))

    tabs = st.tabs([f"{n} weights" for n in EXP])
    for tb, n in zip(tabs, EXP):
        with tb:
            show(fig_stacked(pd.DataFrame(Wd[n], index=bt.dates, columns=bt.assets), ASSET_COL, height=300,
                             split=SPLIT), key=f"ex_{n}")

    a, b = st.columns(2)
    with a:
        title("Can the learner tell them apart?", "in-sample expert correlations")
        lt = C.learnability(bt, Wd, SPLIT)
        table(lt.style.format({"Corr": "{:.2f}", "Yrs to detect ΔSR 0.2": "{:.0f}", "Yrs to detect ΔSR 0.5": "{:.1f}"}))
        html('<div class="note">Years of data needed to detect a Sharpe gap at 5% significance / 80% power. '
             'Highly correlated experts give the learner very little signal.</div>')
    with b:
        title("Rolling 1-year Sharpe", "experts")
        rs = {}
        for n in EXP:
            ex = bench[n]["daily"] - rf_.reindex(bench[n]["daily"].index).fillna(0)
            rs[n] = (ex.rolling(252).mean() / ex.rolling(252).std() * np.sqrt(252)).dropna()
        show(fig_lines(rs, height=300, split=SPLIT), key="ex_rs")

elif page == "LRN":
    title("Weight on each expert over time (π_t)", ls.name)
    show(fig_stacked(res["pi"], COL, height=320, split=SPLIT), key="lr_pi")
    a, b = st.columns(2)
    with a:
        title("Cumulative utility vs 1/3 static mix", "above 0 = learner ahead")
        gap = C.utility_gap(bt, res, bench["1/3 static mix"], ls.gamma)
        fig = fig_lines({ls.name: gap}, height=300, split=SPLIT)
        fig.add_hline(y=0, line=dict(color=INK3, width=1))
        show(fig, key="lr_gap")
    with b:
        eta = C.learning_rates(res)
        if eta is not None:
            title("Per-expert learning rates η", "adaptive (BOA)")
            show(fig_lines({c: eta[c] for c in eta.columns}, height=300, split=SPLIT), key="lr_eta")
        else:
            title("Learning rates")
            html('<div class="note">Adaptive learning rates are reported for AdaptiveBOA only.</div>')
    title("Does the allocation respond to market conditions?", "mean π by observable state")
    cp = C.conditional_pi(res, F)
    cols = st.columns(len(cp) or 1)
    for c, (k, v) in zip(cols, cp.items()):
        with c:
            fig = go.Figure()
            for n in v.columns:
                fig.add_trace(go.Bar(x=[str(i) for i in v.index], y=v[n].values, name=n, marker_color=COL[n]))
            base_layout(fig, 280, title_text=k, yfmt=".0%")
            fig.update_layout(barmode="group", hovermode="x unified")
            show(fig, key=f"lr_c_{k}")

elif page == "RISK":
    opts = [ls.name] + [n for n in C.BENCHMARKS if n != ls.name]
    a, b = st.columns([1, 2])
    with a:
        if S.get("risk_who") not in opts:
            S["risk_who"] = opts[0]
        who = st.selectbox("Strategy", opts, key="risk_who")
    with b:
        choice = st.radio("Window", ["FULL", "IN-SAMPLE", "OUT-OF-SAMPLE"], index=0, horizontal=True,
                          key="risk_win", label_visibility="collapsed")
    start, end = period_bounds(choice)
    r = allres[who]["daily"].loc[start:end]
    var95 = -np.quantile(r, 0.05)
    cvar95 = -r[r <= -var95].mean()
    m = (1 + r).resample("ME").prod() - 1
    kpi_row([
        ("VaR 95% (1d)", pct(var95, 2), "historical"),
        ("CVaR 95% (1d)", pct(cvar95, 2), "expected shortfall"),
        ("Worst day", pct(r.min(), 2), r.idxmin().strftime("%d-%b-%Y")),
        ("Worst month", pct(m.min()), m.idxmin().strftime("%b-%Y")),
        ("Skew", f"{r.skew():.2f}", "daily"),
        ("Excess kurtosis", f"{r.kurt():.1f}", "daily"),
    ])
    a, b = st.columns(2)
    with a:
        title("Rolling 63-day volatility")
        comp = [who] + ([S.base] if S.base != who else [])
        vol = {n: (allres[n]["daily"].rolling(63).std() * np.sqrt(252)).loc[start:end].dropna() for n in comp}
        show(fig_lines(vol, height=280, yfmt=".0%"), key="rk_vol")
    with b:
        title("Daily return distribution")
        fig = go.Figure(go.Histogram(x=r.values, nbinsx=120, marker_color="#3987e5", opacity=0.9, name=who))
        for v, lab, colr in [(-var95, "VaR95", AMBER), (-cvar95, "CVaR95", DOWN)]:
            fig.add_vline(x=v, line=dict(color=colr, width=1.5, dash="dash"))
            fig.add_annotation(x=v, y=1, yref="paper", text=lab, showarrow=False, font=dict(color=colr, size=10),
                               xanchor="right", yanchor="top")
        base_layout(fig, 280, legend=False)
        fig.update_xaxes(tickformat=".1%")
        fig.update_layout(hovermode="closest", bargap=0.05)
        show(fig, key="rk_hist")
    title("Monthly returns", who)
    mt_ = C.monthly_table(allres[who]["daily"].loc[start:end])
    show(fig_heat(mt_, height=max(260, 22 * len(mt_) + 60), fmt=".1%", zmid=0, zmin=-0.06, zmax=0.06),
         key="rk_month")
    title("Worst drawdowns")
    wd = C.worst_drawdowns(r)
    table(wd.style.format({"Depth": "{:.1%}", "Start": lambda d: d.strftime("%Y-%m-%d"),
                           "Trough": lambda d: d.strftime("%Y-%m-%d"),
                           "Recovered": lambda d: "not yet" if pd.isna(d) else d.strftime("%Y-%m-%d")}),
          hide_index=True)

elif page == "STAT":
    title("Out-of-sample tests", f"{ls.name} vs each benchmark · {OOS_START} onward")
    with st.spinner("Running HAC / bootstrap / Diebold-Mariano tests…"):
        pt = C.pairwise_tests(allres, bt, rf_, ls.name, C.BENCHMARKS, OOS_START, ls.gamma)
    if not pt.empty:
        sty = pt.style.format({"ΔSharpe": "{:+.3f}", "HAC p": "{:.3f}", "CI low": "{:+.3f}", "CI high": "{:+.3f}",
                               "ΔUtility bp/wk": "{:+.2f}", "DM p": "{:.3f}"})
        sty = styler_map(sty, lambda v: f"color: {AMBER}; font-weight: 600" if isinstance(v, float) and v < 0.05 else "",
                         subset=["HAC p", "DM p"])
        table(sty)
    html('<div class="note">ΔSharpe: Ledoit-Wolf (2008) HAC test on daily excess returns; CI: circular block '
         'bootstrap (21-day blocks); DM: Diebold-Mariano HAC test on weekly quadratic-utility differences. '
         'p &lt; 0.05 highlighted.</div>')

    title("Hyper-parameter tuning (AdaptiveBOA)", f"in-sample only, up to {SPLIT}")
    if st.button("RUN TUNING GRID  <GO>", key="tune_btn"):
        with st.spinner("Evaluating 36 configurations…"):
            S["tune"] = C.tune_boa(bt, Wd, SPLIT, rf_)
            S["tune_key"] = (mt, es, xs, SPLIT)
    tb = S.get("tune")
    if tb is not None and S.get("tune_key") == (mt, es, xs, SPLIT):
        best = tb.iloc[0]
        a, b = st.columns([1.3, 1])
        with a:
            g_sel = st.radio("γ", sorted(tb.gamma.unique()), index=list(sorted(tb.gamma.unique())).index(best.gamma),
                             horizontal=True, key="tune_g", format_func=lambda g: f"γ={g:g}")
            hm = tb[tb.gamma == g_sel].pivot(index="rho", columns="alpha", values="IS_Sharpe")
            hm.index = [f"ρ={i:g}" for i in hm.index]
            hm.columns = [f"α={c:g}" for c in hm.columns]
            show(fig_heat(hm, height=280, fmt=".3f", zmid=None, scale="seq",
                          title_text="In-sample Sharpe: a flat surface means the choice barely matters"),
                 key="st_hm")
        with b:
            ex = (res["daily"] - rf_.reindex(res["daily"].index).fillna(0)).loc[OOS_START:]
            dsr, sr0 = ST.deflated_sharpe(ex.values, len(tb), tb["IS_SR_daily"].values)
            kpi_row([("Best IS config", f"γ={best.gamma:g} ρ={best.rho:g} α={best.alpha:g}",
                      f"IS Sharpe {best.IS_Sharpe:.3f}")])
            kpi_row([("PSR (OOS SR>0)", f"{ST.probabilistic_sharpe(ex.values):.2f}", "current settings"),
                     ("Deflated SR", f"{dsr:.2f}", f"vs best of {len(tb)} null trials")])

            def apply_best():
                S["learner"] = "AdaptiveBOA"
                S["gamma"], S["rho"], S["alpha"] = float(best.gamma), float(best.rho), float(best.alpha)

            st.button("APPLY BEST CONFIG", on_click=apply_best)
        table(tb.head(12).style.format({"IS_Sharpe": "{:.3f}", "IS_CAGR": "{:.2%}", "IS_MaxDD": "{:.1%}",
                                        "IS_SR_daily": "{:.4f}"}), hide_index=True)

elif page == "PNL":
    opts = [ls.name] + [n for n in C.BENCHMARKS if n != ls.name]
    last_day = bt.bt_index[-1].date()
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        if S.get("pnl_who") not in opts:
            S["pnl_who"] = opts[0]
        who = st.selectbox("Portfolio", opts, key="pnl_who")
    with c2:
        day = st.date_input("Date", value=last_day, min_value=bt.bt_index[0].date(), max_value=last_day,
                            key="pnl_date")
    with c3:
        amount = st.number_input("Portfolio value ($)", 100.0, 1e10, 10000.0, step=1000.0, key="pnl_amt")
    b = C.day_breakdown(bt, allres[who], day)
    base_b = C.day_breakdown(bt, allres[S.base], day) if S.base != who else None
    r = b["net"]
    up = r > 0
    verdict = "POSITIVE DAY" if r > 0 else "NEGATIVE DAY" if r < 0 else "FLAT DAY"
    vcls = "up" if r > 0 else "down" if r < 0 else "flat"
    arrow = "▲" if r > 0 else "▼" if r < 0 else "■"
    dstr = b["date"].strftime("%A %d %B %Y")
    html(f'''<div class="kpi" style="border-left:4px solid {UP if up else DOWN if r < 0 else INK3};margin-bottom:8px">
    <div class="l">{who} · {dstr}</div>
    <div class="v {vcls}" style="font-size:30px">{arrow} {verdict} &nbsp; {r:+.2%} &nbsp; {'+' if r >= 0 else '−'}${abs(r) * amount:,.2f}</div>
    <div class="s">on a portfolio of ${amount:,.0f}</div></div>''')
    if not b["is_trading_day"]:
        html(f'<div class="note">{b["requested"].strftime("%a %d %b %Y")} was not a trading day (weekend or holiday): '
             f'showing the previous session, {b["date"].strftime("%a %d %b %Y")}.</div>')
    if abs(b.get("financing", 0.0)) > 0:
        html(f'<div class="note">Leverage {b["leverage"]:.2f}x: ETF contributions are scaled by the leverage; '
             f'financing for the day {b["financing"]:+.3%} ({b["financing"] * amount:+,.2f} $).</div>')
    if b["rebalance_day"]:
        html(f'<div class="note">Rebalance day: the result includes {b["cost"] * 1e4:.2f} bp of trading costs.</div>')
    kpi_row([
        (f"vs {S.base}", (f"{r - base_b['net']:+.2%}" if base_b else "–"),
         (f"{S.base} {base_b['net']:+.2%}" if base_b else "same portfolio")),
        ("Rank in history", f"{b['percentile']:.0%}", "of past days were worse"),
        ("Month to date", f"{b['mtd']:+.2%}", f"{'+' if b['mtd'] >= 0 else '−'}${abs(b['mtd']) * amount:,.0f}"),
        ("Year to date", f"{b['ytd']:+.2%}", f"{'+' if b['ytd'] >= 0 else '−'}${abs(b['ytd']) * amount:,.0f}"),
        ("Best ETF today", f"{b['asset_ret'].idxmax()} {b['asset_ret'].max():+.2%}", NAMES[b['asset_ret'].idxmax()]),
        ("Worst ETF today", f"{b['asset_ret'].idxmin()} {b['asset_ret'].min():+.2%}", NAMES[b['asset_ret'].idxmin()]),
    ])
    a, bb = st.columns([1.1, 1])
    with a:
        title("What drove the day", "contribution of each ETF = weight at the open × ETF move")
        cs = b["contrib"].sort_values()
        fig = go.Figure(go.Bar(x=cs.values, y=list(cs.index), orientation="h",
                               marker_color=[UP if v >= 0 else DOWN for v in cs.values],
                               text=[f"{v:+.2%}  ({'+' if v >= 0 else '−'}${abs(v) * amount:,.2f})" for v in cs.values],
                               textposition="outside", textfont=dict(color=INK),
                               hovertemplate="%{y}: %{x:+.3%}<extra></extra>"))
        base_layout(fig, 300, legend=False)
        m = max(abs(cs.min()), abs(cs.max()), 1e-4) * 1.9
        fig.update_xaxes(tickformat="+.2%", range=[-m, m])
        fig.update_layout(hovermode="closest", bargap=0.3)
        fig.add_vline(x=0, line=dict(color=INK3, width=1))
        show(fig, key="pnl_contrib")
    with bb:
        title("Breakdown", f"total {r:+.2%}")
        t = pd.DataFrame({"Asset": [NAMES[x] for x in b["weights"].index], "Weight": b["weights"].values,
                          "ETF move": b["asset_ret"].values, "Contribution": b["contrib"].values,
                          "P&L $": b["contrib"].values * amount}, index=b["weights"].index)
        t = t.sort_values("Weight", ascending=False)
        sty = t.style.format({"Weight": "{:.1%}", "ETF move": "{:+.2%}", "Contribution": "{:+.3%}", "P&L $": "{:+,.2f}"})
        sty = styler_map(sty, color_sign, subset=["ETF move", "Contribution", "P&L $"])
        table(sty)
        if b["rebalance_day"] and abs(b["cost"]) > 0:
            html(f'<div class="note">Trading costs: {-b["cost"]:+.3%} ({-b["cost"] * amount:+,.2f} $)</div>')
    title("Last 30 trading days", "selected day outlined in amber")
    wv = b["window"]
    fig = go.Figure(go.Bar(x=wv.index, y=wv.values, marker_color=[UP if v >= 0 else DOWN for v in wv.values],
                           marker_line_color=[AMBER if d == b["date"] else "rgba(0,0,0,0)" for d in wv.index],
                           marker_line_width=[2.5 if d == b["date"] else 0 for d in wv.index],
                           hovertemplate="%{x|%a %d %b %Y}: %{y:+.2%}<extra></extra>"))
    base_layout(fig, 250, legend=False, yfmt="+.1%")
    fig.update_layout(hovermode="closest", bargap=0.25)
    fig.update_xaxes(type="category", tickvals=list(wv.index[::3]),
                     ticktext=[d.strftime("%d %b") for d in wv.index[::3]])
    show(fig, key="pnl_strip")
    up_days = (wv > 0).sum()
    html(f'<div class="note">{up_days} of the last {len(wv)} trading days were positive. Prices are daily closes; '
         f'results are for the model portfolio net of costs. For your own account, multiply shares held by each '
         f'ETF\'s price change.</div>')

elif page == "STRS":
    html(f'<div class="kpi-cap">Does {ls.name} hold up when markets fall? Each period compares it with the three '
         f'recipes alone and with simple portfolios of the same seven ETFs. All figures net of costs.</div>')
    per = st.radio("Period", list(C.STRESS_PERIODS), index=0, horizontal=True, key="strs_per",
                   label_visibility="collapsed")
    cmp_ = C.period_compare(allres, ls.name, per)
    d0, d1 = C.period_dates(allres, ls.name, per)
    me, ew, mix = cmp_.loc[ls.name], cmp_.loc["1/N"], cmp_.loc["1/3 static mix"]
    saved = me["Return"] - ew["Return"]
    html(f'<div class="note">{per}: {C.STRESS_NOTES[per]} · {d0.strftime("%d %b %Y")} → {d1.strftime("%d %b %Y")}</div>')
    kpi_row([
        (ls.name, f"{me['Return']:+.1%}", f"rank {int(me['Rank'])} of {len(cmp_)}"),
        ("1/N same ETFs", f"{ew['Return']:+.1%}", "equal weight"),
        ("Cushion vs 1/N", f"{saved:+.1%}", "green = lost less" if saved > 0 else "lost more"),
        ("vs 1/3 static mix", f"{me['Return'] - mix['Return']:+.1%}", f"static mix {mix['Return']:+.1%}"),
        ("Worst drawdown", f"{me['Max drawdown']:.1%}", f"1/N {ew['Max drawdown']:.1%}"),
    ])
    a, b = st.columns([1.15, 1])
    with a:
        title("Return over the period", "selected strategy outlined in amber")
        rs = cmp_["Return"].sort_values()
        fig = go.Figure(go.Bar(x=rs.values, y=list(rs.index), orientation="h",
                               marker_color=[COL.get(n, INK2) for n in rs.index],
                               marker_line_color=[AMBER if n == ls.name else "rgba(0,0,0,0)" for n in rs.index],
                               marker_line_width=[2.5 if n == ls.name else 0 for n in rs.index],
                               text=[f"{v:+.1%}" for v in rs.values], textposition="outside",
                               textfont=dict(color=INK), hovertemplate="%{y}: %{x:+.2%}<extra></extra>"))
        base_layout(fig, 320, legend=False)
        lo, hi = min(rs.min(), 0), max(rs.max(), 0)
        pad = max(hi - lo, 0.01) * 0.22
        fig.update_xaxes(tickformat="+.0%", range=[lo - pad, hi + pad])
        fig.update_layout(hovermode="closest", bargap=0.3)
        fig.add_vline(x=0, line=dict(color=INK3, width=1))
        show(fig, key="strs_ret")
    with b:
        title("What the ETFs did", "the raw material every portfolio had to work with")
        mv = C.period_asset_moves(bt.returns, d0, d1).sort_values()
        fig = go.Figure(go.Bar(x=mv.values, y=list(mv.index), orientation="h",
                               marker_color=[UP if v >= 0 else DOWN for v in mv.values],
                               text=[f"{v:+.1%}" for v in mv.values], textposition="outside",
                               textfont=dict(color=INK), hovertemplate="%{y}: %{x:+.2%}<extra></extra>"))
        base_layout(fig, 320, legend=False)
        lo, hi = min(mv.min(), 0), max(mv.max(), 0)
        pad = max(hi - lo, 0.01) * 0.22
        fig.update_xaxes(tickformat="+.0%", range=[lo - pad, hi + pad])
        fig.update_layout(hovermode="closest", bargap=0.3)
        fig.add_vline(x=0, line=dict(color=INK3, width=1))
        show(fig, key="strs_etf")
    ppi = C.period_pi(res, allres, ls.name, per)
    if ppi is not None:
        title("Where the learner had placed its trust", "average weight on each recipe during the period")
        kpi_row([(n, f"{ppi[n]:.0%}", "of the portfolio") for n in ppi.index])
    title("Period detail")
    sty = cmp_.style.format({"Return": "{:+.1%}", "Max drawdown": "{:.1%}", "Vol (ann.)": "{:.1%}",
                             "Up days": "{:.0%}", "Rank": "{:d}"})
    table(styler_map(sty, color_sign, subset=["Return"]))

    title("Scorecard across all stress periods", f"{ls.name} against the simple alternatives")
    sc = C.stress_scorecard(allres, ls.name)
    pc = [c for c in sc.columns if c != "Rank of 7"]
    sty = sc.style.format({**{c: "{:+.1%}" for c in pc}, "Rank of 7": "{:d}"})
    table(styler_map(sty, color_sign, subset=["vs 1/N"]))
    wins = int((sc["vs 1/N"] > 0).sum())
    wins_mix = int((sc[ls.name] > sc["1/3 static mix"] + 0.0005).sum())

    title("Downside capture", f"all days since {OOS_START}, measured against 1/N of the same ETFs")
    cap = C.capture_ratios(allres, ls.name, "1/N", OOS_START)
    kpi_row([
        ("Down-day capture", f"{cap['down']:.0%}", f"of 1/N losses on its {cap['n_down']:,} down days"),
        ("Up-day capture", f"{cap['up']:.0%}", f"of 1/N gains on its {cap['n_up']:,} up days"),
        ("Worst 5% of days", f"{cap['worst5_strategy']:+.2%}", f"1/N averaged {cap['worst5_bench']:+.2%}"),
    ])
    html(f'<div class="note"><b>How to read this.</b> {ls.name} lost less than an equal-weight portfolio of the same '
         f'ETFs in {wins} of {len(sc)} stress periods, and beat the 1/3 static mix in {wins_mix} of {len(sc)}. '
         f'It takes {cap["down"]:.0%} of the bad days and {cap["up"]:.0%} of the good days: when those two numbers '
         f'are close, the protection comes from carrying less risk, not from avoiding losses selectively. '
         f'A cushion against 1/N shows the system is defensive; a cushion against the 1/3 static mix is what '
         f'would show the learning layer itself adds value.</div>')

elif page == "DATA":
    title("Price file")
    if have_data:
        raw = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        kpi_row([
            ("File", os.path.basename(DATA_PATH), DATA_DIR[-40:]),
            ("Rows", f"{len(raw):,}", "trading days"),
            ("First date", raw.index[0].strftime("%d-%b-%Y"), "raw history"),
            ("Last date", raw.index[-1].strftime("%d-%b-%Y"), "latest close"),
            ("Updated", datetime.fromtimestamp(mt).strftime("%d-%b %H:%M"), "file modified"),
        ])
    else:
        html('<div class="note">No data yet.</div>')
    a, b = st.columns(2)
    with a:
        title("Update from Yahoo Finance", "adjusted closes, 2004 → today")
        if st.button("UPDATE FROM YAHOO FINANCE  <GO>", key="upd"):
            bar = st.progress(0.0)
            try:
                fetch_yahoo_csv(DATA_PATH, progress=lambda f, t: bar.progress(f))
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Data updated.")
                time.sleep(0.6)
                rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Download failed: {exc}")
        html('<div class="note">Downloads DBC GLD IEF TLT SPY VEA IEMG, plus EEM / EFA (used to extend IEMG / VEA '
             'before their launch) and ^IRX (T-bill rate for excess returns).</div>')
    with b:
        title("Or load your own CSV", "Date index + one column per ticker")
        up = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
        if up is not None and st.button("USE THIS FILE", key="use_up"):
            with open(DATA_PATH, "wb") as f:
                f.write(up.getvalue())
            st.cache_data.clear()
            st.cache_resource.clear()
            rerun()
    if have_data:
        title("Exports")
        bt = get_bt(mt, es)
        Wd = get_experts(mt, es, xs)
        F = get_features(mt, es)
        res, _ = C.run_strategy(bt, Wd, ls, F)
        res = C.apply_risk_target(bt, res, rs)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("PRICES (CSV)", prices.to_csv().encode(), "apt_prices.csv", "text/csv")
        with c2:
            st.download_button(f"{ls.name} DAILY RETURNS", res["daily"].to_frame("return").to_csv().encode(),
                               "apt_strategy_returns.csv", "text/csv")
        with c3:
            st.download_button(f"{ls.name} WEIGHTS", res["weights"].to_csv().encode(), "apt_weights.csv", "text/csv")
        title("Latest rows")
        table(raw.tail(10).style.format("{:.2f}", na_rep="–"))

html(f'<div class="note" style="margin-top:18px">Research software, not investment advice. Past performance '
     f'does not predict future results. Settings: {PROFILE_TXT} · {ls.name} · γ={ls.gamma:g} ρ={ls.rho:g} α={ls.alpha:g} · '
     f'{S.rebal.lower()} rebalance · {es.tc_bps:g}bp · lag {es.lag}d · IS ends {SPLIT}</div>')
