# Adaptive Portfolio Optimization: Online Expert Aggregation

Combines Mean-Variance (Markowitz), Risk Parity (ERC) and Minimum-CVaR portfolios with online
learning (Bernstein Online Aggregation, Hedge/Fixed-Share, Online Newton Step, contextual gating).
Universe: DBC, GLD, IEF, TLT, SPY, VEA, IEMG.

## Interactive terminal (app)
Double-click `run_app.bat` (Windows), or:
```bash
pip install streamlit plotly
streamlit run app.py
```
It opens in your browser at http://localhost:8501. Pages (click, or type the code in the command box):

| Code | Page |
|---|---|
| MON | Market monitor: ticker tiles, relative performance, correlations |
| PORT | Strategy backtest: KPIs vs benchmark, growth of $1, drawdowns, comparison table |
| ALLOC | Recommended allocation today, expert mix, risk contributions, order ticket (CSV) |
| EXPT | Expert analytics: weights through time, learnability, rolling Sharpe |
| LRN | Learner diagnostics: π over time, utility vs static mix, learning rates, conditional weights |
| RISK | VaR / CVaR, rolling vol, return distribution, monthly heatmap, worst drawdowns |
| STAT | HAC Sharpe tests, Diebold-Mariano, in-sample tuning grid, PSR / Deflated Sharpe |
| DATA | Update prices from Yahoo Finance, load your own CSV, exports |

All settings (learner, γ, ρ, α, rebalance frequency, costs, lag, in-sample split, expert parameters)
are in the left panel and re-run the backtest live. Expert optimisation is cached.

## Notebook
```bash
pip install numpy pandas scipy matplotlib jinja2 yfinance jupyter
jupyter notebook portfolio_optimization_w_backtesting.ipynb   # Kernel > Restart & Run All
```
Keep the notebook next to the `adaptive_portfolio/` folder. The full notebook runs in under a minute.

Configuration is in the first code cell (data source, IS/OOS split, rebalance frequency, costs,
risk aversion). If the download fails, the notebook falls back to a synthetic market and says so
loudly. Results from that market are for testing only. Other data options:
`DATA_SOURCE="stooq"`, or `"csv"` with `CSV_PATH` set (date index, one column per ticker).

## Layout
| File | Contents |
|---|---|
| `adaptive_portfolio/data.py` | yfinance/stooq/CSV loaders, EEM→IEMG and EFA→VEA back-fill, T-bill risk-free rate, regime-switching simulator |
| `adaptive_portfolio/estimators.py` | Ledoit-Wolf constant-correlation shrinkage, Bayes-Stein means, EWMA covariance |
| `adaptive_portfolio/experts.py` | MVO, ERC risk parity (coordinate descent), Min-CVaR (Rockafellar-Uryasev LP), benchmarks |
| `adaptive_portfolio/meta.py` | StaticMix, FollowTheLeader, Hedge/Fixed-Share, AdaptiveBOA, ONS, ContextualGate, Hierarchical |
| `adaptive_portfolio/features.py` | Past-only standardised market-state features for the contextual gate |
| `adaptive_portfolio/backtest.py` | Walk-forward engine (lag, drift, costs), hindsight-oracle mix, in-sample grid search |
| `adaptive_portfolio/metrics.py` | CAGR, vol, Sharpe, Sortino, MaxDD, Calmar, CVaR, turnover |
| `adaptive_portfolio/stats_tests.py` | Ledoit-Wolf HAC Sharpe test, block bootstrap, Diebold-Mariano, PSR, DSR, min track record |
| `build_notebook.py` | Regenerates the notebook |

## Using the learners on your own expert returns
```python
from adaptive_portfolio.meta import AdaptiveBOA
L = AdaptiveBOA(gamma=5, rho=0.99, alpha=0.01); L.reset(K=3)
for R_t in expert_period_returns:      # shape (K,)
    pi_t = L.weights()                 # allocate before seeing R_t
    L.update(R_t)
```
