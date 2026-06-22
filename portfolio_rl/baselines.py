"""
baselines.py — Four baseline portfolio strategies for comparison against the RL agent.

1. Equal-weight (rebalanced monthly)
2. Buy & Hold SPY
3. Markowitz Mean-Variance Optimization (MPT)
4. Momentum — always hold top-3 performers from last calendar month
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (
    INITIAL_CAPITAL, TRANSACTION_COST, ASSET_NAMES, N_ASSETS
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _portfolio_values(weights_series: pd.DataFrame, raw_returns: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame of daily weights (index=dates, columns=assets) and
    a DataFrame of daily log-returns with the same index, compute portfolio value.
    """
    port_log_rets = (weights_series * raw_returns).sum(axis=1)
    port_vals = INITIAL_CAPITAL * np.exp(port_log_rets.cumsum())
    return port_vals


# ─────────────────────────────────────────────────────────────────────────────
# 1. EQUAL WEIGHT (monthly rebalance)
# ─────────────────────────────────────────────────────────────────────────────

def equal_weight(raw_returns: pd.DataFrame) -> dict:
    """
    Rebalances to 1/N each month. No transaction cost applied here for simplicity
    (the RL env charges it; for baselines we track turnover separately).
    """
    n = N_ASSETS
    equal_w = np.ones(n) / n

    weights = pd.DataFrame(
        index=raw_returns.index,
        columns=ASSET_NAMES,
        dtype=float,
    )

    prev_weights = equal_w.copy()
    current_month = None

    for date in raw_returns.index:
        month = (date.year, date.month)
        if month != current_month:
            current_month = month
            weights.loc[date] = equal_w
            prev_weights = equal_w.copy()
        else:
            weights.loc[date] = prev_weights

    port_vals = _portfolio_values(weights, raw_returns)
    return {"name": "Equal-Weight", "weights": weights, "portfolio_values": port_vals}


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUY & HOLD SPY
# ─────────────────────────────────────────────────────────────────────────────

def buy_and_hold_spy(raw_returns: pd.DataFrame) -> dict:
    """100% allocation to SPY from day 1."""
    spy_idx = ASSET_NAMES.index("SPY")
    weights = pd.DataFrame(0.0, index=raw_returns.index, columns=ASSET_NAMES)
    weights.iloc[:, spy_idx] = 1.0
    port_vals = _portfolio_values(weights, raw_returns)
    return {"name": "Buy&Hold SPY", "weights": weights, "portfolio_values": port_vals}


# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKOWITZ MEAN-VARIANCE (MPT)
# ─────────────────────────────────────────────────────────────────────────────

def _markowitz_weights(returns_window: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
    """
    Minimize portfolio variance subject to:
      - sum(w) = 1
      - w >= 0  (no shorting)
    Uses scipy minimize with SLSQP.
    """
    n = returns_window.shape[1]
    cov = np.cov(returns_window.T) + 1e-6 * np.eye(n)
    mu  = returns_window.mean(axis=0)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_var = w @ cov @ w
        return -(port_ret / (np.sqrt(port_var) + 1e-8))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    try:
        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 500, "ftol": 1e-9})
        if result.success:
            return np.clip(result.x, 0, 1)
    except Exception:
        pass
    return w0  # fallback to equal-weight


def markowitz(raw_returns: pd.DataFrame, lookback: int = 60, rebalance_freq: str = "M") -> dict:
    """
    Monthly rebalance using Markowitz optimization on trailing 60-day window.
    """
    weights = pd.DataFrame(0.0, index=raw_returns.index, columns=ASSET_NAMES)
    current_w = np.ones(N_ASSETS) / N_ASSETS

    rebalance_dates = pd.date_range(raw_returns.index[0], raw_returns.index[-1], freq=rebalance_freq)

    for date in raw_returns.index:
        if date in rebalance_dates:
            iloc_pos = raw_returns.index.get_loc(date)
            start    = max(0, iloc_pos - lookback)
            window   = raw_returns.iloc[start : iloc_pos].values
            if len(window) >= 20:
                current_w = _markowitz_weights(window)
        weights.loc[date] = current_w

    port_vals = _portfolio_values(weights, raw_returns)
    return {"name": "Markowitz MPT", "weights": weights, "portfolio_values": port_vals}


# ─────────────────────────────────────────────────────────────────────────────
# 4. MOMENTUM (top-3 performers last month)
# ─────────────────────────────────────────────────────────────────────────────

def momentum(raw_returns: pd.DataFrame, top_k: int = 3) -> dict:
    """
    At the start of each month, select the top-k assets by cumulative return
    over the prior calendar month. Equal-weight those top-k.
    """
    weights = pd.DataFrame(0.0, index=raw_returns.index, columns=ASSET_NAMES)
    current_w = np.ones(N_ASSETS) / N_ASSETS

    # Group by month
    grouped = raw_returns.groupby([raw_returns.index.year, raw_returns.index.month])
    months  = sorted(grouped.groups.keys())

    # Build a map: month → top-k weights (computed from previous month)
    month_weights = {}
    for i, month in enumerate(months):
        if i == 0:
            month_weights[month] = current_w
            continue
        prev_month = months[i - 1]
        prev_rets  = grouped.get_group(prev_month)
        cum_rets   = prev_rets.sum(axis=0).values
        top_idx    = np.argsort(cum_rets)[-top_k:]
        w = np.zeros(N_ASSETS)
        w[top_idx] = 1.0 / top_k
        month_weights[month] = w

    for date in raw_returns.index:
        month_key = (date.year, date.month)
        if month_key in month_weights:
            weights.loc[date] = month_weights[month_key]

    port_vals = _portfolio_values(weights, raw_returns)
    return {"name": "Momentum Top-3", "weights": weights, "portfolio_values": port_vals}


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────

def run_all_baselines(raw_returns: pd.DataFrame) -> dict:
    """Returns dict of {name: result_dict} for all four baselines."""
    results = {}
    for fn in [equal_weight, buy_and_hold_spy, markowitz, momentum]:
        r = fn(raw_returns)
        results[r["name"]] = r
        print(f"[baselines] {r['name']} done — final val={r['portfolio_values'].iloc[-1]:.2f}")
    return results
