"""
evaluate.py — Compute evaluation metrics for all strategies and write results.

Metrics computed:
  - Cumulative return
  - Annualized Sharpe ratio
  - Maximum drawdown
  - Calmar ratio  (annualized return / max drawdown)
  - Portfolio turnover (agent only)
"""

import os
import json
import numpy as np
import pandas as pd

from config import (
    METRICS_PATH, PORTFOLIO_VALUES_PATH, WEIGHTS_HISTORY_PATH,
    RUNS_DIR, RESULTS_DIR, ASSET_NAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_return(port_vals: pd.Series) -> float:
    return float((port_vals.iloc[-1] / port_vals.iloc[0]) - 1.0)


def annualized_sharpe(port_vals: pd.Series, rf: float = 0.05, periods_per_year: int = 252) -> float:
    log_rets = np.log(port_vals / port_vals.shift(1)).dropna()
    excess   = log_rets - rf / periods_per_year
    if excess.std() < 1e-10:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))


def max_drawdown(port_vals: pd.Series) -> float:
    rolling_max = port_vals.cummax()
    drawdown    = (port_vals - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(port_vals: pd.Series, periods_per_year: int = 252) -> float:
    n_years    = len(port_vals) / periods_per_year
    ann_return = float((port_vals.iloc[-1] / port_vals.iloc[0]) ** (1 / max(n_years, 1e-6)) - 1)
    mdd        = abs(max_drawdown(port_vals))
    if mdd < 1e-10:
        return 0.0
    return float(ann_return / mdd)


def portfolio_turnover(weights_history: list) -> float:
    """Mean daily L1 turnover (0=no trading, 2=full portfolio rotation)."""
    if len(weights_history) < 2:
        return 0.0
    turnovers = []
    for i in range(1, len(weights_history)):
        turnovers.append(np.sum(np.abs(np.array(weights_history[i]) - np.array(weights_history[i - 1]))))
    return float(np.mean(turnovers))


def compute_metrics(port_vals: pd.Series, weights_hist: list = None) -> dict:
    metrics = {
        "cumulative_return": round(cumulative_return(port_vals) * 100, 4),
        "sharpe_ratio":      round(annualized_sharpe(port_vals), 4),
        "max_drawdown":      round(max_drawdown(port_vals) * 100, 4),
        "calmar_ratio":      round(calmar_ratio(port_vals), 4),
    }
    if weights_hist is not None:
        metrics["turnover"] = round(portfolio_turnover(weights_hist), 6)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(env, agent, label: str = "PPO Agent") -> dict:
    """
    Run agent on environment (deterministically) and collect portfolio values + weights.
    Returns metrics dict.
    """
    from model import ActorCritic
    import torch

    device = agent.device
    model  = agent.model
    model.eval()

    obs, _ = env.reset()
    done   = False
    with torch.no_grad():
        while not done:
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _ = model.act(obs_t, deterministic=True)
            action_np    = action.squeeze(0).cpu().numpy()
            obs, _, done, truncated, info = env.step(action_np)
            done = done or truncated

    port_vals = pd.Series(env.portfolio_values, name=label)
    wh        = env.weights_history
    metrics   = compute_metrics(port_vals, wh)
    return {"name": label, "metrics": metrics, "portfolio_values": port_vals, "weights_history": wh}


def run_full_evaluation(agent, test_env, baseline_results: dict):
    """
    Evaluate agent + all baselines on test period, write all result files.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Agent evaluation ──────────────────────────────────────────────────────
    agent_result = evaluate_agent(test_env, agent)
    print(f"[evaluate] Agent metrics: {agent_result['metrics']}")

    # ── Combine all portfolio values ──────────────────────────────────────────
    all_vals = {"PPO Agent": agent_result["portfolio_values"]}
    for name, res in baseline_results.items():
        all_vals[name] = res["portfolio_values"]

    # Align indices
    pv_df = pd.DataFrame(all_vals)
    pv_df.index = agent_result["portfolio_values"].index if hasattr(
        agent_result["portfolio_values"], "index"
    ) else range(len(agent_result["portfolio_values"]))
    pv_df.to_csv(PORTFOLIO_VALUES_PATH)
    print(f"[evaluate] Portfolio values → {PORTFOLIO_VALUES_PATH}")

    # ── Agent weights history ─────────────────────────────────────────────────
    wh_df = pd.DataFrame(agent_result["weights_history"], columns=ASSET_NAMES)
    wh_df.to_csv(WEIGHTS_HISTORY_PATH, index=False)
    print(f"[evaluate] Weights history → {WEIGHTS_HISTORY_PATH}")

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    all_metrics = {"PPO Agent": agent_result["metrics"]}
    for name, res in baseline_results.items():
        all_metrics[name] = compute_metrics(res["portfolio_values"])

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[evaluate] Metrics → {METRICS_PATH}")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Strategy':<22} {'CumRet%':>9} {'Sharpe':>8} {'MaxDD%':>9} {'Calmar':>8} {'Turnover':>10}")
    print("-" * 75)
    for strat, m in all_metrics.items():
        print(
            f"{strat:<22} "
            f"{m['cumulative_return']:>9.2f} "
            f"{m['sharpe_ratio']:>8.3f} "
            f"{m['max_drawdown']:>9.2f} "
            f"{m['calmar_ratio']:>8.3f} "
            f"{m.get('turnover', 0.0):>10.5f}"
        )
    print("=" * 75)

    return all_metrics
