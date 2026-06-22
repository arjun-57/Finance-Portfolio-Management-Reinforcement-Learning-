"""
environment.py — PortfolioEnv: a Gymnasium environment for portfolio allocation.

State:  sliding window of 20-day normalised log-returns (20×8) + current weights (8,)
Action: continuous 8-dim vector → Softmax → valid portfolio allocation
Reward: log portfolio return – λ·rolling_vol – δ·transaction_cost
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    N_ASSETS, WINDOW_SIZE, INITIAL_CAPITAL,
    TRANSACTION_COST, LAMBDA_VOL, DELTA_TC, VOL_WINDOW,
)


class PortfolioEnv(gym.Env):
    """
    Custom Gymnasium environment for portfolio management.

    Parameters
    ----------
    returns : np.ndarray
        Normalised log-return matrix, shape (T, N_ASSETS).
    raw_returns : np.ndarray
        Actual log-return matrix (for reward computation), shape (T, N_ASSETS).
    mode : str
        One of 'train', 'val', 'test' — just for logging purposes.
    """

    metadata = {"render_modes": []}

    def __init__(self, returns: np.ndarray, raw_returns: np.ndarray, mode: str = "train"):
        super().__init__()
        self.returns     = returns        # normalised, (T, N)
        self.raw_returns = raw_returns    # actual log-returns, (T, N)
        self.mode        = mode
        self.n_assets    = N_ASSETS
        self.window      = WINDOW_SIZE

        # observation: (window, n_assets) flattened + current weights
        obs_dim = self.window * self.n_assets + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # action: raw logits → softmax inside step
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self._reset_state()

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.t             = self.window          # start after first window
        self.portfolio_val = float(INITIAL_CAPITAL)
        self.weights       = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.prev_weights  = self.weights.copy()
        self.done          = False
        self.portfolio_values = [self.portfolio_val]
        self.weights_history  = [self.weights.copy()]
        self.log_return_history = []

    def _get_obs(self) -> np.ndarray:
        window_returns = self.returns[self.t - self.window : self.t]  # (window, N)
        obs = np.concatenate([window_returns.flatten(), self.weights])
        return obs.astype(np.float32)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # ──────────────────────────────────────────────────────────────────────────
    # GYM INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        # ── 1. Convert action to valid portfolio weights ──────────────────────
        new_weights = self.softmax(np.array(action, dtype=np.float64))
        self.prev_weights = self.weights.copy()
        self.weights = new_weights.astype(np.float32)

        # ── 2. Compute portfolio log-return ───────────────────────────────────
        day_raw    = self.raw_returns[self.t]          # actual log-returns today
        port_ret   = float(np.dot(self.weights, day_raw))

        # ── 3. Transaction costs (L1 turnover × TC rate) ──────────────────────
        turnover   = float(np.sum(np.abs(self.weights - self.prev_weights)))
        tc_cost    = TRANSACTION_COST * turnover

        # ── 4. Rolling portfolio volatility (last VOL_WINDOW steps) ──────────
        self.log_return_history.append(port_ret)
        if len(self.log_return_history) >= VOL_WINDOW:
            recent_rets = self.log_return_history[-VOL_WINDOW:]
            sigma_t     = float(np.std(recent_rets))
        else:
            sigma_t = 0.0

        # ── 5. Reward ─────────────────────────────────────────────────────────
        reward = port_ret - LAMBDA_VOL * sigma_t - DELTA_TC * tc_cost

        # ── 6. Update portfolio value ─────────────────────────────────────────
        self.portfolio_val *= np.exp(port_ret - DELTA_TC * tc_cost)
        self.portfolio_values.append(self.portfolio_val)
        self.weights_history.append(self.weights.copy())

        # ── 7. Step forward ───────────────────────────────────────────────────
        self.t += 1
        self.done = self.t >= len(self.returns)

        obs     = self._get_obs() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info    = {
            "portfolio_value": self.portfolio_val,
            "turnover":        turnover,
            "tc_cost":         tc_cost,
            "sigma_t":         sigma_t,
            "port_ret":        port_ret,
            "weights":         self.weights.tolist(),
        }
        return obs, float(reward), bool(self.done), False, info

    def render(self):
        print(f"[PortfolioEnv] t={self.t}  val={self.portfolio_val:.2f}  weights={np.round(self.weights, 3)}")
