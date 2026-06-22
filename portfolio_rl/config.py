"""
config.py — Central configuration for all hyperparameters, paths, and asset settings.
"""

# ─────────────────────────────────────────────────────────────────────────────
# ASSETS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = [
    "RELIANCE.NS",   # Indian large-cap (energy)
    "TCS.NS",        # Indian large-cap (IT)
    "HDFCBANK.NS",   # Indian large-cap (banking)
    "INFY.NS",       # Indian mid-large (IT)
    "GOLDBEES.NS",   # Indian ETF proxy for Nifty50
    "AAPL",          # US tech (global exposure)
    "SPY",           # US market ETF
    "GC=F",          # Gold (commodity hedge)
]
N_ASSETS = len(TICKERS)
ASSET_NAMES = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "GOLDBEES", "AAPL", "SPY", "GOLD"]

# ─────────────────────────────────────────────────────────────────────────────
# DATE RANGES
# ─────────────────────────────────────────────────────────────────────────────
START_DATE      = "2018-01-01"
END_DATE        = "2024-12-31"
TRAIN_END       = "2022-12-31"
VAL_START       = "2023-01-01"
VAL_END         = "2023-12-31"
TEST_START      = "2024-01-01"
TEST_END        = "2024-12-31"

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE         = 20       # days of history fed into Transformer
TRANSACTION_COST    = 0.001    # 0.1% per trade
INITIAL_CAPITAL     = 100_000  # USD / INR — units don't matter, only ratios do
LAMBDA_VOL          = 0.1      # penalty weight for rolling volatility
DELTA_TC            = 1.0      # penalty weight for transaction costs
VOL_WINDOW          = 5        # days for rolling volatility calculation

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER ENCODER
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL         = 64           # embedding dimension
N_HEADS         = 2            # attention heads
N_LAYERS        = 2            # transformer encoder layers
D_FF            = 128          # feedforward hidden dim
DROPOUT         = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# PPO HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
LEARNING_RATE   = 3e-4
GAMMA           = 0.99         # discount factor
GAE_LAMBDA      = 0.95         # GAE lambda
CLIP_EPS        = 0.2          # PPO clip epsilon
VALUE_COEF      = 0.5          # value loss coefficient
ENTROPY_COEF    = 0.01         # entropy bonus coefficient
MAX_GRAD_NORM   = 0.5          # gradient clipping
PPO_EPOCHS      = 4            # update epochs per rollout
MINI_BATCH_SIZE = 64
ROLLOUT_STEPS   = 256          # steps per rollout collection
TOTAL_TIMESTEPS = 500_000      # total training steps
SAVE_EVERY      = 10           # save checkpoint every N episodes

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RUNS_DIR        = os.path.join(BASE_DIR, "runs")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

RAW_PRICES_PATH     = os.path.join(DATA_DIR, "raw_prices.csv")
PROCESSED_PATH      = os.path.join(DATA_DIR, "processed.pkl")
REWARD_LOG_PATH     = os.path.join(RESULTS_DIR, "reward_log.csv")
METRICS_PATH        = os.path.join(RESULTS_DIR, "metrics.json")
PORTFOLIO_VALUES_PATH = os.path.join(RESULTS_DIR, "portfolio_values.csv")
WEIGHTS_HISTORY_PATH  = os.path.join(RESULTS_DIR, "weights_history.csv")

for _dir in [DATA_DIR, RUNS_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
