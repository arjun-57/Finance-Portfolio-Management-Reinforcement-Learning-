"""
data_loader.py — Downloads stock data via yfinance, computes log-returns,
and splits into train / validation / test sets.
"""

import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    TICKERS, ASSET_NAMES, START_DATE, END_DATE,
    TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    RAW_PRICES_PATH, PROCESSED_PATH, WINDOW_SIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_prices(force: bool = False) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    if os.path.exists(RAW_PRICES_PATH) and not force:
        print(f"[data_loader] Loading cached prices from {RAW_PRICES_PATH}")
        return pd.read_csv(RAW_PRICES_PATH, index_col=0, parse_dates=True)

    print(f"[data_loader] Downloading data for {TICKERS} from {START_DATE} to {END_DATE}")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=True)

    # Extract 'Close' prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = TICKERS

    prices.columns = ASSET_NAMES  # rename for readability

    # Forward-fill then back-fill missing values (NSE holidays vs NYSE holidays)
    prices = prices.ffill().bfill()

    # Drop any remaining NaN rows
    prices.dropna(inplace=True)

    os.makedirs(os.path.dirname(RAW_PRICES_PATH), exist_ok=True)
    prices.to_csv(RAW_PRICES_PATH)
    print(f"[data_loader] Saved raw prices → {RAW_PRICES_PATH}  shape={prices.shape}")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns: ln(P_t / P_{t-1})."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret


def normalise(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-std normalisation per asset column."""
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + eps
    return (arr - mean) / std


def process_data(prices: pd.DataFrame, force: bool = False) -> dict:
    """
    Build the processed data dict with:
      - log_returns (DataFrame)
      - normalised_returns (np.ndarray)
      - train / val / test slices (normalised arrays + date indices)
    """
    if os.path.exists(PROCESSED_PATH) and not force:
        print(f"[data_loader] Loading cached processed data from {PROCESSED_PATH}")
        with open(PROCESSED_PATH, "rb") as f:
            return pickle.load(f)

    log_returns = compute_log_returns(prices)

    # Global normalisation stats (fit on train only to avoid lookahead bias)
    train_mask   = log_returns.index <= TRAIN_END
    train_arr    = log_returns[train_mask].values
    global_mean  = train_arr.mean(axis=0, keepdims=True)
    global_std   = train_arr.std(axis=0, keepdims=True) + 1e-8
    norm_returns = (log_returns.values - global_mean) / global_std

    norm_df = pd.DataFrame(norm_returns, index=log_returns.index, columns=ASSET_NAMES)

    splits = {
        "train": norm_df[norm_df.index <= TRAIN_END],
        "val":   norm_df[(norm_df.index >= VAL_START) & (norm_df.index <= VAL_END)],
        "test":  norm_df[norm_df.index >= TEST_START],
    }
    raw_splits = {
        "train": log_returns[log_returns.index <= TRAIN_END],
        "val":   log_returns[(log_returns.index >= VAL_START) & (log_returns.index <= VAL_END)],
        "test":  log_returns[log_returns.index >= TEST_START],
    }

    data = {
        "prices":       prices,
        "log_returns":  log_returns,
        "norm_returns": norm_df,
        "splits":       splits,       # normalised — for state observations
        "raw_splits":   raw_splits,   # actual log returns — for reward calculation
        "global_mean":  global_mean,
        "global_std":   global_std,
        "asset_names":  ASSET_NAMES,
        "window_size":  WINDOW_SIZE,
    }

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"[data_loader] Saved processed data → {PROCESSED_PATH}")

    for split_name, df in splits.items():
        print(f"  {split_name}: {df.index[0].date()} → {df.index[-1].date()}  rows={len(df)}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_data(force_download: bool = False, force_process: bool = False) -> dict:
    """Full pipeline: download → process → return data dict."""
    prices = download_prices(force=force_download)
    data   = process_data(prices, force=force_process)
    return data


if __name__ == "__main__":
    data = load_data()
    print("\nSplit shapes:")
    for k, df in data["splits"].items():
        print(f"  {k}: {df.shape}")
