"""
train.py — Main entry point: loads data, builds envs, trains PPO agent.
Run: python train.py
"""

import os
import sys
import argparse

# Add parent directory to path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TOTAL_TIMESTEPS, RUNS_DIR, RESULTS_DIR
from data_loader import load_data
from environment import PortfolioEnv
from agent import PPOAgent
from baselines import run_all_baselines
from evaluate import run_full_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO portfolio agent")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total training timesteps")
    parser.add_argument("--device",    type=str, default="cpu",  help="Torch device: cpu or cuda")
    parser.add_argument("--force-download", action="store_true", help="Force re-download data from yfinance")
    parser.add_argument("--force-process",  action="store_true", help="Force re-process data")
    parser.add_argument("--status-file",    type=str, default=None, help="Path to JSON status file for API polling")
    parser.add_argument("--eval-only",      action="store_true",    help="Skip training, only evaluate last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  RL Portfolio Optimizer — PPO + Transformer")
    print("=" * 60)

    # ── 1. Load and process data ──────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    data = load_data(force_download=args.force_download, force_process=args.force_process)

    train_split = data["splits"]["train"]
    val_split   = data["splits"]["val"]
    test_split  = data["splits"]["test"]

    raw_train = data["raw_splits"]["train"]
    raw_val   = data["raw_splits"]["val"]
    raw_test  = data["raw_splits"]["test"]

    print(f"  Train: {train_split.shape}  Val: {val_split.shape}  Test: {test_split.shape}")

    # ── 2. Create environments ────────────────────────────────────────────────
    print("\n[2/4] Creating environments...")
    train_env = PortfolioEnv(train_split.values, raw_train.values, mode="train")
    val_env   = PortfolioEnv(val_split.values,   raw_val.values,   mode="val")
    test_env  = PortfolioEnv(test_split.values,  raw_test.values,  mode="test")

    # ── 3. Train (or load) agent ──────────────────────────────────────────────
    print("\n[3/4] Training PPO agent...")
    agent = PPOAgent(train_env, device=args.device)

    if args.eval_only:
        # Find the latest checkpoint
        import glob
        ckpts = sorted(glob.glob(os.path.join(RUNS_DIR, "checkpoint_*.pt")))
        if not ckpts:
            print("[ERROR] No checkpoints found. Run without --eval-only first.")
            sys.exit(1)
        latest = ckpts[-1]
        agent.load_checkpoint(latest)
        print(f"  Loaded checkpoint: {latest}")
    else:
        agent.train(status_file=args.status_file, total_timesteps=args.timesteps)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating all strategies on test period (2024)...")
    baseline_results = run_all_baselines(raw_test)
    all_metrics = run_full_evaluation(agent, test_env, baseline_results)

    print("\n✅ Done. Results written to results/")
    return all_metrics

if __name__ == "__main__":
    main()
