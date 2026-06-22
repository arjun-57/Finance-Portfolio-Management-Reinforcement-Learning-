"""
agent.py — PPO training loop with Generalized Advantage Estimation (GAE).

Key design choices:
  - Clipped surrogate objective (prevents destructive policy updates)
  - GAE for low-variance advantage estimates
  - Entropy bonus to encourage exploration
  - Value function loss with optional clipping
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE, GAMMA, GAE_LAMBDA, CLIP_EPS,
    VALUE_COEF, ENTROPY_COEF, MAX_GRAD_NORM,
    PPO_EPOCHS, MINI_BATCH_SIZE, ROLLOUT_STEPS,
    TOTAL_TIMESTEPS, SAVE_EVERY, RUNS_DIR, REWARD_LOG_PATH,
)
from model import ActorCritic


# ─────────────────────────────────────────────────────────────────────────────
# ROLLOUT BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores transitions collected during environment rollout."""

    def __init__(self):
        self.obs       = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def size(self):
        return len(self.rewards)


# ─────────────────────────────────────────────────────────────────────────────
# PPO AGENT
# ─────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self, env, device: str = "cpu"):
        self.env    = env
        self.device = torch.device(device)

        self.model  = ActorCritic().to(self.device)
        self.optim  = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.buffer = RolloutBuffer()

        os.makedirs(RUNS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(REWARD_LOG_PATH), exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # GAE
    # ──────────────────────────────────────────────────────────────────────────

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val    = last_value
                next_not_done = 1.0 - float(dones[t])
            else:
                next_val    = values[t + 1]
                next_not_done = 1.0 - float(dones[t])
            delta    = rewards[t] + GAMMA * next_val * next_not_done - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * next_not_done * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ──────────────────────────────────────────────────────────────────────────
    # PPO UPDATE
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, obs_t, actions_t, old_log_probs_t, advantages_t, returns_t):
        """Run PPO_EPOCHS passes over the rollout buffer."""
        n = obs_t.size(0)
        total_loss_sum = 0.0

        for _ in range(PPO_EPOCHS):
            # Shuffle mini-batches
            indices = torch.randperm(n)
            for start in range(0, n, MINI_BATCH_SIZE):
                idx = indices[start : start + MINI_BATCH_SIZE]

                mb_obs        = obs_t[idx]
                mb_actions    = actions_t[idx]
                mb_old_lp     = old_log_probs_t[idx]
                mb_adv        = advantages_t[idx]
                mb_returns    = returns_t[idx]

                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                log_probs, entropy, values = self.model.evaluate_actions(mb_obs, mb_actions)

                # Policy ratio
                ratio = torch.exp(log_probs - mb_old_lp)

                # Clipped surrogate
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = VALUE_COEF * nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy_loss = -ENTROPY_COEF * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optim.step()

                total_loss_sum += loss.item()

        return total_loss_sum

    # ──────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, status_file: str = None, total_timesteps: int = TOTAL_TIMESTEPS):
        """
        Main training loop.
        Writes reward_log.csv after each episode.
        Writes checkpoint files every SAVE_EVERY episodes.
        If status_file is provided, writes JSON status for API polling.
        """
        log_path = REWARD_LOG_PATH
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "episode_reward", "portfolio_value", "time_elapsed"])

        obs, _   = self.env.reset()
        episode  = 0
        timestep = 0
        ep_reward = 0.0
        start_time = time.time()

        print(f"[PPO] Starting training for {total_timesteps} timesteps...")

        while timestep < total_timesteps:
            self.buffer.clear()

            # ── Collect rollout ───────────────────────────────────────────────
            for _ in range(ROLLOUT_STEPS):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, log_prob, value = self.model.act(obs_t)

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, done, truncated, info = self.env.step(action_np)

                self.buffer.add(
                    obs,
                    action_np,
                    log_prob.item(),
                    reward,
                    value.item(),
                    done or truncated,
                )

                obs        = next_obs
                ep_reward += reward
                timestep  += 1

                if done or truncated:
                    episode += 1
                    port_val = info.get("portfolio_value", 0.0)
                    elapsed  = time.time() - start_time

                    with open(log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([episode, timestep, ep_reward, port_val, elapsed])

                    print(
                        f"[PPO] ep={episode}  t={timestep}  "
                        f"reward={ep_reward:.4f}  val={port_val:.2f}  "
                        f"elapsed={elapsed:.1f}s"
                    )

                    if status_file:
                        import json
                        with open(status_file, "w") as sf:
                            json.dump({
                                "status": "training",
                                "episode": episode,
                                "timestep": timestep,
                                "latest_reward": ep_reward,
                                "portfolio_value": port_val,
                            }, sf)

                    if episode % SAVE_EVERY == 0:
                        ckpt_path = os.path.join(RUNS_DIR, f"checkpoint_ep{episode}.pt")
                        torch.save(self.model.state_dict(), ckpt_path)
                        print(f"[PPO] Saved checkpoint → {ckpt_path}")

                    obs, _    = self.env.reset()
                    ep_reward = 0.0

                    if timestep >= total_timesteps:
                        break

            # ── GAE and PPO update ────────────────────────────────────────────
            with torch.no_grad():
                last_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                _, last_val = self.model(last_obs_t)
                last_val   = last_val.item()

            advantages, returns = self.compute_gae(
                self.buffer.rewards,
                self.buffer.values,
                self.buffer.dones,
                last_val,
            )

            obs_t       = torch.FloatTensor(np.array(self.buffer.obs)).to(self.device)
            actions_t   = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
            old_lp_t    = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
            adv_t       = torch.FloatTensor(advantages).to(self.device)
            returns_t   = torch.FloatTensor(returns).to(self.device)

            self.update(obs_t, actions_t, old_lp_t, adv_t, returns_t)

        # ── Final checkpoint ──────────────────────────────────────────────────
        final_path = os.path.join(RUNS_DIR, "checkpoint_final.pt")
        torch.save(self.model.state_dict(), final_path)
        print(f"[PPO] Training complete. Final checkpoint → {final_path}")

        if status_file:
            import json
            with open(status_file, "w") as sf:
                json.dump({"status": "done", "episode": episode, "timestep": timestep}, sf)

        return self.model

    # ──────────────────────────────────────────────────────────────────────────
    # LOAD CHECKPOINT
    # ──────────────────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[PPO] Loaded checkpoint from {path}")
