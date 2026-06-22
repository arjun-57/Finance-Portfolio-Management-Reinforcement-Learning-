"""
model.py — Transformer-based Actor-Critic for PPO portfolio allocation.

Architecture:
    Input:  (batch, obs_dim)  where obs_dim = window*n_assets + n_assets
    ┌─ TransformerEncoder ─┐
    │  Reshape → (window, n_assets)
    │  Linear project → (window, d_model)
    │  + Positional Encoding
    │  N×TransformerEncoderLayer (n_heads, d_ff)
    │  Mean-pool over time-axis → (d_model,)
    │  Concat current weights   → (d_model + n_assets,)
    └──────────────────────┘
    Actor:  MLP → logits (n_assets,)       [raw; softmax applied in env]
    Critic: MLP → scalar value estimate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_ASSETS, WINDOW_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)              # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Encodes a (window, n_assets) return sequence into a fixed-size embedding.
    Returns a vector of size d_model + n_assets (encoded context + current weights).
    """

    def __init__(
        self,
        n_assets:    int   = N_ASSETS,
        window:      int   = WINDOW_SIZE,
        d_model:     int   = D_MODEL,
        n_heads:     int   = N_HEADS,
        n_layers:    int   = N_LAYERS,
        d_ff:        int   = D_FF,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.window   = window
        self.d_model  = d_model

        # Project n_assets → d_model
        self.input_proj = nn.Linear(n_assets, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=window + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,         # (batch, seq, d_model)
            norm_first=True,          # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch, window*n_assets + n_assets)
        returns: (batch, d_model + n_assets)
        """
        batch = obs.size(0)
        window_flat    = obs[:, : self.window * self.n_assets]   # (B, W*N)
        current_weights = obs[:, self.window * self.n_assets:]    # (B, N)

        # Reshape to sequence
        seq = window_flat.view(batch, self.window, self.n_assets)  # (B, W, N)

        # Project to d_model
        x = self.input_proj(seq)        # (B, W, d_model)
        x = self.pos_enc(x)             # (B, W, d_model)

        # Transformer
        x = self.transformer(x)         # (B, W, d_model)
        x = self.norm(x)

        # Mean pool over time dimension
        ctx = x.mean(dim=1)             # (B, d_model)

        # Concatenate current weights
        out = torch.cat([ctx, current_weights], dim=-1)  # (B, d_model + n_assets)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# ACTOR-CRITIC
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared Transformer encoder with separate Actor and Critic heads.

    Actor  → outputs mean of a Gaussian policy over action space
    Critic → outputs scalar state-value estimate
    """

    def __init__(
        self,
        n_assets: int   = N_ASSETS,
        window:   int   = WINDOW_SIZE,
        d_model:  int   = D_MODEL,
        n_heads:  int   = N_HEADS,
        n_layers: int   = N_LAYERS,
        d_ff:     int   = D_FF,
        dropout:  float = DROPOUT,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            n_assets=n_assets, window=window, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        )
        feat_dim = d_model + n_assets

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, n_assets),
        )
        # Log-std (learnable, per-action-dim)
        self.log_std = nn.Parameter(torch.zeros(n_assets))

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for actor output
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        """Returns (action_mean, value)."""
        features = self.encoder(obs)
        action_mean = self.actor(features)
        value       = self.critic(features).squeeze(-1)
        return action_mean, value

    def get_distribution(self, obs: torch.Tensor):
        """Returns a Gaussian distribution over actions."""
        action_mean, value = self(obs)
        std = self.log_std.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        return dist, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Returns log_probs, entropy, and values for given obs and actions."""
        dist, value = self.get_distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1)
        return log_probs, entropy, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action from policy. Returns (action, log_prob, value)."""
        dist, value = self.get_distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
