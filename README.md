# Reinforcement Learning Portfolio Optimizer
### Proximal Policy Optimization (PPO) with Transformer State Encoder (2018–2024)

An autonomous quantitative asset allocation system employing Proximal Policy Optimization (PPO) coupled with a multi-head Transformer state encoder. Evaluated across an 8-asset universe spanning Indian large-caps (NSE), domestic Gold ETFs, and US index ETFs, featuring transaction-cost-penalized reward modeling.

---

## Architecture Overview

```
State Tensor: [Batch, 20-Day Lookback Window, 8 Assets]
    │
    ▼
Transformer Encoder (2 Attention Heads, 2 Layers, d_model = 64)
    ├── Learnable Positional Encoding
    ├── Temporal Multi-Head Self-Attention
    └── Mean Pooling Projection ──► Context Vector (64-dim)
                                         │
                                         ▼ (Concatenate current portfolio weights, 8-dim)
                                 ┌──────────────┐
                                 │ Latent State │ (72-dim)
                                 └───────┬──────┘
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
             Actor Head (Policy)                     Critic Head (Value)
                     │                                       │
     Gaussian Parameterization (Mean/Std)             Scalar Value Estimate V(s)
                     │
            Softmax Normalization
                     │
            Target Allocation (w_t)
```

### Architectural Rationale
- **Transformer Encoder over MLP / GRU**: Historical return matrices exhibit non-Markovian temporal correlations. Self-attention weights dynamically reveal which historical market conditions (e.g., prior volatility spikes or macro shifts) exert the strongest influence on the current allocation decision.
- **PPO over Off-Policy Methods (DDPG/SAC)**: PPO's clipped surrogate objective mitigates destabilizing policy updates common in financial environments, guaranteeing monotonic policy improvement across volatile market regimes without replay buffer stale-gradient issues.

---

## Asset Universe and Data Split

The asset universe reflects a multi-currency, diversified portfolio:

| Ticker | Asset Name | Asset Class / Role |
|---|---|---|
| `RELIANCE.NS` | Reliance Industries | Indian Large-Cap Energy & Conglomerate |
| `TCS.NS` | Tata Consultancy Services | Indian Large-Cap Information Technology |
| `HDFCBANK.NS` | HDFC Bank | Indian Large-Cap Banking & Financials |
| `INFY.NS` | Infosys | Indian Mid/Large-Cap Technology |
| `GOLDBEES.NS` | Nippon India ETF Gold BeES | Domestic Gold / Inflation Hedge |
| `AAPL` | Apple Inc. | US Technology & Growth |
| `SPY` | SPDR S&P 500 ETF Trust | US Broad Equity Market Proxy |
| `GC=F` | Gold Futures (COMEX) | International Commodity Hedge |

**Data Partitioning**:
- **Training**: 2018-01-01 to 2022-12-31 (In-sample policy training)
- **Validation**: 2023-01-01 to 2023-12-31 (Hyperparameter calibration & early stopping)
- **Testing**: 2024-01-01 to Present (Out-of-sample evaluation)

---

## Reward Formulation

The step reward balances capital appreciation, downside risk, and turnover friction:

$$R_t = \ln\left(\frac{V_t}{V_{t-1}}\right) - \lambda \sigma_t - \delta \cdot \text{TC}_t$$

Where:
- $V_t$: Portfolio net asset value at time step $t$.
- $\sigma_t$: Rolling 5-day realized portfolio volatility ($\lambda = 0.10$).
- $\text{TC}_t$: Transaction cost penalty defined as $\text{TC}_t = 0.001 \times \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$ ($\delta = 1.0$), assuming a conservative 10 bps slippage and brokerage charge.

---

## Benchmark Comparison Strategies

The trained PPO agent is benchmarked against four classical and quantitative baselines:

1. **Equal-Weight (1/N)**: Uniform capital distribution rebalanced periodically.
2. **Buy & Hold (SPY)**: Passive index tracking without rebalancing.
3. **Markowitz Modern Portfolio Theory (MPT)**: Mean-variance quadratic optimization using rolling 60-day covariance estimates.
4. **Momentum Top-3**: Monthly rotation into the top three assets by trailing return.

---

## Performance Evaluation Metrics

- **Cumulative Return**: Total percentage capital growth over the test horizon.
- **Sharpe Ratio**: Risk-adjusted excess return over the risk-free rate ($r_f = 5.0\%$).
- **Maximum Drawdown (MDD)**: Peak-to-trough equity drop.
- **Calmar Ratio**: Annualized return divided by absolute maximum drawdown.
- **Turnover Rate**: Average daily L1 reallocation metric to measure execution friction.

---

## Repository Structure

```
.
├── portfolio_rl/
│   ├── config.py              # Hyperparameters, tickers, date splits
│   ├── data_loader.py         # Yahoo Finance pipeline & return processing
│   ├── environment.py         # Gymnasium PortfolioEnv environment
│   ├── model.py               # Transformer state encoder + ActorCritic heads
│   ├── agent.py               # PPO training loop with GAE buffer
│   ├── baselines.py           # Equal-Weight, MPT, Momentum implementations
│   ├── train.py               # CLI training and evaluation runner
│   └── evaluate.py            # Financial performance metric calculators
├── frontend/
│   ├── src/
│   │   ├── pages/             # Dashboard, Training, Evaluation, Weights views
│   │   ├── components/        # Metrics cards, allocation & return charts
│   │   └── api.js             # Axios client for FastAPI backend
│   ├── package.json
│   └── vite.config.js
├── api.py                     # FastAPI REST server bridging model and UI
├── requirements.txt
└── README.md
```

---

## Setup and Quickstart

### 1. Environment Installation
```bash
# Clone the repository
git clone https://github.com/arjun-57/Finance-Portfolio-Management-Reinforcement-Learning-.git
cd Finance-Portfolio-Management-Reinforcement-Learning-

# Install Python requirements
pip install -r requirements.txt
```

### 2. Training the PPO Agent
```bash
# Execute PPO training with Transformer encoder
python portfolio_rl/train.py --timesteps 500000 --device cpu

# Evaluate out-of-sample test performance
python portfolio_rl/train.py --eval-only
```

### 3. Launching the Web Interface
```bash
# Terminal 1: Start FastAPI Backend (http://localhost:8000)
python api.py

# Terminal 2: Start Vite Frontend (http://localhost:5173)
cd frontend
npm install
npm run dev
```

---

## Hyperparameter Configuration

| Hyperparameter | Value | Description |
|---|---|---|
| Historical Lookback Window | 20 trading days | Trailing window of log returns |
| Embedding Dimension ($d_{model}$) | 64 | Latent representation size |
| Multi-Head Attention Heads | 2 | Parallel temporal attention mechanisms |
| Transformer Layers | 2 | Depth of transformer encoder stack |
| PPO Clipping Factor ($\epsilon$) | 0.20 | Policy probability ratio clipping bound |
| GAE Parameter ($\lambda$) | 0.95 | Generalized Advantage Estimation horizon factor |
| Entropy Coefficient | 0.01 | Policy entropy regularization for exploration |
| Transaction Cost | 0.10% (10 bps) | Reallocation turnover penalty |
