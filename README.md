# RL Portfolio Optimizer
### PPO + Transformer Encoder · Indian & Global Markets · 2018–2024

> A reinforcement learning portfolio management system using Proximal Policy Optimization (PPO) with a Transformer-based state encoder. Trained on 8 assets spanning Indian large-caps, gold, and US ETFs with a novel reward signal that includes transaction cost penalties.

---

## 🏗️ Architecture

```
State (20-day window, 8 assets)
    ↓
TransformerEncoder (2 heads × 2 layers, d_model=64)
    ├── Positional Encoding
    ├── Self-attention → temporal dependencies
    └── Mean-pool → context vector (64-dim)
    
         ↓ concat current weights (8-dim)
    ┌──────────────┐
    │ Actor Head   │ → Gaussian policy → Softmax → weights
    │ Critic Head  │ → Value estimate
    └──────────────┘
         ↓
PPO Update (clipped objective, GAE, entropy bonus)
```

### Why Transformer over MLP?
The 20-day return matrix is inherently sequential. A Transformer's self-attention learns *which days mattered most* for each allocation decision — attention weights are interpretable and examinable. This is novel at course-project level.

### Why PPO over DDPG?
PPO's clipped surrogate objective prevents destructive policy updates — it's stable, reproducible, and used in production by OpenAI and DeepMind. DDPG is notoriously brittle in portfolio problems due to its off-policy nature and dependency on a perfect replay buffer.

---

## 📦 Asset Universe

| Ticker        | Name          | Category               |
|---------------|---------------|------------------------|
| RELIANCE.NS   | Reliance      | Indian Large-Cap Energy|
| TCS.NS        | TCS           | Indian Large-Cap IT    |
| HDFCBANK.NS   | HDFC Bank     | Indian Large-Cap Bank  |
| INFY.NS       | Infosys       | Indian Mid-Large IT    |
| GOLDBEES.NS   | GoldBees ETF  | Indian Nifty50 Proxy   |
| AAPL          | Apple         | US Tech                |
| SPY           | S&P 500 ETF   | US Market              |
| GC=F          | Gold Futures  | Commodity Hedge        |

**Time split:** Train 2018–2022 · Val 2023 · Test 2024

---

## 🧮 Reward Signal

```
R_t = log(V_t / V_{t-1}) − λ·σ_t − δ·TC_t

where:
  σ_t  = rolling 5-day portfolio volatility   (λ = 0.1)
  TC_t = 0.1% × L1 turnover                  (δ = 1.0)
```

Transaction costs are what most student projects skip — including them makes the simulation realistic and prevents the agent from over-trading.

---

## 📈 Baselines

| Strategy     | Description                                              |
|--------------|----------------------------------------------------------|
| Equal-Weight | 1/N allocation, rebalanced monthly                       |
| Buy&Hold SPY | 100% S&P 500 ETF, no rebalancing                        |
| Markowitz MPT| Monthly optimization on trailing 60-day covariance       |
| Momentum Top-3 | Equal-weight the top-3 assets by prior month return   |

---

## 📊 Evaluation Metrics

| Metric           | Description                                        |
|------------------|----------------------------------------------------|
| Cumulative Return| Total % gain over test period                     |
| Sharpe Ratio     | Annualised risk-adjusted return (rf = 5%)         |
| Max Drawdown     | Largest peak-to-trough decline                    |
| Calmar Ratio     | Ann. Return / |Max Drawdown| — hedge fund standard|
| Turnover         | Mean daily L1 weight change                       |

---

## 🚀 Setup & Run

### 1. Install Python dependencies

```bash
cd portfolio
pip install -r requirements.txt
```

### 2. Start the FastAPI backend

```bash
python api.py
# Server starts at http://localhost:8000
```

### 3. Install and start the React frontend

```bash
cd frontend
npm install
npm run dev
# Frontend at http://localhost:5173
```

### 4. Train (via UI or CLI)

**Via UI:** Open `http://localhost:5173` → Train tab → Start Training

**Via CLI:**
```bash
cd portfolio_rl
python train.py --timesteps 500000 --device cpu
```

### 5. Evaluate

**Via UI:** Evaluate tab → Run Evaluation

**Via CLI:**
```bash
python train.py --eval-only
```

---

## 📂 File Structure

```
portfolio/
├── portfolio_rl/
│   ├── config.py         # All hyperparameters, paths, tickers
│   ├── data_loader.py    # yfinance download, log-returns, train/val/test
│   ├── environment.py    # PortfolioEnv (Gymnasium), step/reset/reward
│   ├── model.py          # TransformerEncoder + ActorCritic
│   ├── agent.py          # PPO loop, GAE, rollout buffer
│   ├── baselines.py      # Equal-weight, B&H, Markowitz, Momentum
│   ├── train.py          # CLI training entry point
│   └── evaluate.py       # Metrics: Sharpe, drawdown, Calmar, turnover
├── api.py                # FastAPI server — bridges Python ↔ React
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Router + sidebar layout
│   │   ├── api.js                 # Axios wrappers
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # KPI cards + returns chart + table
│   │   │   ├── Train.jsx          # Start/stop + live reward curve
│   │   │   ├── Evaluate.jsx       # Metrics comparison
│   │   │   └── WeightsView.jsx    # Stacked allocation chart
│   │   ├── components/
│   │   │   ├── MetricCard.jsx
│   │   │   ├── ReturnsChart.jsx
│   │   │   ├── WeightsChart.jsx
│   │   │   ├── RewardChart.jsx
│   │   │   ├── MetricsTable.jsx
│   │   │   └── StatusBadge.jsx
│   │   └── hooks/
│   │       ├── useMetrics.js
│   │       ├── usePortfolioValues.js
│   │       └── useTrainingStatus.js
│   ├── package.json
│   └── vite.config.js
├── requirements.txt
└── README.md
```

---

## 🔑 What Makes This Novel (Viva Points)

1. **Transformer state encoder** — treats the 20-day return history as a sequence, not a flat vector. Temporal attention weights reveal *which historical days* most influenced the allocation decision.

2. **Indian + global asset mix** — no existing open-source RL portfolio repo combines NSE stocks with SPY and Gold in this way.

3. **Transaction cost in reward** — most implementations skip TC; our simulation is robust to over-trading.

4. **Calmar ratio as primary metric** — measures risk-adjusted return the way hedge funds do, not just Sharpe.

5. **4 rigorous baselines** — including Momentum, which most papers omit.

---

## ⚙️ Key Hyperparameters

| Parameter        | Value   | Rationale                                    |
|------------------|---------|----------------------------------------------|
| Window size      | 20 days | ~1 trading month of history                  |
| d_model          | 64      | Compact encoder for fast convergence         |
| Attention heads  | 2       | Captures 2 independent temporal patterns     |
| PPO clip ε       | 0.2     | Standard; prevents large policy jumps        |
| GAE λ            | 0.95    | High — trusts long-horizon advantage signals |
| Entropy coef     | 0.01    | Light exploration bonus                      |
| Total timesteps  | 500k    | Sufficient for convergence                   |
| Transaction cost | 0.1%   | Realistic NSE/NYSE estimate                  |
