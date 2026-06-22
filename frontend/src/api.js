import axios from 'axios'

const BASE = '/api'

export const api = {
  // ── Training ──────────────────────────────────────────────────────────────
  startTraining: (timesteps = 500000, device = 'cpu') =>
    axios.post(`${BASE}/train`, null, { params: { timesteps, device } }),

  stopTraining: () =>
    axios.post(`${BASE}/train/stop`),

  getTrainStatus: () =>
    axios.get(`${BASE}/train/status`).then(r => r.data),

  // ── Evaluation ────────────────────────────────────────────────────────────
  runEvaluation: () =>
    axios.get(`${BASE}/evaluate`).then(r => r.data),

  // ── Data ──────────────────────────────────────────────────────────────────
  getMetrics: () =>
    axios.get(`${BASE}/metrics`).then(r => r.data),

  getPortfolioValues: () =>
    axios.get(`${BASE}/portfolio-values`).then(r => r.data),

  getWeights: () =>
    axios.get(`${BASE}/weights`).then(r => r.data),

  getRewardLog: () =>
    axios.get(`${BASE}/reward-log`).then(r => r.data),

  health: () =>
    axios.get(`${BASE}/health`).then(r => r.data),
}
