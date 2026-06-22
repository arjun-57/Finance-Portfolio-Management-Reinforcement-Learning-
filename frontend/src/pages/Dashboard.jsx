import { useEffect, useState } from 'react'
import MetricCard from '../components/MetricCard'
import ReturnsChart from '../components/ReturnsChart'
import MetricsTable from '../components/MetricsTable'
import { useMetrics } from '../hooks/useMetrics'
import { usePortfolioValues } from '../hooks/usePortfolioValues'

export default function Dashboard() {
  const { metrics, loading: mLoad, error: mErr } = useMetrics()
  const { data: pvData, loading: pvLoad, error: pvErr } = usePortfolioValues()

  const agent = metrics?.['PPO Agent']

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Portfolio Dashboard</h1>
        <p className="text-slate-400 text-sm mt-1">
          Test period: 2024 — PPO Transformer agent vs 4 baselines
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {mLoad ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="glass p-5 animate-pulse h-24 bg-surface-700/50" />
          ))
        ) : mErr ? (
          <div className="col-span-4 glass p-4 text-red-400 text-sm">
            ⚠️ Metrics not available — train and evaluate first.
          </div>
        ) : (
          <>
            <MetricCard
              label="Cumulative Return"
              value={agent ? `${agent.cumulative_return?.toFixed(2)}` : '—'}
              unit="%"
              icon="📈"
              color="brand"
            />
            <MetricCard
              label="Sharpe Ratio"
              value={agent ? agent.sharpe_ratio?.toFixed(3) : '—'}
              icon="⚡"
              color="green"
            />
            <MetricCard
              label="Max Drawdown"
              value={agent ? `${agent.max_drawdown?.toFixed(2)}` : '—'}
              unit="%"
              icon="📉"
              color="red"
            />
            <MetricCard
              label="Calmar Ratio"
              value={agent ? agent.calmar_ratio?.toFixed(3) : '—'}
              icon="🏆"
              color="amber"
            />
          </>
        )}
      </div>

      {/* Cumulative Returns Chart */}
      <div className="glass p-6">
        <h2 className="text-base font-semibold text-slate-200 mb-4">
          Cumulative Returns — All Strategies (Normalised to 100)
        </h2>
        {pvLoad ? (
          <div className="flex items-center justify-center h-64">
            <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <ReturnsChart data={pvData} />
        )}
      </div>

      {/* Metrics Table */}
      <div className="glass p-6">
        <h2 className="text-base font-semibold text-slate-200 mb-4">Strategy Comparison</h2>
        {mLoad ? (
          <div className="animate-pulse space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-10 bg-surface-700 rounded" />
            ))}
          </div>
        ) : (
          <MetricsTable metrics={metrics} />
        )}
      </div>

      {/* Architecture info */}
      <div className="glass p-6">
        <h2 className="text-base font-semibold text-slate-200 mb-3">Model Architecture</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="glass-sm p-4">
            <p className="text-indigo-400 font-semibold mb-2">🤖 Transformer Encoder</p>
            <ul className="text-slate-400 space-y-1 text-xs">
              <li>• 20-day sliding window</li>
              <li>• 2 attention heads × 2 layers</li>
              <li>• d_model = 64, d_ff = 128</li>
              <li>• Sinusoidal positional encoding</li>
            </ul>
          </div>
          <div className="glass-sm p-4">
            <p className="text-emerald-400 font-semibold mb-2">🎯 PPO Policy</p>
            <ul className="text-slate-400 space-y-1 text-xs">
              <li>• Clipped objective ε = 0.2</li>
              <li>• GAE λ = 0.95</li>
              <li>• Entropy bonus = 0.01</li>
              <li>• Gradient clipping = 0.5</li>
            </ul>
          </div>
          <div className="glass-sm p-4">
            <p className="text-amber-400 font-semibold mb-2">💰 Reward Signal</p>
            <ul className="text-slate-400 space-y-1 text-xs">
              <li>• Log portfolio return</li>
              <li>• − 0.1 × rolling 5-day vol</li>
              <li>• − 1.0 × 0.1% × turnover</li>
              <li>• Softmax action → no shorting</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
