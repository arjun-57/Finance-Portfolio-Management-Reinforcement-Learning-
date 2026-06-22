import { useEffect, useState } from 'react'
import WeightsChart from '../components/WeightsChart'
import { api } from '../api'

const ASSET_NAMES = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'GOLDBEES', 'AAPL', 'SPY', 'GOLD']
const ASSET_COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#06b6d4',
  '#f97316', '#ec4899', '#a3e635', '#8b5cf6',
]
const ASSET_DESCRIPTIONS = {
  RELIANCE:  'Indian Large-Cap — Energy & Petrochemicals',
  TCS:       'Indian Large-Cap — IT Services',
  HDFCBANK:  'Indian Large-Cap — Private Banking',
  INFY:      'Indian Mid-Large — IT Outsourcing',
  GOLDBEES:  'Indian ETF — Nifty 50 Proxy',
  AAPL:      'US Tech — Global Exposure',
  SPY:       'US Market ETF — S&P 500',
  GOLD:      'Commodity — Inflation & Crisis Hedge',
}

export default function WeightsView() {
  const [weights, setWeights] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [avgWeights, setAvgWeights] = useState(null)

  useEffect(() => {
    api.getWeights()
      .then(data => {
        // Add index column for chart
        const indexed = data.map((row, i) => ({ ...row, index: i }))
        setWeights(indexed)

        // Compute average allocation
        const avgs = {}
        ASSET_NAMES.forEach(name => {
          const vals = data.map(r => Number(r[name] ?? 0))
          avgs[name] = vals.reduce((a, b) => a + b, 0) / vals.length
        })
        setAvgWeights(avgs)
        setError(null)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Asset Allocation</h1>
        <p className="text-slate-400 text-sm mt-1">
          Agent's daily portfolio weights across 8 assets — test period 2024
        </p>
      </div>

      {/* Stacked area chart */}
      <div className="glass p-6">
        <h2 className="text-base font-semibold text-slate-200 mb-4">
          Daily Allocation Over Test Period
        </h2>
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : error ? (
          <p className="text-slate-500 text-sm p-4">
            No weight data yet. Run evaluation first.
          </p>
        ) : (
          <WeightsChart data={weights} />
        )}
      </div>

      {/* Average allocation breakdown */}
      {avgWeights && (
        <div className="glass p-6">
          <h2 className="text-base font-semibold text-slate-200 mb-4">
            Mean Allocation (2024 Average)
          </h2>
          <div className="space-y-3">
            {ASSET_NAMES
              .map((name, i) => ({ name, color: ASSET_COLORS[i], avg: avgWeights[name] ?? 0 }))
              .sort((a, b) => b.avg - a.avg)
              .map(({ name, color, avg }) => (
                <div key={name} className="flex items-center gap-3">
                  <div className="w-20 text-right text-xs font-semibold" style={{ color }}>{name}</div>
                  <div className="flex-1 bg-surface-700 rounded-full h-2.5 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${(avg * 100).toFixed(1)}%`, background: color, opacity: 0.85 }}
                    />
                  </div>
                  <div className="w-12 text-xs text-slate-400 tabular-nums">
                    {(avg * 100).toFixed(1)}%
                  </div>
                  <div className="hidden md:block text-xs text-slate-500 w-56 truncate">
                    {ASSET_DESCRIPTIONS[name]}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Asset universe */}
      <div className="glass p-6">
        <h2 className="text-base font-semibold text-slate-200 mb-4">Asset Universe</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {ASSET_NAMES.map((name, i) => (
            <div key={name} className="glass-sm p-3 flex items-start gap-2.5">
              <div
                className="w-3 h-3 rounded-full mt-0.5 flex-shrink-0"
                style={{ background: ASSET_COLORS[i] }}
              />
              <div>
                <p className="text-xs font-semibold text-slate-200">{name}</p>
                <p className="text-xs text-slate-500 leading-snug mt-0.5">
                  {ASSET_DESCRIPTIONS[name]}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
