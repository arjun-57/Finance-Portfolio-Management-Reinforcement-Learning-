import { useState } from 'react'
import MetricsTable from '../components/MetricsTable'
import MetricCard from '../components/MetricCard'
import { useMetrics } from '../hooks/useMetrics'
import { api } from '../api'

export default function Evaluate() {
  const { metrics, loading, error, refetch } = useMetrics()
  const [running, setRunning] = useState(false)
  const [log, setLog] = useState('')

  const agent = metrics?.['PPO Agent']

  const handleEvaluate = async () => {
    setRunning(true)
    setLog('')
    try {
      const res = await api.runEvaluation()
      setLog(res.stdout ?? 'Done.')
      refetch()
    } catch (e) {
      setLog(`Error: ${e.response?.data?.detail ?? e.message}`)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-white">Evaluate</h1>
          <p className="text-slate-400 text-sm mt-1">
            Run agent on 2024 test set and compare against all baselines
          </p>
        </div>
        <button
          className={`btn-glow ${running ? 'opacity-60 cursor-not-allowed' : ''}`}
          onClick={handleEvaluate}
          disabled={running}
        >
          {running ? (
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin inline-block" />
              Running…
            </span>
          ) : '⚡ Run Evaluation'}
        </button>
      </div>

      {/* Agent KPI summary */}
      {agent && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="Cumulative Return" value={`${agent.cumulative_return?.toFixed(2)}`} unit="%" icon="📈" color="brand" />
          <MetricCard label="Sharpe Ratio"       value={agent.sharpe_ratio?.toFixed(3)} icon="⚡" color="green" />
          <MetricCard label="Max Drawdown"        value={`${agent.max_drawdown?.toFixed(2)}`} unit="%" icon="📉" color="red" />
          <MetricCard label="Calmar Ratio"        value={agent.calmar_ratio?.toFixed(3)} icon="🏆" color="amber" />
        </div>
      )}

      {/* Full table */}
      <div className="glass p-6">
        <h2 className="text-base font-semibold text-slate-200 mb-4">
          Full Strategy Comparison — Test Period (2024)
        </h2>
        {loading ? (
          <div className="animate-pulse space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-10 bg-surface-700 rounded" />
            ))}
          </div>
        ) : error ? (
          <p className="text-slate-500 text-sm">
            No metrics found. Run evaluation above (requires a trained checkpoint in <code className="text-slate-300 bg-white/5 px-1 rounded">runs/</code>).
          </p>
        ) : (
          <MetricsTable metrics={metrics} />
        )}
      </div>

      {/* Metric explanations */}
      <div className="glass p-6">
        <h2 className="text-base font-semibold text-slate-200 mb-4">Metric Glossary</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          {[
            ['Cumulative Return', 'Total % gain from 1 Jan 2024 to 31 Dec 2024.'],
            ['Sharpe Ratio', 'Annualised excess return (rf=5%) divided by return std. Higher = better risk-adjusted performance.'],
            ['Max Drawdown', 'Largest peak-to-trough decline. Measures downside risk. Less negative = better.'],
            ['Calmar Ratio', 'Annualised return / |Max Drawdown|. Hedge fund standard. Higher = better. Most projects skip this — we don\'t.'],
            ['Turnover', 'Mean daily L1 change in weights. Lower = agent learned to avoid excessive trading.'],
          ].map(([title, desc]) => (
            <div key={title} className="glass-sm p-4">
              <p className="text-indigo-400 font-semibold text-xs mb-1">{title}</p>
              <p className="text-slate-400 text-xs leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Log output */}
      {log && (
        <div className="glass p-5">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Evaluation Output</h3>
          <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap max-h-60 overflow-y-auto leading-relaxed">
            {log}
          </pre>
        </div>
      )}
    </div>
  )
}
