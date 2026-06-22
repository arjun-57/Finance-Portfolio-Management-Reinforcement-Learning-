import { useState, useEffect, useRef } from 'react'
import RewardChart from '../components/RewardChart'
import StatusBadge from '../components/StatusBadge'
import { useTrainingStatus } from '../hooks/useTrainingStatus'
import { api } from '../api'

const TIMESTEP_OPTIONS = [
  { label: '100k steps (fast demo)', value: 100_000 },
  { label: '250k steps (balanced)', value: 250_000 },
  { label: '500k steps (full)', value: 500_000 },
]

export default function Train() {
  const { status, isTraining, startTraining, stopTraining } = useTrainingStatus(2000)
  const [rewardLog, setRewardLog] = useState([])
  const [timesteps, setTimesteps] = useState(500_000)
  const [fetchError, setFetchError] = useState(null)
  const pollRef = useRef(null)

  // Poll reward log while training
  useEffect(() => {
    const fetchLog = () => {
      api.getRewardLog()
        .then(data => { setRewardLog(data); setFetchError(null) })
        .catch(err => setFetchError(err.message))
    }

    if (isTraining || status.status === 'done') {
      fetchLog()
      pollRef.current = setInterval(fetchLog, 3000)
    } else {
      // Try to load existing log on mount
      fetchLog()
    }
    return () => clearInterval(pollRef.current)
  }, [isTraining, status.status])

  const episode  = status.episode ?? 0
  const reward   = typeof status.latest_reward === 'number' ? status.latest_reward.toFixed(4) : '—'
  const portVal  = typeof status.portfolio_value === 'number'
    ? status.portfolio_value.toFixed(2)
    : '—'

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Train Agent</h1>
        <p className="text-slate-400 text-sm mt-1">
          PPO + Transformer on 2018–2022 training data
        </p>
      </div>

      {/* Control panel */}
      <div className="glass p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="text-slate-300 font-semibold text-sm">Status</span>
            <StatusBadge status={status.status ?? 'idle'} />
          </div>

          <div className="flex gap-3">
            {!isTraining ? (
              <button
                className="btn-glow"
                onClick={() => startTraining(timesteps)}
              >
                ▶ Start Training
              </button>
            ) : (
              <button
                className="btn-glow btn-danger"
                onClick={stopTraining}
              >
                ■ Stop
              </button>
            )}
          </div>
        </div>

        {/* Timestep selector */}
        <div className="flex flex-wrap gap-2">
          {TIMESTEP_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setTimesteps(opt.value)}
              disabled={isTraining}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border
                ${timesteps === opt.value
                  ? 'bg-indigo-600/40 border-indigo-500/60 text-indigo-300'
                  : 'bg-white/[0.04] border-white/10 text-slate-400 hover:border-indigo-500/30'
                }
                ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Live stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="glass-sm p-4 text-center">
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Episode</p>
          <p className="text-2xl font-bold text-white font-mono">{episode}</p>
        </div>
        <div className="glass-sm p-4 text-center">
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Latest Reward</p>
          <p className={`text-2xl font-bold font-mono ${parseFloat(reward) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {reward}
          </p>
        </div>
        <div className="glass-sm p-4 text-center">
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Portfolio Value</p>
          <p className="text-2xl font-bold text-indigo-400 font-mono">${portVal}</p>
        </div>
      </div>

      {/* Reward chart */}
      <div className="glass p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-slate-200">Episode Reward Curve</h2>
          <span className="text-xs text-slate-500 font-mono">{rewardLog.length} episodes</span>
        </div>
        <RewardChart data={rewardLog} />
        {fetchError && (
          <p className="text-xs text-slate-500 mt-2">
            Reward log: {fetchError}
          </p>
        )}
      </div>

      {/* Training tips */}
      <div className="glass p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">💡 Training Tips</h3>
        <ul className="text-xs text-slate-400 space-y-2 leading-relaxed">
          <li>• <span className="text-indigo-400 font-medium">100k steps</span> is sufficient to see learning — good for demo runs.</li>
          <li>• <span className="text-indigo-400 font-medium">500k steps</span> typically converges to a stable Sharpe &gt; 1.0 on validation.</li>
          <li>• Reward trending upward = agent learning to avoid volatile positions.</li>
          <li>• Checkpoints are saved every 10 episodes to <code className="text-slate-300 bg-white/5 px-1 rounded">runs/</code></li>
          <li>• After training stops, click <strong className="text-white">Evaluate</strong> to generate metrics.</li>
        </ul>
      </div>
    </div>
  )
}
