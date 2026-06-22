const STRATEGIES = ['PPO Agent', 'Equal-Weight', 'Buy&Hold SPY', 'Markowitz MPT', 'Momentum Top-3']

const STRATEGY_COLORS = {
  'PPO Agent':     'text-indigo-400',
  'Equal-Weight':  'text-emerald-400',
  'Buy&Hold SPY':  'text-amber-400',
  'Markowitz MPT': 'text-cyan-400',
  'Momentum Top-3':'text-orange-400',
}

function Cell({ value, isAgent, highlight }) {
  let cls = isAgent ? 'agent-row' : 'text-slate-300'
  let extra = ''
  if (highlight === 'best') extra = 'text-emerald-400 font-bold'
  else if (highlight === 'worst') extra = 'text-red-400'
  return <td className={`${cls} ${extra}`}>{value}</td>
}

function fmt(v, decimals = 2, suffix = '') {
  if (v === undefined || v === null) return '—'
  return `${Number(v).toFixed(decimals)}${suffix}`
}

function getBest(metrics, key, higherBetter = true) {
  let best = null, bestStrat = null
  for (const [strat, m] of Object.entries(metrics)) {
    const val = m[key]
    if (val === undefined) continue
    if (best === null || (higherBetter ? val > best : val < best)) {
      best = val; bestStrat = strat
    }
  }
  return bestStrat
}

export default function MetricsTable({ metrics }) {
  if (!metrics) {
    return (
      <div className="flex items-center justify-center h-32 text-slate-500 text-sm">
        No metrics yet — run training and evaluation first.
      </div>
    )
  }

  const bestSharpe   = getBest(metrics, 'sharpe_ratio', true)
  const bestCumRet   = getBest(metrics, 'cumulative_return', true)
  const bestDrawdown = getBest(metrics, 'max_drawdown', false)  // less negative = better
  const bestCalmar   = getBest(metrics, 'calmar_ratio', true)

  return (
    <div className="overflow-x-auto">
      <table className="metrics-table w-full">
        <thead>
          <tr>
            <th className="text-left">Strategy</th>
            <th className="text-right">Cum. Return</th>
            <th className="text-right">Sharpe</th>
            <th className="text-right">Max Drawdown</th>
            <th className="text-right">Calmar</th>
            <th className="text-right">Turnover</th>
          </tr>
        </thead>
        <tbody>
          {STRATEGIES.map(strat => {
            const m = metrics[strat]
            if (!m) return null
            const isAgent = strat === 'PPO Agent'
            return (
              <tr key={strat}>
                <td className={`font-semibold ${STRATEGY_COLORS[strat] || 'text-slate-300'}`}>
                  {strat}
                </td>
                <Cell
                  value={fmt(m.cumulative_return, 2, '%')}
                  isAgent={isAgent}
                  highlight={bestCumRet === strat ? 'best' : undefined}
                />
                <Cell
                  value={fmt(m.sharpe_ratio, 3)}
                  isAgent={isAgent}
                  highlight={bestSharpe === strat ? 'best' : undefined}
                />
                <Cell
                  value={fmt(m.max_drawdown, 2, '%')}
                  isAgent={isAgent}
                  highlight={bestDrawdown === strat ? 'best' : undefined}
                />
                <Cell
                  value={fmt(m.calmar_ratio, 3)}
                  isAgent={isAgent}
                  highlight={bestCalmar === strat ? 'best' : undefined}
                />
                <Cell
                  value={m.turnover !== undefined ? fmt(m.turnover, 5) : '—'}
                  isAgent={isAgent}
                />
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
