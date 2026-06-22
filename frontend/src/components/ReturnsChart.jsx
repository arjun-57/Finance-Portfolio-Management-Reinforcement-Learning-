import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const STRATEGY_COLORS = {
  'PPO Agent':     '#6366f1',
  'Equal-Weight':  '#10b981',
  'Buy&Hold SPY':  '#f59e0b',
  'Markowitz MPT': '#06b6d4',
  'Momentum Top-3':'#f97316',
}

const STRATEGY_KEYS = Object.keys(STRATEGY_COLORS)

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-sm px-3 py-2 text-xs min-w-[170px]">
      <p className="text-slate-400 mb-1 font-mono">{label}</p>
      {payload.map(p => (
        <div key={p.dataKey} className="flex justify-between gap-4">
          <span style={{ color: p.color }}>{p.dataKey}</span>
          <span className="font-semibold tabular-nums">{Number(p.value).toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

export default function ReturnsChart({ data }) {
  if (!data?.length) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 text-sm">
        No portfolio data yet — run evaluation first.
      </div>
    )
  }

  // Determine which keys actually exist in the data
  const availableKeys = STRATEGY_KEYS.filter(k => data[0] && k in data[0])

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" />
        <XAxis
          dataKey="date"
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: 'rgba(99,102,241,0.2)' }}
          tickFormatter={v => v?.slice(0, 10) ?? v}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={v => `${v.toFixed(0)}`}
          width={48}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: '12px', paddingTop: '12px' }}
          formatter={(value) => <span style={{ color: STRATEGY_COLORS[value] }}>{value}</span>}
        />
        {availableKeys.map(key => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={STRATEGY_COLORS[key]}
            strokeWidth={key === 'PPO Agent' ? 2.5 : 1.5}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
