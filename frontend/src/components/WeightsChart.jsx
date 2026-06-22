import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const ASSET_COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#06b6d4',
  '#f97316', '#ec4899', '#a3e635', '#8b5cf6',
]

const ASSET_NAMES = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'GOLDBEES', 'AAPL', 'SPY', 'GOLD']

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-sm px-3 py-2 text-xs min-w-[170px]">
      <p className="text-slate-400 mb-1">Day {label}</p>
      {payload.map(p => (
        <div key={p.dataKey} className="flex justify-between gap-4">
          <span style={{ color: p.fill }}>{p.dataKey}</span>
          <span className="font-semibold tabular-nums">{(Number(p.value) * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

export default function WeightsChart({ data }) {
  if (!data?.length) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 text-sm">
        No weight data yet — run evaluation first.
      </div>
    )
  }

  // Down-sample to max 200 points for performance
  const step = Math.max(1, Math.floor(data.length / 200))
  const sampled = data.filter((_, i) => i % step === 0)

  return (
    <ResponsiveContainer width="100%" height={340}>
      <AreaChart data={sampled} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
        <XAxis
          dataKey="index"
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: 'rgba(99,102,241,0.2)' }}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          width={44}
          domain={[0, 1]}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: '11px', paddingTop: '12px' }}
          formatter={(v) => <span style={{ color: ASSET_COLORS[ASSET_NAMES.indexOf(v)] }}>{v}</span>}
        />
        {ASSET_NAMES.map((name, i) => (
          <Area
            key={name}
            type="monotone"
            dataKey={name}
            stackId="1"
            stroke={ASSET_COLORS[i]}
            fill={ASSET_COLORS[i]}
            fillOpacity={0.85}
            isAnimationActive={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  )
}
