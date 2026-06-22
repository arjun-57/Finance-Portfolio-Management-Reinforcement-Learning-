import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-sm px-3 py-2 text-xs">
      <p className="text-slate-400">Episode {payload[0]?.payload?.episode}</p>
      <p className="text-indigo-400 font-semibold">Reward: {Number(payload[0]?.value).toFixed(4)}</p>
    </div>
  )
}

export default function RewardChart({ data }) {
  if (!data?.length) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-500 text-sm">
        Start training to see the live reward curve.
      </div>
    )
  }

  // Show last 200 episodes max (auto-scroll effect)
  const display = data.slice(-200)

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={display} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
        <XAxis
          dataKey="episode"
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: 'rgba(99,102,241,0.2)' }}
          label={{ value: 'Episode', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 11 }}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          width={55}
          tickFormatter={v => v.toFixed(2)}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={0} stroke="rgba(148,163,184,0.3)" strokeDasharray="4 4" />
        <Line
          type="monotone"
          dataKey="episode_reward"
          stroke="#6366f1"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
