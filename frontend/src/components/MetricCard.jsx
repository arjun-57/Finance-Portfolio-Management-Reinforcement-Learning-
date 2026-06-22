export default function MetricCard({ label, value, unit = '', delta = null, icon = null, color = 'brand' }) {
  const colorMap = {
    brand:   'from-indigo-500/20 to-violet-500/10 border-indigo-500/25',
    green:   'from-emerald-500/20 to-teal-500/10 border-emerald-500/25',
    amber:   'from-amber-500/20 to-orange-500/10 border-amber-500/25',
    red:     'from-red-500/20 to-rose-500/10 border-red-500/25',
    cyan:    'from-cyan-500/20 to-sky-500/10 border-cyan-500/25',
  }

  const isPositive = delta !== null && parseFloat(delta) >= 0

  return (
    <div className={`glass bg-gradient-to-br ${colorMap[color] || colorMap.brand} p-5 animate-slide-up`}>
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs text-slate-400 font-semibold uppercase tracking-widest">{label}</p>
        {icon && <span className="text-xl">{icon}</span>}
      </div>

      <p className="text-3xl font-bold text-white tabular-nums">
        {value}
        {unit && <span className="text-base font-normal text-slate-400 ml-1">{unit}</span>}
      </p>

      {delta !== null && (
        <div className="mt-2">
          <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${isPositive ? 'badge-up' : 'badge-down'}`}>
            {isPositive ? '▲' : '▼'} {Math.abs(delta)}
          </span>
        </div>
      )}
    </div>
  )
}
