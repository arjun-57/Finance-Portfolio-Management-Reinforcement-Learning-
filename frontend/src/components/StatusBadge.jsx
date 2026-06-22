export default function StatusBadge({ status }) {
  const config = {
    idle:     { label: 'Idle',     dot: 'bg-slate-400',   text: 'text-slate-400',  ring: 'border-slate-500/30' },
    starting: { label: 'Starting', dot: 'bg-yellow-400',  text: 'text-yellow-400', ring: 'border-yellow-500/30' },
    training: { label: 'Training', dot: 'bg-emerald-400', text: 'text-emerald-400',ring: 'border-emerald-500/30' },
    stopped:  { label: 'Stopped',  dot: 'bg-orange-400',  text: 'text-orange-400', ring: 'border-orange-500/30' },
    done:     { label: 'Done',     dot: 'bg-brand-400',   text: 'text-brand-400',  ring: 'border-brand-500/30' },
    error:    { label: 'Error',    dot: 'bg-red-500',     text: 'text-red-400',    ring: 'border-red-500/30' },
  }

  const cfg = config[status] || config.idle

  return (
    <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border ${cfg.ring} bg-white/[0.04]`}>
      <span className={`w-2 h-2 rounded-full ${cfg.dot} ${status === 'training' ? 'blink' : ''}`} />
      <span className={cfg.text}>{cfg.label}</span>
    </span>
  )
}
