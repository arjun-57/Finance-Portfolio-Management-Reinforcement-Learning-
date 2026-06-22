import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import Dashboard  from './pages/Dashboard'
import Train      from './pages/Train'
import Evaluate   from './pages/Evaluate'
import WeightsView from './pages/WeightsView'

const NAV_ITEMS = [
  { to: '/',         label: 'Dashboard', icon: '📊' },
  { to: '/train',    label: 'Train',     icon: '🚀' },
  { to: '/evaluate', label: 'Evaluate',  icon: '⚡' },
  { to: '/weights',  label: 'Weights',   icon: '🎯' },
]

function Sidebar() {
  return (
    <aside className="w-56 flex-shrink-0 flex flex-col glass border-r-0 rounded-none rounded-r-2xl min-h-screen p-5">
      {/* Logo */}
      <div className="mb-8">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-sm font-bold">
            RL
          </div>
          <span className="font-bold text-white text-sm">PortfolioRL</span>
        </div>
        <p className="text-xs text-slate-500 pl-10">PPO · Transformer · Indian Markets</p>
      </div>

      {/* Navigation */}
      <nav className="space-y-1 flex-1">
        {NAV_ITEMS.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150
               ${isActive
                 ? 'bg-indigo-600/25 text-indigo-300 border border-indigo-500/25'
                 : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] border border-transparent'
               }`
            }
          >
            <span className="text-base">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer info */}
      <div className="mt-auto pt-4 border-t border-white/5">
        <p className="text-xs text-slate-600 leading-relaxed">
          8 assets · 2018–2024<br />
          Train → Val → Test split<br />
          4 baselines included
        </p>
      </div>
    </aside>
  )
}

function Layout({ children }) {
  return (
    <div className="gradient-bg min-h-screen flex">
      <Sidebar />
      <main className="flex-1 p-6 md:p-8 overflow-y-auto">
        <div className="max-w-5xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Layout>
        <Routes>
          <Route path="/"         element={<Dashboard />} />
          <Route path="/train"    element={<Train />} />
          <Route path="/evaluate" element={<Evaluate />} />
          <Route path="/weights"  element={<WeightsView />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
