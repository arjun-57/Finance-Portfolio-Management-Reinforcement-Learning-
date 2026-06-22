import { useState, useEffect } from 'react'
import { api } from '../api'

export function useMetrics(refreshMs = 0) {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetch = () => {
    setLoading(true)
    api.getMetrics()
      .then(data => { setMetrics(data); setError(null) })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    fetch()
    if (refreshMs > 0) {
      const id = setInterval(fetch, refreshMs)
      return () => clearInterval(id)
    }
  }, [refreshMs])

  return { metrics, loading, error, refetch: fetch }
}
