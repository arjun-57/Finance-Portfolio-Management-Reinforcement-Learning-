import { useState, useEffect } from 'react'
import { api } from '../api'

export function usePortfolioValues() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetch = () => {
    setLoading(true)
    api.getPortfolioValues()
      .then(rows => { setData(rows); setError(null) })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetch() }, [])

  return { data, loading, error, refetch: fetch }
}
