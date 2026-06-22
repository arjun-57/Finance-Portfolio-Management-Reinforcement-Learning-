import { useState, useEffect, useRef } from 'react'
import { api } from '../api'

export function useTrainingStatus(pollMs = 2000) {
  const [status, setStatus] = useState({ status: 'idle' })
  const [isTraining, setIsTraining] = useState(false)
  const intervalRef = useRef(null)

  const fetchStatus = () => {
    api.getTrainStatus()
      .then(data => {
        setStatus(data)
        setIsTraining(data.status === 'training' || data.status === 'starting')
      })
      .catch(() => {})
  }

  // Poll when training, stop when done
  useEffect(() => {
    fetchStatus()
    intervalRef.current = setInterval(fetchStatus, pollMs)
    return () => clearInterval(intervalRef.current)
  }, [pollMs])

  const startTraining = async (timesteps = 500000) => {
    try {
      await api.startTraining(timesteps)
      setStatus({ status: 'starting', episode: 0, timestep: 0, latest_reward: 0 })
      setIsTraining(true)
    } catch (e) {
      console.error('Start training failed:', e)
    }
  }

  const stopTraining = async () => {
    await api.stopTraining()
    setIsTraining(false)
    setStatus(prev => ({ ...prev, status: 'stopped' }))
  }

  return { status, isTraining, startTraining, stopTraining, refetch: fetchStatus }
}
