import { useEffect, useState } from 'react'

export interface Health {
  status: string
  backends: Record<string, string>
  inference_ready: boolean
  agent?: { reasoner: string; memory: string }
  websocket: { active: number; max: number; total_accepted: number; total_rejected: number }
  event_loop_lag: { samples: number; p50_ms?: number; p99_ms?: number; max_ms?: number }
}

export function useHealth(intervalMs = 5000): { health?: Health; error?: string } {
  const [health, setHealth] = useState<Health>()
  const [error, setError] = useState<string>()

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const r = await fetch('/health')
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const body = (await r.json()) as Health
        if (!cancelled) {
          setHealth(body)
          setError(undefined)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    }
    void load()
    const timer = setInterval(() => void load(), intervalMs)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [intervalMs])

  return { health, error }
}
