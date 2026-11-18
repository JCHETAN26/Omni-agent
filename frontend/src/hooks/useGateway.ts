import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { initialConversation, reduce } from '../lib/conversation'
import { type ClientStats, type ConnState, GatewayClient, gatewayUrl } from '../lib/gatewayClient'
import type { Mode } from '../lib/protocol'

const THREAD_KEY = 'omniagent.thread_id'

const EMPTY_STATS: ClientStats = {
  framesIn: 0,
  framesOut: 0,
  seqGaps: 0,
  integrityOk: 0,
  integrityFailures: 0,
  reconnects: 0,
  lastSeq: 0,
}

function storedThread(): string | undefined {
  try {
    return localStorage.getItem(THREAD_KEY) ?? undefined
  } catch {
    return undefined
  }
}

function storeThread(id: string | undefined): void {
  try {
    if (id) localStorage.setItem(THREAD_KEY, id)
    else localStorage.removeItem(THREAD_KEY)
  } catch {
    /* storage unavailable: conversation simply won't resume */
  }
}

function newId(): string {
  return crypto.randomUUID().replace(/-/g, '').slice(0, 16)
}

export interface Gateway {
  state: ConnState
  detail?: string
  stats: ClientStats
  conversation: ReturnType<typeof reduce>
  /** Tokens received per second over the last 3 s (all requests). */
  liveTokenRate: number
  ask: (question: string) => boolean
  newConversation: () => void
}

export function useGateway(mode: Mode): Gateway {
  const [conversation, dispatch] = useReducer(reduce, initialConversation)
  const [state, setState] = useState<ConnState>('idle')
  const [detail, setDetail] = useState<string>()
  const [stats, setStats] = useState<ClientStats>(EMPTY_STATS)
  const [liveTokenRate, setLiveTokenRate] = useState(0)
  const clientRef = useRef<GatewayClient | null>(null)
  const tokenTimes = useRef<number[]>([])
  const [epoch, setEpoch] = useState(0)

  useEffect(() => {
    dispatch({ kind: 'reset' })
    const client = new GatewayClient({
      url: () =>
        mode === 'agent'
          ? gatewayUrl('/ws/agent', { thread_id: storedThread() })
          : gatewayUrl('/ws/stream'),
      onState: (s, d) => {
        setState(s)
        setDetail(d)
        setStats({ ...client.stats }) // a new client starts from zeroed stats
      },
      onEvent: (event) => {
        const at = performance.now()
        if (event.type === 'response.token') tokenTimes.current.push(at)
        if (event.type === 'session.created' && event.thread_id) storeThread(event.thread_id)
        dispatch({ kind: 'event', event, at })
        setStats({ ...client.stats })
      },
    })
    clientRef.current = client
    client.connect()
    return () => client.close()
  }, [mode, epoch])

  useEffect(() => {
    const timer = setInterval(() => {
      const now = performance.now()
      tokenTimes.current = tokenTimes.current.filter((t) => now - t < 3000)
      setLiveTokenRate(tokenTimes.current.length / 3)
    }, 250)
    return () => clearInterval(timer)
  }, [])

  const ask = useCallback(
    (question: string) => {
      const client = clientRef.current
      const text = question.trim()
      if (!client || !text) return false
      const id = newId()
      if (!client.send({ type: 'query', id, text, top_k: 4 })) return false
      dispatch({ kind: 'sent', id, question: text, mode, at: performance.now() })
      setStats({ ...client.stats })
      return true
    },
    [mode],
  )

  const newConversation = useCallback(() => {
    storeThread(undefined)
    setEpoch((e) => e + 1)
  }, [])

  return {
    state,
    detail,
    stats,
    conversation,
    liveTokenRate,
    ask,
    newConversation,
  }
}
