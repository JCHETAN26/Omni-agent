// Wire protocol of the OmniAgent WebSocket gateway (/ws/stream and /ws/agent).
// Mirrors backend/api/websocket.py, backend/api/streaming.py and backend/agent/runner.py.

export type Mode = 'agent' | 'stream'

export interface Source {
  chunk_id: string
  document_title: string
  section_heading: string
  rrf_score: number
  rerank_score: number | null
  dense_rank: number | null
  sparse_rank: number | null
}

export interface RelevanceGrade {
  score: number
  relevant: boolean
  reason: string
}

export interface GroundingGrade {
  score: number
  grounded: boolean
  unsupported_claims: string[]
  reason: string
}

export type ToolName = 'knowledge_base' | 'live_logs' | 'calculator' | 'direct'
export type Outcome = 'answered' | 'abstained' | 'insufficient_context' | 'ungrounded'

/** One `agent.step` payload. Fields depend on `node`; all optional here. */
export interface AgentStep {
  node: string
  fallback?: boolean
  standalone?: string
  changed?: boolean
  tool?: ToolName
  argument?: string
  reason?: string
  query?: string
  attempt?: number
  hits?: string[]
  reranked?: boolean
  score?: number
  relevant?: boolean
  retry?: number
  previous?: string
  strategy?: string
  because?: string
  host?: string | null
  lines?: number
  expression?: string
  ok?: boolean
  result?: string
  chars?: number
  abstained?: boolean
  insufficient_context?: boolean
  grounded?: boolean
  unsupported?: string[]
  will_retry?: boolean
  outcome?: Outcome
  retries?: number
  attempts?: number
}

interface Base {
  seq: number
}

export type ServerEvent =
  | (Base & {
      type: 'session.created'
      session_id: string
      heartbeat_interval_s: number
      heartbeat_timeout_s: number
      rate_limit: { capacity: number; refill_per_s: number }
      buffer_size: number
      thread_id?: string
      resumed?: boolean
      history_turns?: number
      reasoner?: string
    })
  | (Base & {
      type: 'agent.history'
      thread_id: string
      messages: { role: 'user' | 'assistant'; content: string }[]
    })
  | (Base & { type: 'ping'; ts: number })
  | (Base & { type: 'pong'; ts: unknown })
  | (Base & { type: 'rate_limited'; id: string | null; retry_after_ms: number })
  | (Base & {
      type: 'buffer.overflow'
      dropped: string
      dropped_id: string | null
      total_dropped: number
    })
  | (Base & { type: 'error'; id?: string | null; code: string; error: string })
  | (Base & { type: 'response.start'; id: string; trace_id: string | null; thread_id?: string })
  | (Base & {
      type: 'response.sources'
      id: string
      sources: Source[]
      attempt?: number
      reranked?: boolean
      retrieval_ms?: number
    })
  | (Base & { type: 'response.token'; id: string; index: number; text: string; attempt?: number })
  | (Base & AgentStep & { type: 'agent.step'; id: string })
  | (Base & { type: 'agent.generation.discarded'; id: string; attempt: number; reason: string })
  | (Base & { type: 'agent.warning'; id: string; message: string })
  | (Base & {
      type: 'response.end'
      id: string
      tokens: number
      sha256: string
      latency_ms: number
      first_token_ms?: number | null
      attempt?: number
      outcome?: Outcome
      tool?: ToolName
      retries?: number
      rewrites?: string[]
      relevance?: RelevanceGrade | Record<string, never>
      grounding?: GroundingGrade | Record<string, never>
    })
  | (Base & { type: 'response.error'; id: string; error: string })
  | (Base & { type: 'audio.received'; id: string; chunks: number; bytes: number; sha256: string })

/** Events the client synthesizes on top of the server stream. */
export type ClientEvent =
  | ServerEvent
  | { type: 'client.integrity'; id: string; ok: boolean; detail: string }
  | { type: 'client.seq_gap'; expected: number; got: number }

export type ResponseEnd = Extract<ServerEvent, { type: 'response.end' }>

export interface QueryFrame {
  type: 'query'
  id: string
  text: string
  top_k?: number
  traceparent?: string
}
