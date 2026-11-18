// Pure reducer: client events -> conversation turns (unit-tested in conversation.test.ts).

import type {
  AgentStep,
  ClientEvent,
  GroundingGrade,
  Mode,
  Outcome,
  RelevanceGrade,
  ResponseEnd,
  Source,
  ToolName,
} from './protocol'

export interface Attempt {
  n: number
  text: string
  tokens: number
  firstTokenAt?: number
  lastTokenAt?: number
  sources: Source[]
  discarded?: string
}

export interface TimedStep extends AgentStep {
  at: number
}

export type TurnStatus = 'sent' | 'streaming' | 'done' | 'error' | 'rate_limited' | 'dropped'

export interface Turn {
  id: string
  mode: Mode
  question: string
  status: TurnStatus
  sentAt: number
  startedAt?: number
  endedAt?: number
  traceId?: string | null
  attempts: Attempt[]
  steps: TimedStep[]
  end?: ResponseEnd
  integrity: 'pending' | 'ok' | 'failed'
  integrityDetail?: string
  warning?: string
  error?: string
}

export interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ConversationState {
  turns: Turn[]
  history: HistoryMessage[]
  threadId?: string
  sessionId?: string
  reasoner?: string
  resumed?: boolean
  notices: string[]
}

export type Action =
  | { kind: 'sent'; id: string; question: string; mode: Mode; at: number }
  | { kind: 'event'; event: ClientEvent; at: number }
  | { kind: 'reset' }

export const initialConversation: ConversationState = { turns: [], history: [], notices: [] }

function updateTurn(
  state: ConversationState,
  id: string | null | undefined,
  fn: (t: Turn) => Turn,
): ConversationState {
  if (!id) return state
  const i = state.turns.findIndex((t) => t.id === id)
  if (i < 0) return state
  const turns = state.turns.slice()
  turns[i] = fn(turns[i]!)
  return { ...state, turns }
}

function withAttempt(t: Turn, n: number, fn: (a: Attempt) => Attempt): Turn {
  const attempts = t.attempts.slice()
  const i = attempts.findIndex((a) => a.n === n)
  const current = i >= 0 ? attempts[i]! : { n, text: '', tokens: 0, sources: [] }
  const next = fn(current)
  if (i >= 0) attempts[i] = next
  else attempts.push(next)
  attempts.sort((a, b) => a.n - b.n)
  return { ...t, attempts }
}

export function reduce(state: ConversationState, action: Action): ConversationState {
  if (action.kind === 'reset') return initialConversation
  if (action.kind === 'sent') {
    const turn: Turn = {
      id: action.id,
      mode: action.mode,
      question: action.question,
      status: 'sent',
      sentAt: action.at,
      attempts: [],
      steps: [],
      integrity: 'pending',
    }
    return { ...state, turns: [...state.turns, turn] }
  }

  const { event: ev, at } = action
  switch (ev.type) {
    case 'session.created':
      return {
        ...state,
        sessionId: ev.session_id,
        threadId: ev.thread_id ?? state.threadId,
        reasoner: ev.reasoner ?? state.reasoner,
        resumed: ev.resumed,
      }
    case 'agent.history':
      return { ...state, history: ev.messages, threadId: ev.thread_id }
    case 'response.start':
      return updateTurn(state, ev.id, (t) => ({
        ...t,
        status: 'streaming',
        startedAt: at,
        traceId: ev.trace_id,
      }))
    case 'response.sources':
      return updateTurn(state, ev.id, (t) =>
        withAttempt(t, ev.attempt ?? 1, (a) => ({ ...a, sources: ev.sources })),
      )
    case 'response.token':
      return updateTurn(state, ev.id, (t) =>
        withAttempt(t, ev.attempt ?? 1, (a) => ({
          ...a,
          text: a.text + ev.text,
          tokens: a.tokens + 1,
          firstTokenAt: a.firstTokenAt ?? at,
          lastTokenAt: at,
        })),
      )
    case 'agent.step': {
      const { type: _t, seq: _s, id, ...step } = ev
      return updateTurn(state, id, (t) => ({ ...t, steps: [...t.steps, { ...step, at }] }))
    }
    case 'agent.generation.discarded':
      return updateTurn(state, ev.id, (t) =>
        withAttempt(t, ev.attempt, (a) => ({ ...a, discarded: ev.reason })),
      )
    case 'agent.warning':
      return updateTurn(state, ev.id, (t) => ({ ...t, warning: ev.message }))
    case 'response.end':
      return updateTurn(state, ev.id, (t) => ({ ...t, status: 'done', endedAt: at, end: ev }))
    case 'response.error':
      return updateTurn(state, ev.id, (t) => ({ ...t, status: 'error', endedAt: at, error: ev.error }))
    case 'rate_limited':
      return updateTurn(state, ev.id, (t) => ({
        ...t,
        status: 'rate_limited',
        error: `rate limited, retry in ${Math.ceil(ev.retry_after_ms)} ms`,
      }))
    case 'buffer.overflow':
      return updateTurn(state, ev.dropped_id, (t) => ({
        ...t,
        status: 'dropped',
        error: 'dropped: server buffer overflow',
      }))
    case 'error':
      return ev.id
        ? updateTurn(state, ev.id, (t) => ({ ...t, status: 'error', error: ev.error }))
        : { ...state, notices: [...state.notices, `server error: ${ev.error}`] }
    case 'client.integrity':
      return updateTurn(state, ev.id, (t) => ({
        ...t,
        integrity: ev.ok ? 'ok' : 'failed',
        integrityDetail: ev.detail,
      }))
    case 'client.seq_gap':
      return { ...state, notices: [...state.notices, `sequence gap: expected ${ev.expected}, got ${ev.got}`] }
    default:
      return state
  }
}

// ---------------------------------------------------------------------------
// Derived views used by the UI
// ---------------------------------------------------------------------------

export function finalAttempt(t: Turn): Attempt | undefined {
  const n = t.end?.attempt
  if (n !== undefined) return t.attempts.find((a) => a.n === n)
  return t.attempts.at(-1)
}

export function tokenRate(a: Attempt | undefined): number | undefined {
  if (!a?.firstTokenAt || !a.lastTokenAt || a.tokens < 2) return undefined
  const secs = (a.lastTokenAt - a.firstTokenAt) / 1000
  return secs > 0 ? (a.tokens - 1) / secs : undefined
}

export interface TurnSummary {
  tool?: ToolName
  routeReason?: string
  outcome?: Outcome
  retries: number
  relevance: { score: number; relevant: boolean; reason: string; query?: string }[]
  grounding: { score: number; grounded: boolean; reason: string; unsupported: string[] }[]
  rewrites: { query: string; strategy: string; because: string }[]
  fallbacks: number
}

export function summarize(t: Turn): TurnSummary {
  const route = t.steps.find((s) => s.node === 'route')
  const relevance: TurnSummary['relevance'] = []
  let lastQuery: string | undefined
  for (const s of t.steps) {
    if (s.node === 'retrieve') lastQuery = s.query
    if (s.node === 'grade_documents')
      relevance.push({
        score: s.score ?? 0,
        relevant: Boolean(s.relevant),
        reason: s.reason ?? '',
        query: lastQuery,
      })
  }
  return {
    tool: t.end?.tool ?? route?.tool,
    routeReason: route?.reason,
    outcome: t.end?.outcome,
    retries: t.end?.retries ?? t.steps.filter((s) => s.node === 'rewrite_query').length,
    relevance,
    grounding: t.steps
      .filter((s) => s.node === 'grade_generation')
      .map((s) => ({
        score: s.score ?? 0,
        grounded: Boolean(s.grounded),
        reason: s.reason ?? '',
        unsupported: s.unsupported ?? [],
      })),
    rewrites: t.steps
      .filter((s) => s.node === 'rewrite_query')
      .map((s) => ({ query: s.query ?? '', strategy: s.strategy ?? '', because: s.because ?? '' })),
    fallbacks: t.steps.filter((s) => s.fallback).length,
  }
}

export type { GroundingGrade, RelevanceGrade }
