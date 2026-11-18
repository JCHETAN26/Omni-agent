import { describe, expect, it } from 'vitest'
import { finalAttempt, initialConversation, reduce, summarize, tokenRate, type Action } from './conversation'
import type { ClientEvent } from './protocol'

const ev = (event: object, at = 0): Action => ({ kind: 'event', event: event as ClientEvent, at })

function run(actions: Action[]) {
  return actions.reduce(reduce, initialConversation)
}

describe('conversation reducer', () => {
  it('builds a self-corrected agent turn', () => {
    const s = run([
      ev({ type: 'session.created', seq: 1, session_id: 's', thread_id: 'th', reasoner: 'heuristic' }),
      { kind: 'sent', id: 'q', question: 'EIO-5?', mode: 'agent', at: 0 },
      ev({ type: 'response.start', id: 'q', trace_id: 'tr', seq: 2 }, 10),
      ev({ type: 'agent.step', id: 'q', node: 'route', tool: 'knowledge_base', reason: 'kb', seq: 3 }, 12),
      ev({ type: 'agent.step', id: 'q', node: 'retrieve', query: 'eio5', attempt: 1, seq: 4 }, 13),
      ev({ type: 'agent.step', id: 'q', node: 'grade_documents', score: 0.2, relevant: false, reason: 'r', seq: 5 }, 14),
      ev({ type: 'agent.step', id: 'q', node: 'rewrite_query', query: 'eio 5', strategy: 'keyword_focus', because: 'low', seq: 6 }, 15),
      ev({ type: 'agent.step', id: 'q', node: 'retrieve', query: 'eio 5', attempt: 2, seq: 7 }, 16),
      ev({ type: 'agent.step', id: 'q', node: 'grade_documents', score: 0.9, relevant: true, reason: 'ok', seq: 8 }, 17),
      ev({ type: 'response.sources', id: 'q', attempt: 1, sources: [], seq: 9 }, 18),
      ev({ type: 'response.token', id: 'q', index: 0, text: 'bad', attempt: 1, seq: 10 }, 20),
      ev({ type: 'agent.generation.discarded', id: 'q', attempt: 1, reason: 'not grounded', seq: 11 }, 21),
      ev({ type: 'response.token', id: 'q', index: 0, text: 'good ', attempt: 2, seq: 12 }, 30),
      ev({ type: 'response.token', id: 'q', index: 1, text: 'answer', attempt: 2, seq: 13 }, 40),
      ev({ type: 'agent.step', id: 'q', node: 'grade_generation', score: 1, grounded: true, reason: 'g', seq: 14 }, 41),
      ev({ type: 'response.end', id: 'q', attempt: 2, tokens: 2, sha256: 'x', latency_ms: 40, outcome: 'answered', tool: 'knowledge_base', retries: 1, seq: 15 }, 42),
      ev({ type: 'client.integrity', id: 'q', ok: true, detail: 'sha256 verified' }, 43),
    ])
    expect(s.threadId).toBe('th')
    const t = s.turns[0]!
    expect(t.status).toBe('done')
    expect(t.integrity).toBe('ok')
    expect(t.attempts.map((a) => [a.n, a.text, a.discarded])).toEqual([
      [1, 'bad', 'not grounded'],
      [2, 'good answer', undefined],
    ])
    expect(finalAttempt(t)?.text).toBe('good answer')
    expect(tokenRate(finalAttempt(t))).toBeCloseTo(100)
    const sum = summarize(t)
    expect(sum.tool).toBe('knowledge_base')
    expect(sum.retries).toBe(1)
    expect(sum.relevance.map((r) => [r.query, r.relevant])).toEqual([
      ['eio5', false],
      ['eio 5', true],
    ])
    expect(sum.grounding).toEqual([{ score: 1, grounded: true, reason: 'g', unsupported: [] }])
    expect(sum.rewrites[0]?.strategy).toBe('keyword_focus')
  })

  it('records rate limits, drops, errors and seq gaps', () => {
    const s = run([
      { kind: 'sent', id: 'a', question: 'x', mode: 'stream', at: 0 },
      { kind: 'sent', id: 'b', question: 'y', mode: 'stream', at: 0 },
      { kind: 'sent', id: 'c', question: 'z', mode: 'stream', at: 0 },
      ev({ type: 'rate_limited', id: 'a', retry_after_ms: 99.2, seq: 1 }),
      ev({ type: 'buffer.overflow', dropped: 'QueryItem', dropped_id: 'b', total_dropped: 1, seq: 2 }),
      ev({ type: 'response.error', id: 'c', error: 'boom', seq: 3 }),
      ev({ type: 'client.seq_gap', expected: 4, got: 6 }),
    ])
    expect(s.turns.map((t) => t.status)).toEqual(['rate_limited', 'dropped', 'error'])
    expect(s.turns[0]!.error).toContain('100 ms')
    expect(s.notices).toEqual(['sequence gap: expected 4, got 6'])
  })

  it('restores history on resume and resets cleanly', () => {
    const s = run([
      ev({ type: 'agent.history', thread_id: 'th', messages: [{ role: 'user', content: 'hi' }], seq: 2 }),
    ])
    expect(s.history).toHaveLength(1)
    expect(reduce(s, { kind: 'reset' })).toBe(initialConversation)
  })
})
