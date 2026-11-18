import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { GatewayClient } from './gatewayClient'
import type { ClientEvent } from './protocol'
import { sha256Hex } from './sha256'

class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: FakeWebSocket[] = []
  readyState = FakeWebSocket.CONNECTING
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((m: { data: unknown }) => void) | null = null
  onclose: ((e: { code: number; reason: string }) => void) | null = null
  url: string
  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }
  send(data: string) {
    this.sent.push(data)
  }
  close() {
    this.readyState = FakeWebSocket.CLOSED
  }
  // test helpers
  open() {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.()
  }
  push(frame: object) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }
  drop(code: number, reason = '') {
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.({ code, reason })
  }
}

const flush = () => new Promise((r) => setTimeout(r, 0))

function makeClient(url = () => 'ws://test/ws/agent') {
  const events: ClientEvent[] = []
  const states: string[] = []
  const client = new GatewayClient({
    url,
    onEvent: (e) => events.push(e),
    onState: (s) => states.push(s),
    WebSocketImpl: FakeWebSocket as unknown as typeof WebSocket,
    initialBackoffMs: 100,
  })
  client.connect()
  const ws = FakeWebSocket.instances.at(-1)!
  ws.open()
  return { client, ws, events, states }
}

async function streamResponse(ws: FakeWebSocket, seq: number, id: string, tokens: string[], attempt = 1) {
  ws.push({ type: 'response.start', id, trace_id: 't', seq: seq++ })
  tokens.forEach((text, index) => ws.push({ type: 'response.token', id, index, text, attempt, seq: seq++ }))
  ws.push({
    type: 'response.end', id, attempt, tokens: tokens.length, latency_ms: 1,
    sha256: await sha256Hex(tokens.join('')), seq: seq++,
  })
  await flush()
  return seq
}

beforeEach(() => {
  FakeWebSocket.instances = []
})
afterEach(() => vi.useRealTimers())

describe('GatewayClient', () => {
  it('verifies contiguous seq numbers and flags gaps', () => {
    const { client, ws, events } = makeClient()
    ws.push({ type: 'session.created', seq: 1 })
    ws.push({ type: 'pong', seq: 2 })
    ws.push({ type: 'pong', seq: 4 })
    expect(client.stats.seqGaps).toBe(1)
    expect(events).toContainEqual({ type: 'client.seq_gap', expected: 3, got: 4 })
  })

  it('answers server heartbeat pings', () => {
    const { ws } = makeClient()
    ws.push({ type: 'ping', ts: 123, seq: 1 })
    expect(JSON.parse(ws.sent[0]!)).toEqual({ type: 'pong', ts: 123 })
  })

  it('verifies the sha256 of a streamed response', async () => {
    const { client, ws, events } = makeClient()
    await streamResponse(ws, 1, 'q1', ['EIO-5 ', 'means ', 'timeout.'])
    expect(events.at(-1)).toEqual({ type: 'client.integrity', id: 'q1', ok: true, detail: 'sha256 verified' })
    expect(client.stats.integrityOk).toBe(1)
  })

  it('checks only the final attempt after a discarded generation', async () => {
    const { ws, events } = makeClient()
    let seq = 1
    ws.push({ type: 'response.start', id: 'q', trace_id: null, seq: seq++ })
    ws.push({ type: 'response.token', id: 'q', index: 0, text: 'made up', attempt: 1, seq: seq++ })
    ws.push({ type: 'agent.generation.discarded', id: 'q', attempt: 1, reason: 'x', seq: seq++ })
    for (const [index, text] of ['grounded ', 'answer'].entries())
      ws.push({ type: 'response.token', id: 'q', index, text, attempt: 2, seq: seq++ })
    ws.push({
      type: 'response.end', id: 'q', attempt: 2, tokens: 2, latency_ms: 1,
      sha256: await sha256Hex('grounded answer'), seq: seq++,
    })
    await flush()
    expect(events.at(-1)).toMatchObject({ type: 'client.integrity', ok: true })
  })

  it('detects tampered text and index gaps', async () => {
    const { client, ws, events } = makeClient()
    ws.push({ type: 'response.start', id: 'a', trace_id: null, seq: 1 })
    ws.push({ type: 'response.token', id: 'a', index: 0, text: 'x', seq: 2 })
    ws.push({ type: 'response.end', id: 'a', tokens: 1, latency_ms: 1, sha256: 'bogus', seq: 3 })
    ws.push({ type: 'response.start', id: 'b', trace_id: null, seq: 4 })
    ws.push({ type: 'response.token', id: 'b', index: 0, text: 'x', seq: 5 })
    ws.push({ type: 'response.token', id: 'b', index: 2, text: 'y', seq: 6 })
    ws.push({ type: 'response.end', id: 'b', tokens: 2, latency_ms: 1, sha256: await sha256Hex('xy'), seq: 7 })
    await flush()
    const integrity = events.filter((e) => e.type === 'client.integrity')
    expect(integrity.map((e) => e.ok)).toEqual([false, false])
    expect(client.stats.integrityFailures).toBe(2)
  })

  it('reconnects with backoff after heartbeat timeout, resetting seq', () => {
    vi.useFakeTimers()
    let thread: string | undefined
    const { client, ws, states } = makeClient(() => `ws://test/ws/agent${thread ? `?thread_id=${thread}` : ''}`)
    ws.push({ type: 'session.created', seq: 1, thread_id: 'abc12345' })
    thread = 'abc12345'
    ws.drop(4408, 'heartbeat timeout')
    expect(client.state).toBe('reconnecting')
    vi.advanceTimersByTime(100)
    const next = FakeWebSocket.instances.at(-1)!
    expect(next.url).toBe('ws://test/ws/agent?thread_id=abc12345')
    next.open()
    next.push({ type: 'session.created', seq: 1 })
    expect(client.stats.seqGaps).toBe(0)
    expect(client.stats.reconnects).toBe(1)
    expect(states).toEqual(['connecting', 'open', 'reconnecting', 'reconnecting', 'open'])
  })

  it('does not reconnect after a policy close or a client close', () => {
    vi.useFakeTimers()
    const a = makeClient()
    a.ws.drop(1008, 'invalid thread_id')
    vi.advanceTimersByTime(5000)
    expect(a.client.state).toBe('closed')
    const b = makeClient()
    b.client.close()
    b.ws.drop(1000)
    vi.advanceTimersByTime(5000)
    expect(FakeWebSocket.instances).toHaveLength(2)
  })

  it('refuses to send while disconnected', () => {
    const { client, ws } = makeClient()
    ws.drop(1006)
    expect(client.send({ type: 'query', id: 'x', text: 'hi' })).toBe(false)
  })
})
