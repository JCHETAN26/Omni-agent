import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RoomClient } from './roomClient'
import type { RoomState } from './roomState'

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  readyState = 0
  sent: (string | Blob | ArrayBuffer)[] = []
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
    this.readyState = 3
  }
  open() {
    this.readyState = 1
    this.onopen?.()
  }
  push(frame: object) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }
  drop(code: number, reason = '') {
    this.readyState = 3
    this.onclose?.({ code, reason })
  }
  frames() {
    return this.sent.filter((f): f is string => typeof f === 'string').map((f) => JSON.parse(f))
  }
}

const snapshot = { type: 'room.snapshot', room_id: 'room1', session: 's1', name: 'Ana', meta: {}, members: [], pins: [], replay: 0, resync: false, heartbeat_interval_s: 15 }
const msg = (seq: number) => ({ seq, type: 'message', session: 's2', name: 'Bo', text: `m${seq}`, client_msg_id: `x${seq}`, at: seq })

function setup() {
  FakeWebSocket.instances = []
  const states: RoomState[] = []
  const conns: string[] = []
  const client = new RoomClient({
    baseUrl: () => 'ws://test',
    roomId: 'room1',
    name: 'Ana',
    onState: (s) => states.push(s),
    onConn: (c) => conns.push(c),
    WebSocketImpl: FakeWebSocket as unknown as typeof WebSocket,
    initialBackoffMs: 100,
    heartbeatMs: 1000,
  })
  client.connect()
  return { client, states, conns, ws: () => FakeWebSocket.instances.at(-1)! }
}

describe('RoomClient', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('resumes with last_seq after a drop and keeps the timeline duplicate-free', () => {
    const { client, ws } = setup()
    expect(ws().url).toBe('ws://test/ws/rooms/room1?name=Ana')
    ws().open()
    ws().push(snapshot)
    ws().push(msg(1))
    ws().push(msg(2))
    ws().drop(1006)
    expect(client.conn).toBe('reconnecting')
    vi.advanceTimersByTime(100)
    expect(ws().url).toBe('ws://test/ws/rooms/room1?name=Ana&last_seq=2')
    ws().open()
    ws().push(snapshot)
    ws().push(msg(2)) // overlap from the replay
    ws().push(msg(3))
    expect(client.state.timeline.map((e) => e.seq)).toEqual([1, 2, 3])
    expect(client.reconnects).toBe(1)
  })

  it('reconnects to resync when it sees a sequence gap', () => {
    const { client, ws } = setup()
    ws().open()
    ws().push(snapshot)
    ws().push(msg(1))
    const first = ws()
    first.push(msg(5))
    expect(client.conn).toBe('reconnecting')
    vi.advanceTimersByTime(100)
    expect(ws()).not.toBe(first)
    expect(ws().url).toContain('last_seq=1')
  })

  it('re-sends unacked messages after a reconnect, then forgets acked ones', () => {
    const { client, ws } = setup()
    ws().open()
    ws().push(snapshot)
    const id = client.sendMessage('disk 3 failed')
    expect(ws().frames().at(-1)).toMatchObject({ type: 'message', text: 'disk 3 failed', client_msg_id: id })
    ws().drop(1006)
    vi.advanceTimersByTime(100)
    ws().open()
    expect(ws().frames().filter((f) => f.type === 'message')).toHaveLength(1) // re-sent
    ws().push({ type: 'ack', client_msg_id: id, acked_seq: 7 })
    expect(client.pending).toBe(0)
    ws().drop(1006)
    vi.advanceTimersByTime(200)
    ws().open()
    expect(ws().frames().filter((f) => f.type === 'message')).toHaveLength(0)
  })

  it('heartbeats, and stops on an unknown room', () => {
    const { client, ws } = setup()
    ws().open()
    vi.advanceTimersByTime(1000)
    expect(ws().frames().some((f) => f.type === 'ping')).toBe(true)
    ws().drop(4404, 'unknown room')
    vi.advanceTimersByTime(10_000)
    expect(client.conn).toBe('closed')
    expect(FakeWebSocket.instances).toHaveLength(1)
  })

  it('sends room actions only while open', () => {
    const { client, ws } = setup()
    expect(client.ask('why?')).toBe(false)
    ws().open()
    expect(client.ask('why?')).toBe(true)
    client.pin('log', 'kernel log', 'EIO')
    client.typing(true)
    expect(ws().frames().map((f) => f.type)).toEqual(['ask', 'pin', 'typing'])
  })
})
