// Transport for /ws/rooms/{id}, React-agnostic (tested in roomClient.test.ts).
//
//  * reconnects with exponential backoff, always passing `last_seq` so the
//    server replays exactly the missed sequenced events
//  * a sequence gap (missed events) closes and reconnects to resync
//  * messages are idempotent: each carries a client_msg_id and stays queued
//    until acked, and unacked ones are re-sent after a reconnect (the server
//    drops duplicates)
//  * heartbeat ping every interval so the server keeps the session

import type { PinKind, RoomServerEvent } from './roomProtocol'
import { applyRoomEvent, emptyRoom, type RoomState } from './roomState'

export type RoomConnState = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed'

export interface RoomClientOptions {
  baseUrl: () => string // e.g. ws://host
  roomId: string
  name: string
  onState: (s: RoomState) => void
  onConn?: (c: RoomConnState, detail?: string) => void
  WebSocketImpl?: typeof WebSocket
  initialBackoffMs?: number
  maxBackoffMs?: number
  heartbeatMs?: number
}

const FATAL = new Set([4404, 1008])

let counter = 0
export const newMsgId = () => `${Date.now().toString(36)}-${(counter++).toString(36)}-${Math.random().toString(36).slice(2, 8)}`

export class RoomClient {
  state: RoomState = emptyRoom()
  conn: RoomConnState = 'idle'
  reconnects = 0
  private ws?: WebSocket
  private backoff: number
  private timer?: ReturnType<typeof setTimeout>
  private heartbeat?: ReturnType<typeof setInterval>
  private stopped = false
  private unacked = new Map<string, string>() // client_msg_id -> frame

  private o: RoomClientOptions

  constructor(o: RoomClientOptions) {
    this.o = o
    this.backoff = o.initialBackoffMs ?? 500
  }

  connect(): void {
    this.stopped = false
    const WS = this.o.WebSocketImpl ?? WebSocket
    const q = new URLSearchParams({ name: this.o.name })
    if (this.state.lastSeq > 0 && this.state.roomId === this.o.roomId) q.set('last_seq', String(this.state.lastSeq))
    this.setConn(this.reconnects > 0 ? 'reconnecting' : 'connecting')
    const ws = new WS(`${this.o.baseUrl()}/ws/rooms/${encodeURIComponent(this.o.roomId)}?${q}`)
    this.ws = ws
    ws.onopen = () => {
      this.backoff = this.o.initialBackoffMs ?? 500
      this.setConn('open')
      for (const frame of this.unacked.values()) ws.send(frame)
      clearInterval(this.heartbeat)
      this.heartbeat = setInterval(() => this.raw({ type: 'ping', ts: Date.now() }), this.o.heartbeatMs ?? 15_000)
    }
    ws.onmessage = (m) => {
      if (typeof m.data !== 'string') return
      const e = JSON.parse(m.data) as RoomServerEvent
      if (e.type === 'ack') this.unacked.delete(e.client_msg_id)
      this.state = applyRoomEvent(this.state, e)
      this.o.onState(this.state)
      if (this.state.gap) {
        // Missed events: reconnect with last_seq and let the server replay them.
        ws.close()
        this.scheduleReconnect('sequence gap: resyncing')
      }
    }
    ws.onclose = (e) => {
      clearInterval(this.heartbeat)
      if (this.ws !== ws || this.stopped) return
      if (FATAL.has(e.code)) {
        this.setConn('closed', e.reason || `closed (${e.code})`)
        return
      }
      this.scheduleReconnect(e.code === 4409 ? 'too slow: resyncing' : e.reason || `closed (${e.code})`)
    }
  }

  private scheduleReconnect(detail: string) {
    this.ws = undefined
    this.reconnects++
    this.setConn('reconnecting', detail)
    clearTimeout(this.timer)
    this.timer = setTimeout(() => this.connect(), this.backoff)
    this.backoff = Math.min(this.backoff * 2, this.o.maxBackoffMs ?? 10_000)
  }

  close(): void {
    this.stopped = true
    clearTimeout(this.timer)
    clearInterval(this.heartbeat)
    this.ws?.close()
    this.setConn('closed')
  }

  private setConn(c: RoomConnState, detail?: string) {
    this.conn = c
    this.o.onConn?.(c, detail)
  }

  private raw(frame: object): boolean {
    if (this.ws?.readyState !== 1) return false
    this.ws.send(JSON.stringify(frame))
    return true
  }

  /** Queued until acked; re-sent after reconnects (deduplicated server-side). */
  sendMessage(text: string): string {
    const id = newMsgId()
    const frame = JSON.stringify({ type: 'message', text, client_msg_id: id, client_ts: Date.now() })
    this.unacked.set(id, frame)
    if (this.ws?.readyState === 1) this.ws.send(frame)
    return id
  }

  get pending(): number {
    return this.unacked.size
  }

  typing(active: boolean) {
    return this.raw({ type: 'typing', active })
  }
  focus(pane: string) {
    return this.raw({ type: 'focus', pane })
  }
  pin(kind: PinKind, title: string, snippet = '', ref = '') {
    return this.raw({ type: 'pin', kind, title, snippet, ref })
  }
  unpin(pinId: string) {
    return this.raw({ type: 'unpin', pin_id: pinId })
  }
  ask(question: string) {
    return this.raw({ type: 'ask', question })
  }
  sendFrame(image: Blob | ArrayBuffer) {
    if (this.ws?.readyState !== 1) return false
    this.ws.send(image)
    return true
  }
}

export async function createRoom(title: string, fetchImpl: typeof fetch = fetch): Promise<string> {
  const r = await fetchImpl('/rooms', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (!r.ok) throw new Error(`could not create room (HTTP ${r.status})`)
  return ((await r.json()) as { room_id: string }).room_id
}
