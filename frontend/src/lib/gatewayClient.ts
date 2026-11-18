// Transport client for the OmniAgent WebSocket gateway.
//
// Responsibilities (all independent of React, unit-tested in gatewayClient.test.ts):
//  * connection lifecycle with exponential-backoff reconnects
//  * answering server heartbeat pings
//  * sequence verification: every server frame carries a per-session `seq`
//    that must increase by exactly 1 (gaps -> `client.seq_gap`)
//  * response integrity: token `index` must be contiguous per (request, attempt),
//    and on `response.end` the SHA-256 of the final attempt's text must match
//    (result -> `client.integrity`)

import type { ClientEvent, QueryFrame, ServerEvent } from './protocol'
import { sha256Hex } from './sha256'

export type ConnState = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed'

export interface ClientStats {
  framesIn: number
  framesOut: number
  seqGaps: number
  integrityOk: number
  integrityFailures: number
  reconnects: number
  lastSeq: number
}

export interface GatewayClientOptions {
  /** Called on every (re)connect, so it can include a resumable thread id. */
  url: () => string
  onEvent: (event: ClientEvent) => void
  onState?: (state: ConnState, detail?: string) => void
  WebSocketImpl?: typeof WebSocket
  initialBackoffMs?: number
  maxBackoffMs?: number
}

// Close codes after which reconnecting cannot help.
const FATAL_CLOSE_CODES = new Set([1008, 1009])

interface AttemptText {
  parts: string[]
  nextIndex: number
  indexOk: boolean
}

export class GatewayClient {
  private ws: WebSocket | null = null
  private opts: Required<Omit<GatewayClientOptions, 'onState'>> & Pick<GatewayClientOptions, 'onState'>
  private backoff: number
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private userClosed = false
  private responses = new Map<string, Map<number, AttemptText>>()
  state: ConnState = 'idle'
  stats: ClientStats = {
    framesIn: 0,
    framesOut: 0,
    seqGaps: 0,
    integrityOk: 0,
    integrityFailures: 0,
    reconnects: 0,
    lastSeq: 0,
  }

  constructor(options: GatewayClientOptions) {
    this.opts = {
      WebSocketImpl: globalThis.WebSocket,
      initialBackoffMs: 500,
      maxBackoffMs: 10_000,
      ...options,
    }
    this.backoff = this.opts.initialBackoffMs
  }

  connect(): void {
    this.userClosed = false
    this.open(this.stats.reconnects > 0 ? 'reconnecting' : 'connecting')
  }

  close(): void {
    this.userClosed = true
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = null
    this.ws?.close(1000, 'client closed')
    this.ws = null
    this.setState('closed', 'closed by client')
  }

  send(frame: QueryFrame | Record<string, unknown>): boolean {
    if (!this.ws || this.ws.readyState !== this.opts.WebSocketImpl.OPEN) return false
    this.ws.send(JSON.stringify(frame))
    this.stats.framesOut++
    return true
  }

  private setState(state: ConnState, detail?: string): void {
    this.state = state
    this.opts.onState?.(state, detail)
  }

  private open(state: ConnState): void {
    this.setState(state)
    const ws = new this.opts.WebSocketImpl(this.opts.url())
    this.ws = ws
    ws.onopen = () => {
      this.backoff = this.opts.initialBackoffMs
      this.stats.lastSeq = 0 // seq numbering restarts with every session
      this.setState('open')
    }
    ws.onmessage = (msg: MessageEvent) => {
      if (typeof msg.data !== 'string') return
      let event: ServerEvent
      try {
        event = JSON.parse(msg.data) as ServerEvent
      } catch {
        return
      }
      void this.handle(event)
    }
    ws.onclose = (ev: CloseEvent) => {
      if (this.ws !== ws) return
      this.ws = null
      if (this.userClosed) return
      if (FATAL_CLOSE_CODES.has(ev.code)) {
        this.setState('closed', `closed by server: ${ev.code} ${ev.reason}`)
        return
      }
      this.scheduleReconnect(`connection lost (${ev.code}${ev.reason ? ` ${ev.reason}` : ''})`)
    }
  }

  private scheduleReconnect(detail: string): void {
    this.setState('reconnecting', detail)
    const delay = this.backoff
    this.backoff = Math.min(this.backoff * 2, this.opts.maxBackoffMs)
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.stats.reconnects++
      this.open('reconnecting')
    }, delay)
  }

  private async handle(event: ServerEvent): Promise<void> {
    this.stats.framesIn++
    const expected = this.stats.lastSeq + 1
    if (event.seq !== expected) {
      this.stats.seqGaps++
      this.opts.onEvent({ type: 'client.seq_gap', expected, got: event.seq })
    }
    this.stats.lastSeq = event.seq

    switch (event.type) {
      case 'ping':
        this.send({ type: 'pong', ts: event.ts })
        break
      case 'response.start':
        this.responses.set(event.id, new Map())
        break
      case 'response.token': {
        const attempts = this.responses.get(event.id) ?? new Map<number, AttemptText>()
        this.responses.set(event.id, attempts)
        const n = event.attempt ?? 1
        const a = attempts.get(n) ?? { parts: [], nextIndex: 0, indexOk: true }
        attempts.set(n, a)
        if (event.index !== a.nextIndex) a.indexOk = false
        a.nextIndex = event.index + 1
        a.parts.push(event.text)
        break
      }
      default:
        break
    }

    this.opts.onEvent(event)

    if (event.type === 'response.end') {
      const attempt = this.responses.get(event.id)?.get(event.attempt ?? 1)
      this.responses.delete(event.id)
      const text = attempt?.parts.join('') ?? ''
      const count = attempt?.parts.length ?? 0
      let ok = count === event.tokens && (attempt?.indexOk ?? event.tokens === 0)
      let detail = ok ? '' : `token count/index mismatch (got ${count}, expected ${event.tokens})`
      if (ok) {
        ok = (await sha256Hex(text)) === event.sha256
        detail = ok ? 'sha256 verified' : 'sha256 mismatch'
      }
      if (ok) this.stats.integrityOk++
      else this.stats.integrityFailures++
      this.opts.onEvent({ type: 'client.integrity', id: event.id, ok, detail })
    } else if (event.type === 'response.error') {
      this.responses.delete(event.id)
    }
  }
}

/** ws(s)://<current host><path>, so the app works behind any ingress or proxy. */
export function gatewayUrl(path: string, params: Record<string, string | undefined> = {}): string {
  const loc = globalThis.location
  const base = `${loc.protocol === 'https:' ? 'wss:' : 'ws:'}//${loc.host}${path}`
  const qs = new URLSearchParams(
    Object.entries(params).filter((e): e is [string, string] => Boolean(e[1])),
  ).toString()
  return qs ? `${base}?${qs}` : base
}
