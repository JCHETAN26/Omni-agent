import { useEffect, useRef, useState } from 'react'
import { useRoom } from '../hooks/useRoom'
import type { RoomClient, RoomConnState } from '../lib/roomClient'
import { createRoom } from '../lib/roomClient'
import type { Member, Pin, SequencedEvent } from '../lib/roomProtocol'
import { typingLine, type LiveTurn, type RoomState } from '../lib/roomState'
import { Badge } from './ui'

const ROOM_RE = /^[a-z0-9][a-z0-9-]{3,39}$/

function roomFromHash(): string | undefined {
  const id = /room=([a-z0-9-]+)/.exec(location.hash)?.[1]
  return id && ROOM_RE.test(id) ? id : undefined
}

const initials = (name: string) =>
  name
    .split(/\s+/)
    .map((w) => w[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()

export function PresenceBar({ members, me }: { members: Record<string, Member>; me?: string }) {
  const list = Object.values(members)
  return (
    <div className="flex flex-wrap items-center gap-1.5" data-testid="presence">
      {list.map((m) => (
        <span
          key={m.session}
          title={`${m.name}${m.focus ? ` · ${m.focus}` : ''}`}
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs ring-1 ring-inset ${
            m.session === me ? 'bg-sky-950 text-sky-200 ring-sky-700' : 'bg-slate-800 text-slate-200 ring-slate-700'
          }`}
        >
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          {initials(m.name)} <span className="text-slate-400">{m.name}</span>
          {m.focus && m.focus !== 'timeline' && <span className="text-slate-500">· {m.focus}</span>}
        </span>
      ))}
      <span className="text-xs text-slate-500">{list.length} here</span>
    </div>
  )
}

function EventRow({ e, onPin }: { e: SequencedEvent; onPin: (title: string, snippet: string) => void }) {
  switch (e.type) {
    case 'message':
      return (
        <div className="group flex gap-2" data-testid="room-message">
          <span className="w-16 shrink-0 truncate text-xs font-medium text-sky-300">{e.name}</span>
          <span className="flex-1 whitespace-pre-wrap text-sm text-slate-100">{e.text}</span>
          <button
            type="button"
            onClick={() => onPin(`Message from ${e.name}`, e.text)}
            className="hidden text-xs text-slate-500 hover:text-slate-200 group-hover:block"
          >
            pin
          </button>
        </div>
      )
    case 'member.joined':
    case 'member.left':
      return (
        <div className="text-xs text-slate-500">
          {e.name} {e.type === 'member.joined' ? 'joined' : e.reason ? `left (${e.reason})` : 'left'}
        </div>
      )
    case 'pin.added':
      return <div className="text-xs text-amber-300">📌 {e.name} pinned “{e.pin.title}”</div>
    case 'pin.removed':
      return <div className="text-xs text-slate-500">{e.name} unpinned an item</div>
    case 'agent.start':
      return (
        <div className="text-xs text-violet-300">
          {e.name} asked the agent: <span className="text-slate-200">{e.question}</span>
        </div>
      )
    case 'agent.answer':
      return (
        <div className="group rounded-lg border border-violet-900 bg-violet-950/30 p-2" data-testid="room-agent-answer">
          <div className="mb-1 flex items-center gap-2 text-xs text-violet-300">
            Agent · for {e.asked_by}
            {e.outcome && <Badge tone={e.outcome === 'answered' ? 'good' : 'warn'}>{e.outcome}</Badge>}
            {e.visual_context && <Badge tone="info" title={e.visual_context}>used camera context</Badge>}
            <button
              type="button"
              onClick={() => onPin(`Agent: ${e.question}`, e.text)}
              className="ml-auto hidden text-slate-500 hover:text-slate-200 group-hover:block"
            >
              pin
            </button>
          </div>
          <div className="whitespace-pre-wrap text-sm text-slate-100">{e.text}</div>
          {e.sources.length > 0 && (
            <div className="mt-1 text-xs text-slate-400">
              Sources: {e.sources.map((s) => s.document_title || s.doc_id).join(' · ')}
            </div>
          )}
        </div>
      )
    case 'agent.error':
      return <div className="text-xs text-rose-400">Agent error: {e.error}</div>
    case 'visual_context':
      return (
        <div className="text-xs text-teal-300" data-testid="room-visual">
          📷 {e.name}'s camera: {e.labels.slice(0, 3).map((l) => l.label).join(', ')}
        </div>
      )
  }
}

function LiveTurnRow({ t }: { t: LiveTurn }) {
  return (
    <div className="rounded-lg border border-violet-900/60 p-2 text-sm text-slate-300" data-testid="room-live-turn">
      <div className="mb-1 text-xs text-violet-300">Agent is answering {t.askedBy}…</div>
      {t.text || <span className="animate-pulse text-slate-500">thinking</span>}
    </div>
  )
}

export function PinsPanel({ pins, onUnpin }: { pins: Pin[]; onUnpin: (id: string) => void }) {
  return (
    <aside className="flex flex-col gap-2" data-testid="pins">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Pinned evidence ({pins.length})</h3>
      {pins.length === 0 && <p className="text-xs text-slate-500">Pin messages, agent answers or log lines to share them.</p>}
      {pins.map((p) => (
        <div key={p.pin_id} className="rounded-lg border border-slate-800 p-2 text-xs">
          <div className="flex items-center gap-1.5">
            <Badge tone="warn">{p.kind}</Badge>
            <span className="truncate font-medium text-slate-200">{p.title}</span>
            <button type="button" onClick={() => onUnpin(p.pin_id)} className="ml-auto text-slate-500 hover:text-slate-200">
              ✕
            </button>
          </div>
          {p.snippet && <p className="mt-1 line-clamp-3 text-slate-400">{p.snippet}</p>}
          <p className="mt-1 text-slate-600">by {p.by}</p>
        </div>
      ))}
    </aside>
  )
}

export function RoomTimeline({ state, onPin }: { state: RoomState; onPin: (title: string, snippet: string) => void }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => end.current?.scrollIntoView?.({ block: 'end' }), [state.timeline.length, state.liveTurns])
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-3" data-testid="room-timeline">
      {state.timeline.map((e) => (
        <EventRow key={e.seq} e={e} onPin={onPin} />
      ))}
      {Object.values(state.liveTurns).map((t) => (
        <LiveTurnRow key={t.turnId} t={t} />
      ))}
      <div ref={end} />
    </div>
  )
}

const CONN_TONE: Record<RoomConnState, 'good' | 'warn' | 'bad' | 'neutral'> = {
  idle: 'neutral',
  connecting: 'warn',
  open: 'good',
  reconnecting: 'warn',
  closed: 'bad',
}

function Composer({ client, disabled }: { client: RoomClient; disabled: boolean }) {
  const [text, setText] = useState('')
  const typing = useRef(false)
  const setTyping = (active: boolean) => {
    if (typing.current !== active) {
      typing.current = active
      client.typing(active)
    }
  }
  const submit = (asAgent: boolean) => {
    const t = text.trim()
    if (!t) return
    if (asAgent) client.ask(t)
    else client.sendMessage(t)
    setText('')
    setTyping(false)
  }
  return (
    <form
      className="flex gap-2 border-t border-slate-800 p-3"
      onSubmit={(e) => {
        e.preventDefault()
        submit(false)
      }}
    >
      <input
        data-testid="room-input"
        value={text}
        disabled={disabled}
        onChange={(e) => {
          setText(e.target.value)
          setTyping(e.target.value.length > 0)
        }}
        onBlur={() => setTyping(false)}
        onFocus={() => client.focus('composer')}
        placeholder="Message the room…"
        className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-slate-500"
      />
      <button type="submit" disabled={disabled} className="rounded-lg bg-sky-700 px-3 text-sm hover:bg-sky-600 disabled:opacity-40">
        Send
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={() => submit(true)}
        data-testid="room-ask"
        className="rounded-lg bg-violet-700 px-3 text-sm hover:bg-violet-600 disabled:opacity-40"
      >
        Ask agent
      </button>
    </form>
  )
}

export function RoomsView() {
  const [roomId, setRoomId] = useState(roomFromHash)
  const [name, setName] = useState(() => localStorage.getItem('omni.name') ?? '')
  const [draftName, setDraftName] = useState(name)
  const [joinId, setJoinId] = useState('')
  const [title, setTitle] = useState('')
  const [error, setError] = useState<string>()

  useEffect(() => {
    const onHash = () => setRoomId(roomFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const go = (id: string) => {
    location.hash = `room=${id}`
    setRoomId(id)
  }
  const saveName = () => {
    const n = draftName.trim().slice(0, 40)
    if (!n) return
    try {
      localStorage.setItem('omni.name', n)
    } catch {
      /* storage may be unavailable; the name still applies to this tab */
    }
    setName(n)
  }

  if (!name) {
    return (
      <div className="m-auto flex w-80 flex-col gap-2 p-4" data-testid="room-name-form">
        <label className="text-sm text-slate-300">Your name in incident rooms</label>
        <input
          value={draftName}
          onChange={(e) => setDraftName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && saveName()}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        />
        <button type="button" onClick={saveName} className="rounded-lg bg-sky-700 py-2 text-sm">
          Continue
        </button>
      </div>
    )
  }

  const switcher = (
    <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 px-3 py-2 text-xs" data-testid="room-switcher">
      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="New incident title"
        className="w-44 rounded-md border border-slate-700 bg-slate-900 px-2 py-1"
      />
      <button
        type="button"
        data-testid="room-create"
        onClick={() =>
          createRoom(title)
            .then((id) => {
              setTitle('')
              setError(undefined)
              go(id)
            })
            .catch((e: Error) => setError(e.message))
        }
        className="rounded-md bg-sky-800 px-2 py-1 hover:bg-sky-700"
      >
        Create room
      </button>
      <span className="text-slate-600">or</span>
      <input
        value={joinId}
        onChange={(e) => setJoinId(e.target.value.trim().toLowerCase())}
        placeholder="room id"
        className="w-32 rounded-md border border-slate-700 bg-slate-900 px-2 py-1"
      />
      <button
        type="button"
        disabled={!ROOM_RE.test(joinId)}
        onClick={() => go(joinId)}
        className="rounded-md border border-slate-700 px-2 py-1 disabled:opacity-40"
      >
        Join
      </button>
      {error && <span className="text-rose-400">{error}</span>}
    </div>
  )

  if (!roomId) return <div className="flex flex-col">{switcher}<p className="p-4 text-sm text-slate-400">Create an incident room or join one by id. Share the page link to invite others.</p></div>

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {switcher}
      <ActiveRoom key={`${roomId}/${name}`} roomId={roomId} name={name} />
    </div>
  )
}

function ActiveRoom({ roomId, name }: { roomId: string; name: string }) {
  const room = useRoom(roomId, name)
  const s = room.state
  const pin = (t: string, snippet: string) => room.client.pin('note', t, snippet)
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 px-3 py-2">
        <h2 className="text-sm font-semibold">{s.title ?? roomId}</h2>
        <button
          type="button"
          title="Copy a link to this room"
          onClick={() => navigator.clipboard?.writeText(location.href)}
          className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-xs text-slate-300"
        >
          #{roomId}
        </button>
        <Badge tone={CONN_TONE[room.conn]} title={room.detail}>
          {room.conn}
        </Badge>
        <span className="text-xs text-slate-500">seq {s.lastSeq}</span>
        <div className="ml-auto">
          <PresenceBar members={s.members} me={s.session} />
        </div>
      </div>
      {s.visual && (
        <div className="border-b border-slate-800 px-3 py-1 text-xs text-teal-300">
          Camera ({s.visual.by}): {s.visual.labels.slice(0, 3).map((l) => l.label).join(', ')}
        </div>
      )}
      <div className="flex min-h-0 flex-1">
        <div className="flex min-h-0 flex-1 flex-col">
          <RoomTimeline state={s} onPin={pin} />
          <div className="h-4 px-3 text-xs text-slate-500" data-testid="typing">
            {typingLine(s)}
          </div>
          <Composer client={room.client} disabled={room.conn !== 'open'} />
        </div>
        <div className="hidden w-72 overflow-y-auto border-l border-slate-800 p-3 md:block">
          <PinsPanel pins={s.pins} onUnpin={(id) => room.client.unpin(id)} />
        </div>
      </div>
    </div>
  )
}
