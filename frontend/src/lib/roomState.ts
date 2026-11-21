// Pure room state: apply server events in order, React-agnostic (tested in roomState.test.ts).
//
// Sequenced events are kept once each, in seq order: anything at or below
// `lastSeq` is a duplicate (e.g. replayed after a reconnect) and ignored; a seq
// above lastSeq + 1 means events were missed, so `gap` is set and the client
// reconnects with last_seq to fetch them. Ephemeral events update presence,
// typing and the live agent token streams.

import type { Label, Member, Pin, RoomServerEvent, SequencedEvent, Snapshot } from './roomProtocol'
import { isSequenced } from './roomProtocol'

export interface LiveTurn {
  turnId: string
  question: string
  askedBy: string
  text: string
  nextIndex: number
}

export interface RoomState {
  roomId?: string
  session?: string
  name?: string
  title?: string
  members: Record<string, Member>
  pins: Pin[]
  timeline: SequencedEvent[]
  lastSeq: number
  gap: boolean
  typing: Record<string, string> // session -> name
  liveTurns: Record<string, LiveTurn>
  visual?: { by: string; labels: Label[]; at: number }
  duplicatesIgnored: number
}

export const emptyRoom = (): RoomState => ({
  members: {},
  pins: [],
  timeline: [],
  lastSeq: 0,
  gap: false,
  typing: {},
  liveTurns: {},
  duplicatesIgnored: 0,
})

const MAX_TIMELINE = 500

function applySnapshot(s: RoomState, e: Snapshot): RoomState {
  const members: Record<string, Member> = {}
  for (const m of e.members) members[m.session] = m
  const fresh = e.resync || s.roomId !== e.room_id
  return {
    ...(fresh ? emptyRoom() : s),
    roomId: e.room_id,
    session: e.session,
    name: e.name,
    title: e.meta?.title,
    members,
    pins: e.pins,
    gap: false,
    typing: {},
  }
}

function applySequenced(s: RoomState, e: SequencedEvent): RoomState {
  if (e.seq <= s.lastSeq) return { ...s, duplicatesIgnored: s.duplicatesIgnored + 1 }
  if (s.lastSeq > 0 && e.seq > s.lastSeq + 1) return { ...s, gap: true }
  const next: RoomState = { ...s, lastSeq: e.seq, timeline: [...s.timeline, e].slice(-MAX_TIMELINE) }
  switch (e.type) {
    case 'member.joined':
      next.members = { ...s.members, [e.session]: { session: e.session, name: e.name, focus: 'timeline' } }
      break
    case 'member.left': {
      const { [e.session]: _left, ...rest } = s.members
      next.members = rest
      const { [e.session]: _t, ...typing } = s.typing
      next.typing = typing
      break
    }
    case 'pin.added':
      next.pins = [...s.pins.filter((p) => p.pin_id !== e.pin.pin_id), e.pin]
      break
    case 'pin.removed':
      next.pins = s.pins.filter((p) => p.pin_id !== e.pin_id)
      break
    case 'agent.start':
      next.liveTurns = {
        ...s.liveTurns,
        [e.turn_id]: { turnId: e.turn_id, question: e.question, askedBy: e.name, text: '', nextIndex: 0 },
      }
      break
    case 'agent.answer':
    case 'agent.error': {
      const { [e.turn_id]: _done, ...live } = s.liveTurns
      next.liveTurns = live
      break
    }
    case 'visual_context':
      next.visual = { by: e.name, labels: e.labels, at: e.at }
      break
  }
  return next
}

export function applyRoomEvent(s: RoomState, e: RoomServerEvent): RoomState {
  if (e.type === 'room.snapshot') return applySnapshot(s, e)
  if (isSequenced(e)) return applySequenced(s, e)
  switch (e.type) {
    case 'presence.typing': {
      if (e.session === s.session) return s
      const { [e.session]: _old, ...rest } = s.typing
      return { ...s, typing: e.active ? { ...rest, [e.session]: e.name } : rest }
    }
    case 'presence.focus': {
      const m = s.members[e.session]
      return m ? { ...s, members: { ...s.members, [e.session]: { ...m, focus: e.pane } } } : s
    }
    case 'agent.token': {
      const t = s.liveTurns[e.turn_id]
      // Tokens arrive in order per turn; anything else (a late joiner mid-turn) is dropped.
      if (!t || e.index !== t.nextIndex) return s
      return {
        ...s,
        liveTurns: { ...s.liveTurns, [e.turn_id]: { ...t, text: t.text + e.text, nextIndex: t.nextIndex + 1 } },
      }
    }
    default:
      return s
  }
}

/** Typing indicator text, e.g. "Bo is typing…" / "Bo and Cy are typing…". */
export function typingLine(s: RoomState): string {
  const names = Object.values(s.typing)
  if (names.length === 0) return ''
  if (names.length === 1) return `${names[0]} is typing…`
  if (names.length === 2) return `${names[0]} and ${names[1]} are typing…`
  return `${names.length} people are typing…`
}
