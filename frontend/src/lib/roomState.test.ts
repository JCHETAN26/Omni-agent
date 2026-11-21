import { describe, expect, it } from 'vitest'
import type { RoomServerEvent, SequencedEvent, Snapshot } from './roomProtocol'
import { applyRoomEvent, emptyRoom, typingLine, type RoomState } from './roomState'

const snap = (over: Partial<Snapshot> = {}): Snapshot => ({
  type: 'room.snapshot',
  room_id: 'r1abc',
  session: 'me',
  name: 'Ana',
  meta: { title: 'storage-07' },
  members: [{ session: 'me', name: 'Ana' }],
  pins: [],
  replay: 0,
  resync: false,
  heartbeat_interval_s: 15,
  ...over,
})

const msg = (seq: number, text = `m${seq}`, session = 'bo'): SequencedEvent => ({
  seq,
  type: 'message',
  session,
  name: 'Bo',
  text,
  client_msg_id: `c${seq}`,
  at: seq,
})

const apply = (events: RoomServerEvent[], s: RoomState = emptyRoom()) => events.reduce(applyRoomEvent, s)

describe('roomState', () => {
  it('keeps sequenced events once each, in order, and ignores replays', () => {
    const s = apply([snap(), msg(1), msg(2), msg(2), msg(1), msg(3)])
    expect(s.timeline.map((e) => e.seq)).toEqual([1, 2, 3])
    expect(s.lastSeq).toBe(3)
    expect(s.duplicatesIgnored).toBe(2)
    expect(s.gap).toBe(false)
  })

  it('flags a gap instead of applying events past it', () => {
    const s = apply([snap(), msg(1), msg(4)])
    expect(s.gap).toBe(true)
    expect(s.lastSeq).toBe(1) // reconnecting with last_seq=1 replays 2..4
    const healed = apply([snap(), msg(2), msg(3), msg(4)], s)
    expect(healed.gap).toBe(false)
    expect(healed.timeline.map((e) => e.seq)).toEqual([1, 2, 3, 4])
  })

  it('starts at whatever seq the first replayed event has (late joiner)', () => {
    const s = apply([snap(), msg(151), msg(152)])
    expect(s.gap).toBe(false)
    expect(s.timeline).toHaveLength(2)
  })

  it('a resync snapshot starts the timeline over', () => {
    const s = apply([snap(), msg(1), msg(2), snap({ resync: true }), msg(900)])
    expect(s.timeline.map((e) => e.seq)).toEqual([900])
  })

  it('tracks presence, typing and focus', () => {
    let s = apply([
      snap(),
      { seq: 1, type: 'member.joined', session: 'bo', name: 'Bo' },
      { type: 'presence.typing', session: 'bo', name: 'Bo', active: true },
      { type: 'presence.focus', session: 'bo', name: 'Bo', pane: 'pins' },
    ])
    expect(Object.keys(s.members)).toEqual(['me', 'bo'])
    expect(s.members.bo?.focus).toBe('pins')
    expect(typingLine(s)).toBe('Bo is typing…')
    s = apply([{ seq: 2, type: 'member.left', session: 'bo', name: 'Bo' }], s)
    expect(s.members.bo).toBeUndefined()
    expect(typingLine(s)).toBe('')
    // My own typing echo is not shown to me.
    expect(typingLine(apply([{ type: 'presence.typing', session: 'me', name: 'Ana', active: true }], s))).toBe('')
  })

  it('applies pins and streams agent tokens until the stored answer lands', () => {
    const pin = { pin_id: 'p1', kind: 'log' as const, title: 'kernel log', ref: '', snippet: 'EIO', by: 'Bo', at: 1 }
    let s = apply([
      snap(),
      { seq: 1, type: 'pin.added', session: 'bo', name: 'Bo', pin },
      { seq: 2, type: 'agent.start', session: 'bo', name: 'Bo', turn_id: 't1', question: 'why?', at: 2 },
      { type: 'agent.token', turn_id: 't1', index: 0, text: 'Bad ' },
      { type: 'agent.token', turn_id: 't1', index: 2, text: 'skipped' }, // out of order: dropped
      { type: 'agent.token', turn_id: 't1', index: 1, text: 'disk.' },
    ])
    expect(s.pins.map((p) => p.pin_id)).toEqual(['p1'])
    expect(s.liveTurns.t1?.text).toBe('Bad disk.')
    s = apply(
      [
        {
          seq: 3,
          type: 'agent.answer',
          turn_id: 't1',
          question: 'why?',
          asked_by: 'Bo',
          text: 'Bad disk.',
          sources: [],
          at: 3,
        },
        { seq: 4, type: 'pin.removed', session: 'bo', name: 'Bo', pin_id: 'p1' },
      ],
      s,
    )
    expect(s.liveTurns).toEqual({})
    expect(s.pins).toEqual([])
    expect(s.timeline.at(-2)?.type).toBe('agent.answer')
  })

  it('remembers the latest camera context', () => {
    const s = apply([
      snap(),
      { seq: 1, type: 'visual_context', session: 'bo', name: 'Bo', labels: [{ label: 'server rack', score: 9 }], at: 5 },
    ])
    expect(s.visual).toEqual({ by: 'Bo', labels: [{ label: 'server rack', score: 9 }], at: 5 })
  })
})
