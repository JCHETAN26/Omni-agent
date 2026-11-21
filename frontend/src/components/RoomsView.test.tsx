import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { emptyRoom, type RoomState } from '../lib/roomState'
import { PinsPanel, PresenceBar, RoomTimeline } from './RoomsView'

describe('room components', () => {
  it('shows who is here, marking me', () => {
    const html = renderToStaticMarkup(
      <PresenceBar
        me="s1"
        members={{ s1: { session: 's1', name: 'Ana Lee' }, s2: { session: 's2', name: 'Bo', focus: 'pins' } }}
      />,
    )
    expect(html).toContain('AL')
    expect(html).toContain('Bo')
    expect(html).toContain('· pins')
    expect(html).toContain('2 here')
  })

  it('renders the shared timeline, a live agent turn and pins', () => {
    const state: RoomState = {
      ...emptyRoom(),
      lastSeq: 3,
      timeline: [
        { seq: 1, type: 'message', session: 's2', name: 'Bo', text: 'disk 3 is throwing EIO', client_msg_id: 'a', at: 1 },
        { seq: 2, type: 'visual_context', session: 's2', name: 'Bo', labels: [{ label: 'server rack', score: 9 }], at: 2 },
        {
          seq: 3,
          type: 'agent.answer',
          turn_id: 't0',
          question: 'q',
          asked_by: 'Bo',
          text: 'Replace the drive.',
          outcome: 'answered',
          sources: [{ document_title: 'Storage runbook' }],
          visual_context: 'frame #1 shows: server rack',
          at: 3,
        },
      ],
      liveTurns: { t1: { turnId: 't1', question: 'why?', askedBy: 'Ana', text: 'Because', nextIndex: 1 } },
    }
    const html = renderToStaticMarkup(<RoomTimeline state={state} onPin={() => {}} />)
    expect(html).toContain('disk 3 is throwing EIO')
    expect(html).toContain('server rack')
    expect(html).toContain('Replace the drive.')
    expect(html).toContain('Storage runbook')
    expect(html).toContain('used camera context')
    expect(html).toContain('Agent is answering Ana')

    const pins = renderToStaticMarkup(
      <PinsPanel pins={[{ pin_id: 'p', kind: 'log', title: 'kernel log', ref: '', snippet: 'EIO on sdc', by: 'Bo', at: 1 }]} onUnpin={() => {}} />,
    )
    expect(pins).toContain('Pinned evidence (1)')
    expect(pins).toContain('EIO on sdc')
  })
})
