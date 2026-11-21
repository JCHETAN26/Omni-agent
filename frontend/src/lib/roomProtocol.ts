// Wire protocol of /ws/rooms/{room_id} (see backend/api/rooms.py).
//
// Sequenced events carry `seq` (strictly +1 per room, stored and replayable);
// ephemeral ones (typing, focus, agent tokens, acks) do not.

export type PinKind = 'search' | 'log' | 'citation' | 'note'

export interface Pin {
  pin_id: string
  kind: PinKind
  title: string
  ref: string
  snippet: string
  by: string
  at: number
}

export interface Member {
  session: string
  name: string
  focus?: string
  joined_at?: number
  last_seen?: number
}

export interface Label {
  label: string
  score: number
}

export interface AgentSource {
  chunk_id?: string
  doc_id?: string
  document_title?: string
  section_heading?: string
}

interface Who {
  session: string
  name: string
}

export type SequencedEvent =
  | ({ seq: number; type: 'member.joined' | 'member.left'; at?: number; reason?: string } & Who)
  | ({ seq: number; type: 'message'; text: string; client_msg_id: string; at: number; client_ts?: number } & Who)
  | ({ seq: number; type: 'pin.added'; pin: Pin } & Who)
  | ({ seq: number; type: 'pin.removed'; pin_id: string } & Who)
  | ({ seq: number; type: 'agent.start'; turn_id: string; question: string; at: number } & Who)
  | {
      seq: number
      type: 'agent.answer'
      turn_id: string
      question: string
      asked_by: string
      text: string
      outcome?: string
      tool?: string
      sources: AgentSource[]
      visual_context?: string | null
      at: number
    }
  | ({ seq: number; type: 'agent.error'; turn_id: string; error: string } & Who)
  | ({ seq: number; type: 'visual_context'; labels: Label[]; at: number } & Who)

export type EphemeralEvent =
  | ({ type: 'presence.typing'; active: boolean } & Who)
  | ({ type: 'presence.focus'; pane: string } & Who)
  | { type: 'agent.token'; turn_id: string; index: number; text: string }
  | { type: 'ack'; client_msg_id: string; acked_seq?: number; duplicate?: boolean }
  | { type: 'pong'; ts?: number }
  | { type: 'rate_limited'; retry_after_s: number }
  | { type: 'error'; error: string }

export interface Snapshot {
  type: 'room.snapshot'
  room_id: string
  session: string
  name: string
  meta: { title?: string; created_at?: string }
  members: Member[]
  pins: Pin[]
  replay: number
  resync: boolean
  heartbeat_interval_s: number
}

export type RoomServerEvent = Snapshot | SequencedEvent | EphemeralEvent

export function isSequenced(e: RoomServerEvent): e is SequencedEvent {
  return typeof (e as { seq?: unknown }).seq === 'number'
}
