import { useEffect, useMemo, useState } from 'react'
import { RoomClient, type RoomConnState } from '../lib/roomClient'
import { emptyRoom, type RoomState } from '../lib/roomState'

const wsBase = () => `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`

/** One live room connection. Mount it keyed by room and name: a new key is a new client. */
export function useRoom(roomId: string, name: string) {
  const [state, setState] = useState<RoomState>(emptyRoom)
  const [conn, setConn] = useState<RoomConnState>('idle')
  const [detail, setDetail] = useState<string>()
  const client = useMemo(
    () =>
      new RoomClient({
        baseUrl: wsBase,
        roomId,
        name,
        onState: setState,
        onConn: (s, d) => {
          setConn(s)
          setDetail(d)
        },
      }),
    [roomId, name],
  )
  useEffect(() => {
    client.connect()
    return () => client.close()
  }, [client])
  return { state, conn, detail, client }
}
