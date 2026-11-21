import { useState } from 'react'
import { ChatPanel } from './components/ChatPanel'
import { Inspector } from './components/Inspector'
import { RoomsView } from './components/RoomsView'
import { useGateway } from './hooks/useGateway'
import { useHealth } from './hooks/useHealth'
import type { ConnState } from './lib/gatewayClient'
import type { Mode } from './lib/protocol'

const STATE_STYLE: Record<ConnState, string> = {
  idle: 'bg-slate-500',
  connecting: 'bg-amber-400 animate-pulse',
  open: 'bg-emerald-400',
  reconnecting: 'bg-amber-400 animate-pulse',
  closed: 'bg-rose-500',
}

type View = 'assistant' | 'rooms'

export default function App() {
  const [view, setView] = useState<View>(() => (location.hash.includes('room=') ? 'rooms' : 'assistant'))
  return (
    <div className="flex h-dvh flex-col bg-slate-950 text-slate-100">
      <nav className="flex items-center gap-1 border-b border-slate-800 px-3 pt-2 text-sm" role="tablist">
        {(['assistant', 'rooms'] as const).map((v) => (
          <button
            key={v}
            type="button"
            role="tab"
            aria-selected={view === v}
            data-testid={`view-${v}`}
            onClick={() => setView(v)}
            className={`rounded-t-md px-3 py-1.5 ${view === v ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'}`}
          >
            {v === 'assistant' ? 'Assistant' : 'Incident rooms'}
          </button>
        ))}
      </nav>
      {view === 'assistant' ? <Assistant /> : <RoomsView />}
    </div>
  )
}

function Assistant() {
  const [mode, setMode] = useState<Mode>('agent')
  const gw = useGateway(mode)
  const { health, error: healthError } = useHealth()
  const { turns } = gw.conversation
  // Follow the newest turn, unless the user picked one since the last turn arrived.
  const [pinned, setPinned] = useState<{ id: string; turnCount: number }>()
  const pinnedTurn =
    pinned && pinned.turnCount === turns.length ? turns.find((t) => t.id === pinned.id) : undefined
  const selected = pinnedTurn ?? turns.at(-1)
  const select = (id: string) => setPinned({ id, turnCount: turns.length })

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex flex-wrap items-center gap-3 border-b border-slate-800 px-4 py-2.5">
        <h1 className="text-base font-semibold tracking-tight">OmniAgent</h1>
        <span className="flex items-center gap-1.5 text-xs text-slate-400" data-testid="connection-status" title={gw.detail}>
          <span className={`h-2 w-2 rounded-full ${STATE_STYLE[gw.state]}`} />
          {gw.state === 'open' ? 'Connected' : gw.state}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <div className="flex rounded-lg border border-slate-700 p-0.5 text-xs" role="tablist">
            {(['agent', 'stream'] as const).map((m) => (
              <button
                key={m}
                type="button"
                role="tab"
                aria-selected={mode === m}
                data-testid={`mode-${m}`}
                onClick={() => setMode(m)}
                className={`rounded-md px-2.5 py-1 ${mode === m ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'}`}
              >
                {m === 'agent' ? 'Agent /ws/agent' : 'Stream /ws/stream'}
              </button>
            ))}
          </div>
          {mode === 'agent' && (
            <button
              type="button"
              onClick={gw.newConversation}
              data-testid="new-conversation"
              className="rounded-lg border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:border-slate-500"
            >
              New conversation
            </button>
          )}
        </div>
      </header>

      {gw.conversation.notices.length > 0 && (
        <div className="border-b border-amber-900 bg-amber-950/40 px-4 py-1.5 text-xs text-amber-300" data-testid="notices">
          {gw.conversation.notices.at(-1)}
        </div>
      )}

      <main className="flex min-h-0 flex-1 flex-col md:flex-row">
        <ChatPanel
          mode={mode}
          turns={turns}
          history={gw.conversation.history}
          canSend={gw.state === 'open'}
          selectedId={selected?.id}
          onSelect={select}
          onAsk={gw.ask}
        />
        <Inspector
          turn={selected}
          state={gw.state}
          stats={gw.stats}
          liveTokenRate={gw.liveTokenRate}
          sessionId={gw.conversation.sessionId}
          threadId={mode === 'agent' ? gw.conversation.threadId : undefined}
          reasoner={gw.conversation.reasoner}
          health={health}
          healthError={healthError}
        />
      </main>
    </div>
  )
}
