import { useEffect, useRef, useState } from 'react'
import { finalAttempt, summarize, type HistoryMessage, type Turn } from '../lib/conversation'
import type { Mode } from '../lib/protocol'
import { Badge, IntegrityBadge, OutcomeBadge, ToolBadge } from './ui'

const EXAMPLES: Record<Mode, string[]> = {
  agent: [
    'my storage box keeps spitting out EIO5, what is going on?',
    'and which firmware version is affected?',
    'show me recent errors on storage-07',
    'how do I rotate the kafka broker TLS keystore?',
    "if the queue depth is 64 what's 64/2?",
  ],
  stream: ['EIO-5 controller timeout', 'PSU redundancy lost', 'pods slow DNS ndots'],
}

function AssistantMessage({ turn, selected, onSelect }: { turn: Turn; selected: boolean; onSelect: () => void }) {
  const final = finalAttempt(turn)
  const discarded = turn.attempts.filter((a) => a.discarded)
  const s = summarize(turn)
  const streaming = turn.status === 'streaming' || turn.status === 'sent'
  return (
    <button
      type="button"
      onClick={onSelect}
      data-testid="assistant-message"
      data-turn-id={turn.id}
      className={`w-full rounded-xl border px-4 py-3 text-left transition-colors ${
        selected ? 'border-sky-700 bg-slate-900' : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
      }`}
    >
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <ToolBadge tool={s.tool} />
        <OutcomeBadge outcome={s.outcome} />
        {s.retries > 0 && (
          <Badge tone="warn" testId="self-corrected-badge">
            self-corrected ×{s.retries}
          </Badge>
        )}
        <IntegrityBadge integrity={turn.integrity} detail={turn.integrityDetail} />
        {streaming && <Badge tone="info">streaming…</Badge>}
      </div>

      {discarded.map((a) => (
        <details key={a.n} className="mb-2 rounded-lg bg-rose-950/30 px-3 py-2 text-sm" data-testid="discarded-attempt">
          <summary className="cursor-pointer text-rose-300">
            Attempt {a.n} discarded: {a.discarded}
          </summary>
          <p className="mt-1 whitespace-pre-wrap text-rose-200/70 line-through">{a.text}</p>
        </details>
      ))}

      {turn.error ? (
        <p className="text-sm text-rose-300">{turn.error}</p>
      ) : (
        <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-slate-100" data-testid="answer-text">
          {final?.text}
          {streaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-sky-400 align-middle" />}
        </p>
      )}

      {turn.warning && <p className="mt-2 text-sm text-amber-300">⚠ {turn.warning}</p>}

      {final && final.sources.length > 0 && (
        <ol className="mt-3 space-y-0.5 text-xs text-slate-400">
          {final.sources.slice(0, 3).map((src, i) => (
            <li key={src.chunk_id}>
              [{i + 1}] {src.section_heading}
            </li>
          ))}
        </ol>
      )}
    </button>
  )
}

export function ChatPanel({
  mode,
  turns,
  history,
  canSend,
  selectedId,
  onSelect,
  onAsk,
}: {
  mode: Mode
  turns: Turn[]
  history: HistoryMessage[]
  canSend: boolean
  selectedId?: string
  onSelect: (id: string) => void
  onAsk: (q: string) => boolean
}) {
  const [draft, setDraft] = useState('')
  const end = useRef<HTMLDivElement>(null)
  const lastText = turns.at(-1)?.attempts.at(-1)?.text.length ?? 0

  useEffect(() => {
    end.current?.scrollIntoView({ block: 'end' })
  }, [turns.length, lastText])

  const submit = (text: string) => {
    if (onAsk(text)) setDraft('')
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-6 md:px-8" data-testid="chat-log">
        {history.length > 0 && (
          <div className="space-y-2 rounded-xl border border-dashed border-slate-800 p-3" data-testid="history">
            <p className="text-xs uppercase tracking-wider text-slate-500">
              Restored conversation ({history.length / 2} earlier turns)
            </p>
            {history.map((m, i) => (
              <p key={i} className="truncate text-sm text-slate-400">
                <span className="text-slate-500">{m.role === 'user' ? 'You' : 'Agent'}:</span> {m.content}
              </p>
            ))}
          </div>
        )}

        {turns.length === 0 && history.length === 0 && (
          <div className="mx-auto max-w-xl pt-10 text-center">
            <h2 className="text-lg font-semibold text-slate-200">Ask OmniAgent</h2>
            <p className="mt-1 text-sm text-slate-400">
              {mode === 'agent'
                ? 'Runbooks, live logs and quick maths, with self-correcting retrieval.'
                : 'Plain hybrid retrieval stream (no agent).'}
            </p>
          </div>
        )}

        {turns.map((t) => (
          <div key={t.id} className="space-y-2">
            <div className="flex justify-end">
              <p className="max-w-[85%] rounded-xl bg-sky-900/60 px-4 py-2 text-[15px] text-sky-50" data-testid="user-message">
                {t.question}
              </p>
            </div>
            <AssistantMessage turn={t} selected={t.id === selectedId} onSelect={() => onSelect(t.id)} />
          </div>
        ))}
        <div ref={end} />
      </div>

      <div className="border-t border-slate-800 px-4 py-3 md:px-8">
        <div className="mb-2 flex flex-wrap gap-1.5">
          {EXAMPLES[mode].map((q) => (
            <button
              key={q}
              type="button"
              disabled={!canSend}
              onClick={() => submit(q)}
              className="rounded-full border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:border-sky-700 hover:text-sky-200 disabled:opacity-40"
            >
              {q}
            </button>
          ))}
        </div>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            submit(draft)
          }}
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={canSend ? 'Describe the problem…' : 'Connecting…'}
            aria-label="Question"
            data-testid="question-input"
            className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-[15px] text-slate-100 placeholder:text-slate-500 focus:border-sky-600 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!canSend || !draft.trim()}
            data-testid="send-button"
            className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
