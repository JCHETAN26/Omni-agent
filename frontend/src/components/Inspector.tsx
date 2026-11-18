import { finalAttempt, summarize, tokenRate, type Turn } from '../lib/conversation'
import type { ClientStats, ConnState } from '../lib/gatewayClient'
import type { Health } from '../hooks/useHealth'
import { Badge, IntegrityBadge, OutcomeBadge, ScoreBar, Section, Stat, ToolBadge } from './ui'

function ms(v?: number | null): string {
  return v === undefined || v === null ? '—' : `${Math.round(v)} ms`
}

function stepLine(s: Turn['steps'][number]): string {
  switch (s.node) {
    case 'contextualize':
      return s.changed ? `standalone: “${s.standalone}”` : 'already standalone'
    case 'route':
      return `→ ${s.tool}${s.argument && s.tool !== 'knowledge_base' ? ` (${s.argument})` : ''}`
    case 'retrieve':
      return `#${s.attempt} “${s.query}” → ${s.hits?.length ?? 0} hits`
    case 'grade_documents':
      return `${s.relevant ? 'relevant' : 'not relevant'} (${(s.score ?? 0).toFixed(2)})`
    case 'rewrite_query':
      return `[${s.strategy}] “${s.query}”`
    case 'inspect_logs':
      return `${s.lines} lines${s.host ? ` from ${s.host}` : ''}`
    case 'calculate':
      return s.result ?? ''
    case 'generate':
      return `attempt ${s.attempt}, ${s.chars} chars${s.abstained ? ', abstained' : ''}`
    case 'grade_generation':
      return `${s.grounded ? 'grounded' : 'NOT grounded'} (${(s.score ?? 0).toFixed(2)})${s.will_retry ? ' → retry' : ''}`
    case 'finalize':
      return `${s.outcome}, ${s.retries} retries`
    default:
      return ''
  }
}

function TurnInspector({ turn }: { turn: Turn }) {
  const s = summarize(turn)
  const final = finalAttempt(turn)
  const rate = tokenRate(final)
  const ttft = turn.end?.first_token_ms ?? (final?.firstTokenAt && turn.startedAt ? final.firstTokenAt - turn.startedAt : undefined)
  const t0 = turn.startedAt ?? turn.sentAt
  return (
    <>
      <Section title="Response" testId="inspector-response">
        <div className="mb-2 flex flex-wrap gap-1.5">
          <ToolBadge tool={s.tool} />
          <OutcomeBadge outcome={s.outcome} />
          <IntegrityBadge integrity={turn.integrity} detail={turn.integrityDetail} />
        </div>
        <Stat label="Time to first token" value={ms(ttft)} />
        <Stat label="Total latency" value={ms(turn.end?.latency_ms)} />
        <Stat label="Token rate" value={rate ? `${rate.toFixed(0)} tok/s` : '—'} testId="token-rate" />
        <Stat label="Tokens (final attempt)" value={final?.tokens ?? 0} />
        <Stat label="Attempts / retries" value={`${turn.attempts.length} / ${s.retries}`} testId="retries" />
        {turn.traceId && (
          <Stat label="Trace" value={<code className="text-xs text-slate-400">{turn.traceId.slice(0, 16)}…</code>} />
        )}
      </Section>

      {s.tool && (
        <Section title="Tool routing" testId="inspector-routing">
          <p className="text-sm text-slate-300">{s.routeReason}</p>
          {s.fallbacks > 0 && (
            <p className="mt-1 text-xs text-amber-300">{s.fallbacks} step(s) used the heuristic fallback</p>
          )}
        </Section>
      )}

      {s.relevance.length > 0 && (
        <Section title="Retrieval relevance" testId="inspector-relevance">
          <ul className="space-y-2">
            {s.relevance.map((r, i) => (
              <li key={i} data-testid="relevance-grade">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-xs text-slate-400" title={r.query}>
                    #{i + 1} {r.query}
                  </span>
                  <ScoreBar score={r.score} ok={r.relevant} label={r.reason} />
                </div>
                <p className="mt-0.5 text-xs text-slate-500">{r.reason}</p>
              </li>
            ))}
          </ul>
          {s.rewrites.length > 0 && (
            <ul className="mt-3 space-y-1 border-t border-slate-800 pt-2" data-testid="rewrites">
              {s.rewrites.map((r, i) => (
                <li key={i} className="text-xs text-slate-300">
                  <Badge tone="warn">rewrite {i + 1}</Badge> <span className="text-slate-500">{r.strategy}</span> “{r.query}”
                </li>
              ))}
            </ul>
          )}
        </Section>
      )}

      {s.grounding.length > 0 && (
        <Section title="Grounding verification" testId="inspector-grounding">
          <ul className="space-y-2">
            {s.grounding.map((g, i) => (
              <li key={i} className="flex items-center justify-between gap-2">
                <Badge tone={g.grounded ? 'good' : 'bad'} testId="grounding-badge" title={g.reason}>
                  {g.grounded ? 'grounded' : 'not grounded'}
                </Badge>
                <ScoreBar score={g.score} ok={g.grounded} label={g.reason} />
              </li>
            ))}
          </ul>
        </Section>
      )}

      {turn.steps.length > 0 && (
        <Section title="Reasoning timeline" testId="inspector-timeline">
          <ol className="space-y-1">
            {turn.steps.map((step, i) => (
              <li key={i} className="grid grid-cols-[3.5rem_7.5rem_1fr] gap-2 text-xs" data-testid="timeline-step">
                <span className="tabular-nums text-slate-500">+{Math.round(step.at - t0)}ms</span>
                <span className="font-medium text-slate-300">
                  {step.node}
                  {step.fallback ? '*' : ''}
                </span>
                <span className="truncate text-slate-400" title={stepLine(step)}>
                  {stepLine(step)}
                </span>
              </li>
            ))}
          </ol>
        </Section>
      )}
    </>
  )
}

export function Inspector({
  turn,
  state,
  stats,
  liveTokenRate,
  sessionId,
  threadId,
  reasoner,
  health,
  healthError,
}: {
  turn?: Turn
  state: ConnState
  stats: ClientStats
  liveTokenRate: number
  sessionId?: string
  threadId?: string
  reasoner?: string
  health?: Health
  healthError?: string
}) {
  return (
    <aside className="flex min-h-0 flex-col overflow-y-auto border-t border-slate-800 bg-slate-950 md:w-[380px] md:border-l md:border-t-0" data-testid="inspector">
      {turn ? (
        <TurnInspector turn={turn} />
      ) : (
        <Section title="Inspector">
          <p className="text-sm text-slate-500">Send a question to see routing, relevance and grounding.</p>
        </Section>
      )}

      <Section title="Connection" testId="inspector-connection">
        <Stat label="State" value={state} testId="conn-state" />
        <Stat label="Live token rate" value={`${liveTokenRate.toFixed(1)} tok/s`} />
        <Stat label="Frames in / out" value={`${stats.framesIn} / ${stats.framesOut}`} />
        <Stat label="Sequence gaps" value={stats.seqGaps} testId="seq-gaps" />
        <Stat label="Streams verified / failed" value={`${stats.integrityOk} / ${stats.integrityFailures}`} testId="streams-verified" />
        <Stat label="Reconnects" value={stats.reconnects} />
        {reasoner && <Stat label="Reasoner" value={reasoner} />}
        {sessionId && <Stat label="Session" value={<code className="text-xs">{sessionId.slice(0, 12)}</code>} />}
        {threadId && <Stat label="Thread" value={<code className="text-xs" data-testid="thread-id">{threadId.slice(0, 12)}</code>} />}
      </Section>

      <Section title="Backend health" testId="inspector-health">
        {healthError && <p className="text-sm text-rose-300">unreachable: {healthError}</p>}
        {health && (
          <>
            <Stat label="Event-loop lag p99" value={ms(health.event_loop_lag.p99_ms)} />
            <Stat label="WebSocket sessions" value={`${health.websocket.active} / ${health.websocket.max}`} />
            <Stat label="Inference" value={health.inference_ready ? `${health.backends.inference} ✓` : 'not ready'} />
            <Stat label="Dense / sparse" value={`${health.backends.dense} / ${health.backends.sparse}`} />
            {health.agent && <Stat label="Memory" value={health.agent.memory} />}
          </>
        )}
      </Section>
    </aside>
  )
}
