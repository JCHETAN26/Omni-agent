import type { ReactNode } from 'react'
import { TOOL_LABEL } from '../lib/labels'
import type { Outcome, ToolName } from '../lib/protocol'

type Tone = 'neutral' | 'good' | 'warn' | 'bad' | 'info'

const TONES: Record<Tone, string> = {
  neutral: 'bg-slate-800 text-slate-300 ring-slate-700',
  good: 'bg-emerald-950 text-emerald-300 ring-emerald-800',
  warn: 'bg-amber-950 text-amber-300 ring-amber-800',
  bad: 'bg-rose-950 text-rose-300 ring-rose-800',
  info: 'bg-sky-950 text-sky-300 ring-sky-800',
}

export function Badge({
  tone = 'neutral',
  children,
  title,
  testId,
}: {
  tone?: Tone
  children: ReactNode
  title?: string
  testId?: string
}) {
  return (
    <span
      title={title}
      data-testid={testId}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${TONES[tone]}`}
    >
      {children}
    </span>
  )
}

export function ToolBadge({ tool }: { tool?: ToolName }) {
  if (!tool) return null
  return (
    <Badge tone="info" testId="tool-badge" title="Tool chosen by the router">
      {TOOL_LABEL[tool]}
    </Badge>
  )
}

const OUTCOME_TONE: Record<Outcome, Tone> = {
  answered: 'good',
  abstained: 'warn',
  insufficient_context: 'warn',
  ungrounded: 'bad',
}

export function OutcomeBadge({ outcome }: { outcome?: Outcome }) {
  if (!outcome) return null
  return (
    <Badge tone={OUTCOME_TONE[outcome]} testId="outcome-badge">
      {outcome.replace('_', ' ')}
    </Badge>
  )
}

export function IntegrityBadge({ integrity, detail }: { integrity: 'pending' | 'ok' | 'failed'; detail?: string }) {
  if (integrity === 'pending') return null
  return (
    <Badge tone={integrity === 'ok' ? 'good' : 'bad'} title={detail} testId="integrity-badge">
      {integrity === 'ok' ? 'stream verified' : 'stream corrupted'}
    </Badge>
  )
}

export function ScoreBar({ score, ok, label }: { score: number; ok: boolean; label?: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100)
  return (
    <div className="flex items-center gap-2" title={label}>
      <div className="h-1.5 w-28 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full ${ok ? 'bg-emerald-500' : 'bg-amber-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="tabular-nums text-xs text-slate-400">{score.toFixed(2)}</span>
    </div>
  )
}

export function Section({ title, children, testId }: { title: string; children: ReactNode; testId?: string }) {
  return (
    <section className="border-b border-slate-800 px-4 py-3" data-testid={testId}>
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{title}</h3>
      {children}
    </section>
  )
}

export function Stat({ label, value, testId }: { label: string; value: ReactNode; testId?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5 text-sm">
      <span className="text-slate-400">{label}</span>
      <span className="tabular-nums text-slate-200" data-testid={testId}>
        {value}
      </span>
    </div>
  )
}
