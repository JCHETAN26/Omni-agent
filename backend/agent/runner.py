"""Runs one agent turn and translates graph stream events into client events.

Event sequence for one request id (all frames also carry `seq` on WebSocket):

    response.start            {id, thread_id, trace_id}
    agent.step*               {node, ...decision details}
    response.sources          {attempt, sources}            (knowledge_base answers)
    response.token*           {attempt, index, text}
    agent.generation.discarded {attempt, reason}            (self-correction)
    ... (a new attempt restarts token indices at 0) ...
    response.end              {attempt, tokens, sha256, outcome, tool, retries,
                               relevance, grounding, rewrites, latency_ms}
or  response.error

`response.end.tokens` / `sha256` cover the FINAL attempt's tokens only. Clients
replace the visible answer when a new attempt starts.
"""

from __future__ import annotations

import logging
import time
import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver
from opentelemetry import context as otel_context
from opentelemetry.trace import Link, SpanKind, get_current_span

from backend.agent.llm import ChatModel, chat_model_from_settings
from backend.agent.memory import checkpointer_from_settings
from backend.agent.reasoning import HeuristicReasoner, LLMReasoner, Reasoner
from backend.agent.tools import LogStore
from backend.agent.workflow import AgentDeps, build_graph
from backend.api.streaming import Emit, TokenEmitter
from backend.core import metrics
from backend.core.services import Services
from backend.core.telemetry import current_trace_id, extract_context, get_tracer

log = logging.getLogger(__name__)


class AgentRunner:
    def __init__(
        self,
        services: Services,
        reasoner: Reasoner,
        checkpointer: BaseCheckpointSaver,
        chat_model: ChatModel | None = None,
    ):
        s = services.settings
        self.services = services
        self.reasoner = reasoner
        self.checkpointer = checkpointer
        self.chat_model = chat_model
        fallback = reasoner if isinstance(reasoner, HeuristicReasoner) else _heuristic(services)
        self.deps = AgentDeps(
            services=services,
            reasoner=reasoner,
            fallback=fallback,
            logs=LogStore(s.agent_logs_path),
            max_retries=s.agent_max_retries,
            top_k=s.agent_top_k,
        )
        self.graph = build_graph(self.deps).compile(checkpointer=checkpointer)

    @classmethod
    def from_services(cls, services: Services) -> AgentRunner:
        s = services.settings
        model = chat_model_from_settings(s)
        reasoner: Reasoner = (
            LLMReasoner(model, s.agent_history_messages) if model else _heuristic(services)
        )
        return cls(services, reasoner, checkpointer_from_settings(s), model)

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    async def history(self, thread_id: str) -> list[dict]:
        snapshot = await self.graph.aget_state(self._config(thread_id))
        return list(snapshot.values.get("messages", [])) if snapshot else []

    async def run_turn(
        self,
        thread_id: str,
        question: str,
        emit: Emit,
        request_id: str | None = None,
        traceparent: str | None = None,
        session_span=None,
    ) -> dict:
        """Run one turn, streaming events through `emit`. Returns the final state."""
        rid = request_id or uuid.uuid4().hex
        if traceparent:
            parent = extract_context({"traceparent": traceparent})
        elif session_span is not None:
            parent = otel_context.Context()  # new trace per turn, linked to the session
        else:
            parent = None
        links = [Link(session_span.get_span_context())] if session_span is not None else []
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "agent.turn",
            context=parent,
            kind=SpanKind.SERVER,
            links=links,
            attributes={
                "agent.thread_id": thread_id,
                "omniagent.request_id": rid,
                "agent.reasoner": self.reasoner.name,
            },
        ) as span:
            t0 = time.perf_counter()
            await emit(
                {
                    "type": "response.start",
                    "id": rid,
                    "thread_id": thread_id,
                    "trace_id": current_trace_id(),
                }
            )
            tokens: TokenEmitter | None = None
            first_token: float | None = None
            try:
                async for ev in self.graph.astream(
                    {"question": question}, self._config(thread_id), stream_mode="custom"
                ):
                    kind = ev.pop("kind")
                    if kind == "token":
                        first_token = first_token or time.perf_counter()
                        if tokens is None or tokens.extra["attempt"] != ev["attempt"]:
                            tokens = TokenEmitter(emit, rid, attempt=ev["attempt"])
                        await tokens(ev["text"])
                    elif kind == "generation.start":
                        tokens = TokenEmitter(emit, rid, attempt=ev["attempt"])
                        if ev.get("sources"):
                            await emit(
                                {
                                    "type": "response.sources",
                                    "id": rid,
                                    "attempt": ev["attempt"],
                                    "sources": ev["sources"],
                                }
                            )
                    elif kind == "generation.discarded":
                        metrics.AGENT_DISCARDED.inc()
                        await emit({"type": "agent.generation.discarded", "id": rid, **ev})
                    elif kind == "step":
                        if ev.get("fallback"):
                            metrics.AGENT_FALLBACKS.labels(ev["node"]).inc()
                        await emit({"type": "agent.step", "id": rid, **ev})

                state = (await self.graph.aget_state(self._config(thread_id))).values
                outcome = state.get("outcome", "")
                elapsed = time.perf_counter() - t0
                metrics.AGENT_TURNS.labels(state.get("tool") or "none", outcome or "none").inc()
                metrics.AGENT_RETRIES.observe(state.get("retries", 0))
                metrics.TURN_SECONDS.labels("agent").observe(elapsed)
                if first_token is not None:
                    metrics.TTFT_SECONDS.labels("agent").observe(first_token - t0)
                span.set_attributes(
                    {
                        "agent.outcome": outcome,
                        "agent.retries": state.get("retries", 0),
                        "agent.tool": state.get("tool", ""),
                    }
                )
                if outcome == "ungrounded":
                    await emit(
                        {
                            "type": "agent.warning",
                            "id": rid,
                            "message": "The answer could not be fully verified against "
                            "the sources. Treat it with caution.",
                        }
                    )
                await emit(
                    {
                        "type": "response.end",
                        "id": rid,
                        "attempt": state.get("attempts", 0),
                        "tokens": tokens.count if tokens else 0,
                        "sha256": tokens.sha256 if tokens else TokenEmitter(emit, rid).sha256,
                        "outcome": outcome,
                        "tool": state.get("tool"),
                        "retries": state.get("retries", 0),
                        "rewrites": state.get("rewrites", []),
                        "relevance": state.get("relevance", {}),
                        "grounding": state.get("grounding", {}),
                        "latency_ms": round(elapsed * 1000, 2),
                        "first_token_ms": round((first_token - t0) * 1000, 2)
                        if first_token
                        else None,
                    }
                )
                return state
            except Exception as exc:
                metrics.AGENT_TURNS.labels("none", "error").inc()
                log.exception("agent turn failed", extra={"thread_id": thread_id})
                get_current_span().record_exception(exc)
                await emit({"type": "response.error", "id": rid, "error": str(exc)})
                return {}

    async def close(self) -> None:
        if self.chat_model is not None:
            await self.chat_model.close()
        if hasattr(self.checkpointer, "aclose"):
            await self.checkpointer.aclose()


def _heuristic(services: Services) -> HeuristicReasoner:
    s = services.settings
    return HeuristicReasoner(
        relevance_threshold=s.agent_relevance_threshold,
        grounding_threshold=s.agent_grounding_threshold,
        # Real cross-encoder logits (Triton) are calibrated; mock overlap scores are not.
        rerank_calibrated=s.rerank_enabled and s.inference_backend == "triton",
    )
