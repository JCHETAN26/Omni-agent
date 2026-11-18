"""LangGraph agentic RAG workflow with self-correction.

    START -> contextualize -> route --knowledge_base--> retrieve -> grade_documents
                                |                          ^              |
                                |                          |   relevant --+--> generate
                                |                          |              |
                                |                    rewrite_query <------+ not relevant
                                |                          ^                 (retries left)
                                |-- live_logs --> inspect_logs --> generate
                                |-- calculator --> calculate ----> generate
                                '-- direct ----------------------> generate
    generate -> grade_generation --grounded--> finalize -> END
                       '--not grounded (knowledge_base, retries left)--> rewrite_query

* At most `max_retries` rewrites per turn, shared by both loops. When retries run
  out: on low relevance the answer is generated with an "insufficient context"
  flag (it abstains), and on failed grounding the turn ends `ungrounded` with a
  warning to the client.
* Every decision goes through the configured `Reasoner`. If an LLM call fails,
  that single step falls back to the `HeuristicReasoner` and the step event
  records `fallback: true`.
* Progress, sources and answer tokens are pushed through LangGraph's custom
  stream (`get_stream_writer`). `runner.py` turns them into WebSocket/SSE events.
* `messages` is the only field that accumulates across turns. It is persisted
  by the checkpointer and keyed by `thread_id`.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from backend.agent.llm import LLMError
from backend.agent.reasoning import Generation, HeuristicReasoner, Reasoner, ToolContext
from backend.agent.tools import CalculatorError, LogStore, calculate, extract_expression
from backend.core.services import Services
from backend.core.telemetry import get_tracer
from search.pipeline import retrieve_with_rerank

log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # conversation history (persisted)
    question: str  # the user's message this turn
    standalone: str  # question rewritten to stand alone
    query: str  # current retrieval query
    tool: str
    tool_argument: str
    passages: list[dict]
    sources: list[dict]
    tool_output: str
    answer: str
    abstained: bool
    relevance: dict
    grounding: dict
    retries: int  # rewrites used this turn
    attempts: int  # answers generated this turn
    rewrites: list[str]
    feedback: str  # why the last attempt failed (input to the rewriter)
    outcome: str  # answered | abstained | insufficient_context | ungrounded


@dataclass
class AgentDeps:
    services: Services
    reasoner: Reasoner
    fallback: HeuristicReasoner
    logs: LogStore
    max_retries: int
    top_k: int


def _emit(kind: str, **data: Any) -> None:
    get_stream_writer()({"kind": kind, **data})


def _step(node: str, **data: Any) -> None:
    _emit("step", node=node, **data)


async def _decide(deps: AgentDeps, op: str, *args) -> tuple[Any, bool]:
    """Run one reasoning step; on LLM failure, redo it with the heuristic fallback."""
    try:
        return await getattr(deps.reasoner, op)(*args), False
    except LLMError as exc:
        if deps.reasoner is deps.fallback:
            raise
        log.warning("reasoner step %s failed (%s); using heuristic fallback", op, exc)
        return await getattr(deps.fallback, op)(*args), True


def _passage(hit, rerank_score) -> dict:
    c = hit.chunk
    return {
        "chunk_id": c.chunk_id,
        "document_title": c.document_title,
        "section_heading": c.section_heading,
        "text": c.text,
        "rrf_score": hit.score,
        "rerank_score": rerank_score,
    }


def _context(state: AgentState) -> ToolContext:
    tool = state.get("tool", "knowledge_base")
    if tool == "knowledge_base":
        return ToolContext(
            "knowledge_base",
            passages=state.get("passages", []),
            insufficient=not state.get("relevance", {}).get("relevant", False),
        )
    return ToolContext(tool, text=state.get("tool_output", ""))  # type: ignore[arg-type]


def build_graph(deps: AgentDeps) -> StateGraph:
    tracer = get_tracer()

    async def contextualize(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.contextualize"):
            question = state["question"]
            standalone, fb = await _decide(
                deps, "contextualize", question, state.get("messages", [])
            )
            _step(
                "contextualize", standalone=standalone, changed=standalone != question, fallback=fb
            )
            # Reset per-turn fields (everything except the accumulated history).
            return {
                "standalone": standalone,
                "query": standalone,
                "retries": 0,
                "attempts": 0,
                "rewrites": [],
                "passages": [],
                "sources": [],
                "tool_output": "",
                "answer": "",
                "abstained": False,
                "relevance": {},
                "grounding": {},
                "feedback": "",
                "outcome": "",
            }

    async def route(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.route") as span:
            decision, fb = await _decide(
                deps, "route", state["standalone"], state.get("messages", [])
            )
            span.set_attribute("agent.tool", decision.tool)
            _step(
                "route",
                tool=decision.tool,
                argument=decision.argument,
                reason=decision.reason,
                fallback=fb,
            )
            update = {"tool": decision.tool, "tool_argument": decision.argument}
            if decision.tool == "knowledge_base" and decision.argument.strip():
                update["query"] = decision.argument.strip()
            return update

    async def retrieve(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.retrieve") as span:
            s = deps.services
            kb = await retrieve_with_rerank(
                s.retriever, s.inference, s.settings, state["query"], deps.top_k
            )
            passages = [_passage(h, r) for h, r in zip(kb.hits, kb.rerank_scores, strict=True)]
            span.set_attribute("agent.hits", len(passages))
            _step(
                "retrieve",
                query=state["query"],
                attempt=state.get("retries", 0) + 1,
                hits=[p["section_heading"] for p in passages],
                reranked=kb.reranked,
            )
            return {"passages": passages, "sources": kb.sources()}

    async def grade_documents(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.grade_documents") as span:
            grade, fb = await _decide(
                deps, "grade_relevance", state["standalone"], state.get("passages", [])
            )
            span.set_attributes({"agent.relevance": grade.score, "agent.relevant": grade.relevant})
            _step(
                "grade_documents",
                score=grade.score,
                relevant=grade.relevant,
                reason=grade.reason,
                fallback=fb,
            )
            return {
                "relevance": grade.model_dump(),
                "feedback": "" if grade.relevant else f"low relevance: {grade.reason}",
            }

    async def rewrite_query(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.rewrite_query"):
            retries = state.get("retries", 0) + 1
            rw, fb = await _decide(
                deps,
                "rewrite",
                state["standalone"],
                state["query"],
                retries,
                state.get("feedback", ""),
                state.get("passages", []),
            )
            new_query = rw.query.strip() or state["query"]
            _step(
                "rewrite_query",
                retry=retries,
                previous=state["query"],
                query=new_query,
                strategy=rw.strategy,
                because=state.get("feedback", ""),
                fallback=fb,
            )
            return {
                "query": new_query,
                "retries": retries,
                "rewrites": [*state.get("rewrites", []), new_query],
            }

    async def inspect_logs(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.inspect_logs"):
            host = state.get("tool_argument", "").strip() or None
            if host and host not in deps.logs.hosts():
                host_note = f"No log source named {host}. "
                host = None
            else:
                host_note = ""
            wants_errors = any(w in state["standalone"].lower() for w in ("error", "fail", "warn"))
            lines = deps.logs.search(host=host, min_level="WARN" if wants_errors else None)
            if lines:
                where = f" on {host}" if host else ""
                output = (
                    host_note
                    + f"Found {len(lines)} recent log lines{where}:\n"
                    + "\n".join(ln.render() for ln in lines)
                )
            else:
                output = host_note + "I couldn't find matching log lines."
            _step("inspect_logs", host=host, lines=len(lines))
            return {"tool_output": output}

    async def run_calculator(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.calculate"):
            expr = state.get("tool_argument", "").strip() or extract_expression(state["standalone"])
            try:
                if not expr:
                    raise CalculatorError("no arithmetic expression found")
                output = f"{expr} = {calculate(expr)}"
                ok = True
            except CalculatorError as exc:
                output, ok = f"I couldn't compute that: {exc}.", False
            _step("calculate", expression=expr, ok=ok, result=output)
            return {"tool_output": output}

    async def generate(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.generate") as span:
            attempt = state.get("attempts", 0) + 1
            ctx = _context(state)
            writer = get_stream_writer()

            async def on_token(text: str) -> None:
                writer({"kind": "token", "attempt": attempt, "text": text})

            writer(
                {
                    "kind": "generation.start",
                    "attempt": attempt,
                    "sources": state.get("sources", []) if ctx.kind == "knowledge_base" else [],
                }
            )
            history = state.get("messages", [])
            fb = False
            try:
                gen: Generation = await deps.reasoner.generate(
                    state["standalone"], ctx, history, on_token
                )
            except LLMError as exc:
                if deps.reasoner is deps.fallback:
                    raise
                log.warning("generation failed (%s); regenerating with heuristic fallback", exc)
                writer(
                    {
                        "kind": "generation.discarded",
                        "attempt": attempt,
                        "reason": f"generation error: {exc}",
                    }
                )
                attempt += 1
                writer({"kind": "generation.start", "attempt": attempt, "sources": []})
                gen = await deps.fallback.generate(state["standalone"], ctx, history, on_token)
                fb = True
            span.set_attribute("agent.attempt", attempt)
            _step(
                "generate",
                attempt=attempt,
                chars=len(gen.text),
                abstained=gen.abstained,
                insufficient_context=ctx.insufficient,
                fallback=fb,
            )
            return {"answer": gen.text, "abstained": gen.abstained, "attempts": attempt}

    async def grade_generation(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.grade_generation") as span:
            ctx = _context(state)
            if state.get("abstained"):
                grade = {
                    "score": 1.0,
                    "grounded": True,
                    "unsupported_claims": [],
                    "reason": "answer abstains; nothing to verify",
                }
                fb = False
            else:
                g, fb = await _decide(
                    deps, "grade_grounding", state["standalone"], state["answer"], ctx
                )
                grade = g.model_dump()
            span.set_attributes(
                {"agent.grounded": grade["grounded"], "agent.grounding": grade["score"]}
            )
            retry = (
                not grade["grounded"]
                and state.get("tool") == "knowledge_base"
                and state.get("retries", 0) < deps.max_retries
            )
            _step(
                "grade_generation",
                score=grade["score"],
                grounded=grade["grounded"],
                unsupported=grade["unsupported_claims"],
                reason=grade["reason"],
                will_retry=retry,
                fallback=fb,
            )
            if retry:
                _emit(
                    "generation.discarded",
                    attempt=state.get("attempts", 1),
                    reason="answer not grounded in sources",
                )
            feedback = (
                ""
                if grade["grounded"]
                else (
                    "previous answer was not supported by the retrieved passages; unsupported: "
                    + "; ".join(grade["unsupported_claims"][:3])
                )
            )
            return {"grounding": grade, "feedback": feedback}

    async def finalize(state: AgentState) -> dict:
        relevant = state.get("relevance", {}).get("relevant", True)
        grounded = state.get("grounding", {}).get("grounded", True)
        if state.get("tool") == "knowledge_base" and not relevant:
            outcome = "insufficient_context"
        elif not grounded:
            outcome = "ungrounded"
        elif state.get("abstained"):
            outcome = "abstained"
        else:
            outcome = "answered"
        _step(
            "finalize",
            outcome=outcome,
            retries=state.get("retries", 0),
            attempts=state.get("attempts", 0),
        )
        return {
            "outcome": outcome,
            "messages": [
                {"role": "user", "content": state["question"], "standalone": state["standalone"]},
                {
                    "role": "assistant",
                    "content": state.get("answer", ""),
                    "tool": state.get("tool"),
                    "outcome": outcome,
                },
            ],
        }

    # -- edges ------------------------------------------------------------------

    def after_route(state: AgentState) -> str:
        return {
            "knowledge_base": "retrieve",
            "live_logs": "inspect_logs",
            "calculator": "calculate",
            "direct": "generate",
        }[state["tool"]]

    def after_grade_documents(state: AgentState) -> str:
        if state["relevance"]["relevant"]:
            return "generate"
        return "rewrite_query" if state.get("retries", 0) < deps.max_retries else "generate"

    def after_grade_generation(state: AgentState) -> str:
        if state["grounding"]["grounded"]:
            return "finalize"
        if state.get("tool") == "knowledge_base" and state.get("retries", 0) < deps.max_retries:
            return "rewrite_query"
        return "finalize"

    g = StateGraph(AgentState)
    g.add_node("contextualize", contextualize)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("inspect_logs", inspect_logs)
    g.add_node("calculate", run_calculator)
    g.add_node("generate", generate)
    g.add_node("grade_generation", grade_generation)
    g.add_node("finalize", finalize)

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "route")
    g.add_conditional_edges(
        "route", after_route, ["retrieve", "inspect_logs", "calculate", "generate"]
    )
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", after_grade_documents, ["generate", "rewrite_query"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("inspect_logs", "generate")
    g.add_edge("calculate", "generate")
    g.add_edge("generate", "grade_generation")
    g.add_conditional_edges(
        "grade_generation", after_grade_generation, ["finalize", "rewrite_query"]
    )
    g.add_edge("finalize", END)
    return g
