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

* Both loops share the rewrite budget. Exhausted relevance or grounding checks
  produce deterministic abstentions. Detected conflicts get a cited abstention.
* Every decision goes through the configured `Reasoner`. If an LLM call fails,
  that single step falls back to the `HeuristicReasoner` and the step event
  records `fallback: true`.
* LangGraph streams progress. The runner releases answer text after finalization.
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
from backend.agent.reasoning import (
    Generation,
    HeuristicReasoner,
    Reasoner,
    RelevanceGrade,
    ToolContext,
)
from backend.agent.tools import CalculatorError, LogStore, calculate, extract_expression
from backend.core.services import Services
from backend.core.telemetry import get_tracer
from search.pipeline import retrieve_with_rerank

log = logging.getLogger(__name__)

INSUFFICIENT_ANSWER = "I couldn't find enough information in the knowledge base to answer that."
UNVERIFIED_ANSWER = "I couldn't verify an answer against the available evidence."


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # conversation history (persisted)
    question: str  # the user's message this turn
    standalone: str  # question rewritten to stand alone
    query: str  # current retrieval query
    tool: str
    tool_argument: str
    passages: list[dict]
    sources: list[dict]
    conflict: str
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
    output = state.get("tool_output", "")
    return ToolContext(tool, text=output, insufficient=tool == "live_logs" and not output)


async def _discard_delta(_: str) -> None:
    """Provider deltas are drafts. The final answer is emitted by the runner."""


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
                "conflict": "",
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
                s.retriever,
                s.inference,
                s.settings,
                state["query"],
                deps.top_k,
                evidence_question=state["standalone"],
                ltr=s.ltr,
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
            return {"passages": passages, "sources": kb.sources(), "conflict": kb.conflict}

    async def grade_documents(state: AgentState) -> dict:
        with tracer.start_as_current_span("agent.grade_documents") as span:
            passages = state.get("passages", [])
            fb = False
            if state.get("conflict"):
                grade = RelevanceGrade(score=1, relevant=True, reason="matching sources disagree")
            elif not passages:
                grade = RelevanceGrade(score=0, relevant=False, reason="no passages retrieved")
            else:
                grade, fb = await _decide(deps, "grade_relevance", state["standalone"], passages)
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
            wants_errors = any(w in state["standalone"].lower() for w in ("error", "fail", "warn"))
            lines = deps.logs.search(host=host, min_level="WARN" if wants_errors else None)
            output = ""
            if lines:
                where = f" on {host}" if host else ""
                output = f"Found {len(lines)} recent log lines{where}:\n" + "\n".join(
                    ln.render() for ln in lines
                )
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
            history = state.get("messages", [])
            fb = False
            if conflict := state.get("conflict"):
                gen = Generation(conflict, abstained=True)
            elif ctx.insufficient:
                text = (
                    INSUFFICIENT_ANSWER
                    if ctx.kind == "knowledge_base"
                    else ("I couldn't find matching log lines for that request.")
                )
                gen = Generation(text, abstained=True)
            elif ctx.kind == "direct":
                # Open-ended generation has no evidence or grounding step on this route.
                gen = await deps.fallback.generate(
                    state["standalone"], ctx, history, _discard_delta
                )
            else:
                try:
                    gen: Generation = await deps.reasoner.generate(
                        state["standalone"], ctx, history, _discard_delta
                    )
                except LLMError as exc:
                    if deps.reasoner is deps.fallback:
                        raise
                    log.warning("generation failed (%s); regenerating with heuristic fallback", exc)
                    _emit("generation.discarded", attempt=attempt, reason="generation error")
                    attempt += 1
                    gen = await deps.fallback.generate(
                        state["standalone"], ctx, history, _discard_delta
                    )
                    fb = True
                if gen.abstained:
                    gen = Generation(UNVERIFIED_ANSWER, abstained=True)
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
                    "reason": "cited sources disagree"
                    if state.get("conflict")
                    else "answer abstains",
                }
                fb = False
            else:
                g, fb = await _decide(
                    deps, "grade_grounding", state["standalone"], state["answer"], ctx
                )
                grade = g.model_dump()
                if (
                    grade["unsupported_claims"]
                    or not grade["grounded"]
                    or grade["score"] < deps.services.settings.agent_grounding_threshold
                    or not state["answer"].strip()
                ):
                    # Grader reasons and claim lists can quote rejected drafts.
                    grade.update(
                        score=0.0 if grade["grounded"] else grade["score"],
                        grounded=False,
                        unsupported_claims=[],
                        reason="the answer could not be verified against the evidence",
                    )
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
                else "previous answer was not supported by the retrieved evidence"
            )
            return {"grounding": grade, "feedback": feedback}

    async def finalize(state: AgentState) -> dict:
        grounded = state.get("grounding", {}).get("grounded", True)
        if _context(state).insufficient:
            outcome = "insufficient_context"
        elif not grounded:
            outcome = "ungrounded"
        elif state.get("abstained"):
            outcome = "abstained"
        else:
            outcome = "answered"
        answer = state.get("answer", "") if grounded else UNVERIFIED_ANSWER
        _step(
            "finalize",
            outcome=outcome,
            retries=state.get("retries", 0),
            attempts=state.get("attempts", 0),
        )
        return {
            "outcome": outcome,
            "answer": answer,
            "abstained": state.get("abstained", False) or not grounded,
            "messages": [
                {"role": "user", "content": state["question"], "standalone": state["standalone"]},
                {
                    "role": "assistant",
                    "content": answer,
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
