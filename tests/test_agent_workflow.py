"""LangGraph workflow: self-correction loops, fallbacks, routing and memory.

`ScriptedReasoner` forces specific grades so each loop is exercised
deterministically. Everything else (graph, retrieval, rerank, tools, streaming
events, checkpointer) is the real implementation.
"""

import asyncio
import hashlib

import fakeredis
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from backend.agent.llm import LLMError
from backend.agent.memory import RedisCheckpointSaver
from backend.agent.reasoning import (
    Generation,
    GroundingGrade,
    HeuristicReasoner,
    LLMReasoner,
    RelevanceGrade,
    stream_words,
)
from backend.agent.runner import AgentRunner
from backend.core.models import Document
from backend.core.services import Services
from tests.conftest import load_samples, make_settings

HALLUCINATION = "Upgrade the controller to firmware 9.9.9 from the vendor portal."


class ScriptedReasoner(HeuristicReasoner):
    """Heuristic reasoner whose grades / answers can be overridden per call."""

    name = "scripted"

    def __init__(self, relevance=(), grounding=(), answers=()):
        super().__init__()
        self.relevance = list(relevance)  # bools, consumed per grade_relevance call
        self.grounding = list(grounding)  # bools, consumed per grade_grounding call
        self.answers = list(answers)  # str | None (None = heuristic answer)
        self.calls: list[tuple] = []

    async def contextualize(self, question, history):
        self.calls.append(("contextualize", question, len(history)))
        return await super().contextualize(question, history)

    async def grade_relevance(self, question, passages):
        if not self.relevance:
            return await super().grade_relevance(question, passages)
        ok = self.relevance.pop(0)
        return RelevanceGrade(
            score=0.9 if ok else 0.1,
            relevant=ok,
            reason="scripted: " + ("covers" if ok else "off-topic"),
        )

    async def rewrite(self, question, current_query, attempt, feedback, passages):
        self.calls.append(("rewrite", current_query, attempt, feedback))
        return await super().rewrite(question, current_query, attempt, feedback, passages)

    async def generate(self, question, ctx, history, on_token):
        scripted = self.answers.pop(0) if self.answers else None
        if scripted is None:
            return await super().generate(question, ctx, history, on_token)
        await stream_words(scripted, on_token)
        return Generation(scripted, abstained=False)

    async def grade_grounding(self, question, answer, ctx):
        if not self.grounding:
            return await super().grade_grounding(question, answer, ctx)
        ok = self.grounding.pop(0)
        return GroundingGrade(
            score=1.0 if ok else 0.0,
            grounded=ok,
            unsupported_claims=[] if ok else [answer],
            reason="scripted",
        )


@pytest.fixture
async def services():
    svc = await Services.build(make_settings())
    await svc.processor.ingest(load_samples())
    yield svc
    await svc.close()


def make_agent(services, reasoner, checkpointer=None, **overrides):
    if overrides:
        services.settings = services.settings.model_copy(update=overrides)
    return AgentRunner(services, reasoner, checkpointer or InMemorySaver())


async def run(agent, question, thread="t-test-0001"):
    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    state = await agent.run_turn(thread, question, emit, request_id="r1")
    return state, events


def steps(events, node=None):
    return [e for e in events if e["type"] == "agent.step" and (node is None or e["node"] == node)]


def assert_final_tokens_match_end(events):
    end = events[-1]
    assert end["type"] == "response.end"
    final = [e for e in events if e["type"] == "response.token" and e["attempt"] == end["attempt"]]
    assert [t["index"] for t in final] == list(range(len(final)))
    text = "".join(t["text"] for t in final)
    assert end["tokens"] == len(final)
    assert end["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    return text


# --- retrieval self-correction ---------------------------------------------------


async def test_low_relevance_triggers_rewrite_and_reretrieval(services):
    reasoner = ScriptedReasoner(relevance=[False, True])
    q = "hey so my storage box keeps spitting out EIO5, any idea whats going on??"
    state, events = await run(make_agent(services, reasoner), q)

    retrieves = steps(events, "retrieve")
    rewrites = steps(events, "rewrite_query")
    grades = steps(events, "grade_documents")
    assert len(retrieves) == 2 and len(rewrites) == 1
    assert [g["relevant"] for g in grades] == [False, True]
    assert rewrites[0]["strategy"] == "keyword_focus"
    assert rewrites[0]["because"].startswith("low relevance")
    assert retrieves[1]["query"] == rewrites[0]["query"] != retrieves[0]["query"]
    assert "eio 5" in retrieves[1]["query"]
    assert state["outcome"] == "answered" and state["retries"] == 1
    assert events[-1]["rewrites"] == [rewrites[0]["query"]]
    assert "EIO-5" in assert_final_tokens_match_end(events)


async def test_persistently_low_relevance_stops_after_max_retries_and_abstains(services):
    reasoner = ScriptedReasoner(relevance=[False, False, False, False], answers=[HALLUCINATION])
    state, events = await run(make_agent(services, reasoner), "rotate the kafka broker keystore")

    assert len(steps(events, "retrieve")) == 3  # initial + 2 retries
    assert len(steps(events, "rewrite_query")) == 2
    assert [r["attempt"] for r in steps(events, "retrieve")] == [1, 2, 3]
    gen = steps(events, "generate")[0]
    assert gen["insufficient_context"] and gen["abstained"]
    assert state["outcome"] == "insufficient_context" and state["retries"] == 2
    assert assert_final_tokens_match_end(events).startswith("I couldn't find")
    assert reasoner.relevance == [False]  # never graded a 4th time
    assert reasoner.answers == [HALLUCINATION]  # generation was never called
    assert not any(event["type"] == "response.sources" for event in events)


async def test_shared_redis_thread_rejects_overlap_across_runners(services):
    class PausedReasoner(HeuristicReasoner):
        async def contextualize(self, question, history):
            started.set()
            await resume.wait()
            return await super().contextualize(question, history)

    started = asyncio.Event()
    resume = asyncio.Event()
    server = fakeredis.FakeServer()
    first = make_agent(
        services, PausedReasoner(), RedisCheckpointSaver(fakeredis.FakeAsyncRedis(server=server))
    )
    second = make_agent(
        services, HeuristicReasoner(), RedisCheckpointSaver(fakeredis.FakeAsyncRedis(server=server))
    )
    task = asyncio.create_task(run(first, "64/2"))
    try:
        await asyncio.wait_for(started.wait(), 3)
        state, events = await run(second, "42/2")
        assert state == {}
        assert events[-1]["code"] == "thread_busy" and events[-1]["retryable"] is True
        assert not any(event["type"] == "response.token" for event in events)
    finally:
        resume.set()
        await task
        await first.close()
    try:
        assert assert_final_tokens_match_end((await task)[1]) == "64/2 = 32"
        _, events = await run(second, "42/2")
        assert assert_final_tokens_match_end(events) == "42/2 = 21"
        history = await second.history("t-test-0001")
        assert [m["content"] for m in history if m["role"] == "user"] == ["64/2", "42/2"]
    finally:
        await second.close()


async def test_max_retries_is_configurable(services):
    reasoner = ScriptedReasoner(relevance=[False, False, False])
    state, events = await run(make_agent(services, reasoner, agent_max_retries=1), "kafka")
    assert len(steps(events, "rewrite_query")) == 1 and state["retries"] == 1


async def test_real_heuristic_fallback_loop_on_out_of_scope_question(services):
    """No scripting: the heuristic grader finds the KB can't answer, loops, then abstains."""
    state, events = await run(
        make_agent(services, HeuristicReasoner()), "how do I rotate the kafka broker TLS keystore?"
    )
    rewrites = steps(events, "rewrite_query")
    assert [r["strategy"] for r in rewrites] == ["keyword_focus", "pseudo_relevance_feedback"]
    assert all(not g["relevant"] for g in steps(events, "grade_documents"))
    assert state["outcome"] == "insufficient_context"


async def test_short_calculations_use_only_the_current_operands(services):
    agent = make_agent(services, HeuristicReasoner())
    for question, expected in [("4000/2", 2000), ("4002/2", 2001), ("4004/2", 2002)]:
        state, events = await run(agent, question)
        assert state["answer"] == f"{question} = {expected}"
        assert assert_final_tokens_match_end(events) == state["answer"]


async def test_agent_limits_concurrent_turns_on_different_threads(services):
    entered = []
    started = asyncio.Event()
    resume = asyncio.Event()

    class PausedReasoner(HeuristicReasoner):
        async def contextualize(self, question, history):
            entered.append(question)
            started.set()
            await resume.wait()
            return await super().contextualize(question, history)

    agent = make_agent(services, PausedReasoner(), agent_max_in_flight=1)
    first = asyncio.create_task(run(agent, "64/2", "test-limit-1"))
    second = asyncio.create_task(run(agent, "42/2", "test-limit-2"))
    try:
        await asyncio.wait_for(started.wait(), 3)
        assert entered == ["64/2"]
        assert not second.done()
    finally:
        resume.set()
        first_result, second_result = await asyncio.gather(first, second)
    assert assert_final_tokens_match_end(first_result[1]) == "64/2 = 32"
    assert assert_final_tokens_match_end(second_result[1]) == "42/2 = 21"


# --- generation self-correction --------------------------------------------------


async def test_ungrounded_answer_is_discarded_and_regenerated(services):
    reasoner = ScriptedReasoner(answers=[HALLUCINATION, None], grounding=[False, True])
    state, events = await run(make_agent(services, reasoner), "how do I fix EIO-5 timeouts?")

    discarded = [e for e in events if e["type"] == "agent.generation.discarded"]
    assert len(discarded) == 1 and discarded[0]["attempt"] == 1
    grades = steps(events, "grade_generation")
    assert [g["grounded"] for g in grades] == [False, True]
    assert grades[0]["will_retry"] is True
    rewrite = steps(events, "rewrite_query")[0]
    assert rewrite["because"].startswith("previous answer was not supported")
    assert len(steps(events, "retrieve")) == 2

    assert HALLUCINATION not in str(events)
    assert all(e["attempt"] == 2 for e in events if e["type"] == "response.token")
    final = assert_final_tokens_match_end(events)
    assert "9.9.9" not in final and events[-1]["attempt"] == 2
    assert state["outcome"] == "answered" and state["answer"] == final
    sources = [e for e in events if e["type"] == "response.sources"]
    assert [s["attempt"] for s in sources] == [2]


async def test_answer_still_ungrounded_after_retries_ends_with_warning(services):
    reasoner = ScriptedReasoner(answers=[HALLUCINATION] * 3, grounding=[False] * 3)
    agent = make_agent(services, reasoner)
    state, events = await run(agent, "how do I fix EIO-5 timeouts?")
    assert len([e for e in events if e["type"] == "agent.generation.discarded"]) == 2
    assert steps(events, "grade_generation")[-1]["will_retry"] is False
    assert state["outcome"] == "ungrounded" and state["attempts"] == 3
    assert any(e["type"] == "agent.warning" for e in events)
    assert events[-1]["outcome"] == "ungrounded"
    assert HALLUCINATION not in str(events)
    assert assert_final_tokens_match_end(events).startswith("I couldn't verify")
    assert HALLUCINATION not in str(await agent.history("t-test-0001"))


async def test_conflicting_runbooks_are_cited_without_model_recommendation(services):
    await services.processor.ingest(
        [
            Document(
                doc_id=f"policy-{value}",
                title=f"OR-71 Policy {value}",
                text=(
                    f"The OR-71 policy sets max_connections = {value} for storage-07. "
                    "This is the production hard limit."
                ),
            )
            for value in (256, 100)
        ]
    )
    reasoner = ScriptedReasoner(relevance=[True], grounding=[True], answers=[HALLUCINATION])
    agent = make_agent(services, reasoner, agent_top_k=1, rerank_enabled=True)
    _, events = await run(agent, "What is max_connections for storage-07 during OR-71?")
    assert events[-1]["outcome"] == "abstained"
    sources = next(e["sources"] for e in events if e["type"] == "response.sources")
    text = assert_final_tokens_match_end(events)
    assert "sources disagree" in text and "Confirm the authoritative policy" in text
    assert len(sources) == 2
    for i, source in enumerate(sources, 1):
        value = source["document_title"].split()[-1]
        assert f"`{value}` [{i}]" in text
    assert reasoner.answers == [HALLUCINATION]

    # The host and setting remain intact across a short follow-up on the same thread.
    state, events = await run(agent, "and which one should I use?")
    assert "storage-07" in state["standalone"] and "max_connections" in state["standalone"]
    assert events[-1]["outcome"] == "abstained"
    assert "sources disagree" in assert_final_tokens_match_end(events)


async def test_unknown_log_host_abstains_without_borrowing_other_hosts(services):
    reasoner = ScriptedReasoner(answers=[HALLUCINATION])
    _, events = await run(make_agent(services, reasoner), "show recent errors on storage-99")
    assert events[-1]["outcome"] == "insufficient_context"
    assert (
        assert_final_tokens_match_end(events)
        == "I couldn't find matching log lines for that request."
    )
    assert reasoner.answers == [HALLUCINATION]


async def test_out_of_scope_topic_on_known_host_is_not_answered(services):
    await services.processor.ingest(
        [
            Document(
                doc_id="storage-policy",
                title="Storage policy",
                text="The production hard limit is max_connections = 256 for storage-07.",
            )
        ]
    )
    state, events = await run(
        make_agent(services, HeuristicReasoner()),
        "How do I configure the flux capacitor on storage-07?",
    )
    assert state["outcome"] == "insufficient_context" and state["retries"] == 2
    assert assert_final_tokens_match_end(events).startswith("I couldn't find enough information")
    assert not any(e["type"] == "response.sources" for e in events)


async def test_contradictory_grounding_verdict_cannot_release_draft(services):
    class InconsistentGrader(ScriptedReasoner):
        async def grade_grounding(self, question, answer, ctx):
            return GroundingGrade(
                score=1, grounded=True, unsupported_claims=[answer], reason=answer
            )

    agent = make_agent(services, InconsistentGrader(answers=[HALLUCINATION]), agent_max_retries=0)
    _, events = await run(agent, "how do I fix EIO-5 timeouts?")
    assert HALLUCINATION not in str(events)
    assert events[-1]["outcome"] == "ungrounded"
    assert events[-1]["grounding"]["score"] == 0
    assert assert_final_tokens_match_end(events).startswith("I couldn't verify")


# --- LLM failure fallbacks ----------------------------------------------------------


class FlakyModel:
    """Fails structured calls; streams a reply or fails mid-stream."""

    name = "flaky"

    def __init__(self, stream_fail_after: int | None = None):
        self.stream_fail_after = stream_fail_after

    async def complete_json(self, system, messages, schema, effort="low"):
        raise LLMError("upstream 529 overloaded")

    async def stream_text(self, system, messages, effort="medium"):
        for i, word in enumerate("EIO-5 means a controller write timeout [1].".split()):
            if self.stream_fail_after is not None and i == self.stream_fail_after:
                raise LLMError("stream reset")
            yield word + " "

    async def close(self):
        pass


async def test_llm_step_failures_fall_back_to_heuristics(services):
    agent = make_agent(services, LLMReasoner(FlakyModel()))
    state, events = await run(agent, "show me recent errors on storage-07")
    route = steps(events, "route")[0]
    assert route["fallback"] is True and route["tool"] == "live_logs"
    assert state["outcome"] == "answered"
    assert "EIO-5" in state["answer"]


async def test_generation_failure_mid_stream_is_discarded_and_regenerated(services):
    agent = make_agent(services, LLMReasoner(FlakyModel(stream_fail_after=3)))
    state, events = await run(agent, "what does EIO-5 mean?")
    discarded = [e for e in events if e["type"] == "agent.generation.discarded"]
    assert len(discarded) == 1 and discarded[0]["reason"].startswith("generation error")
    gen = steps(events, "generate")[0]
    assert gen["fallback"] is True and gen["attempt"] == 2
    assert all(e["attempt"] == 2 for e in events if e["type"] == "response.token")
    assert assert_final_tokens_match_end(events) == state["answer"]


# --- routing to tools ---------------------------------------------------------------


@pytest.mark.parametrize(
    "question,tool,expect",
    [
        ("show me recent errors on storage-07", "live_logs", "EIO-5 timeout"),
        ("if queue depth is 64 what's 64/2?", "calculator", "64/2 = 32"),
        ("hello", "direct", "OmniAgent"),
    ],
)
async def test_tool_routing(services, question, tool, expect):
    state, events = await run(make_agent(services, HeuristicReasoner()), question)
    assert steps(events, "route")[0]["tool"] == tool
    assert expect in state["answer"]
    assert not steps(events, "retrieve")  # no KB retrieval for non-KB tools
    assert_final_tokens_match_end(events)


# --- memory -------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_multi_turn_memory_and_follow_up_contextualization(services, backend):
    if backend == "redis":
        saver = RedisCheckpointSaver(fakeredis.FakeAsyncRedis(decode_responses=False))
    else:
        saver = InMemorySaver()
    reasoner = ScriptedReasoner()
    agent = make_agent(services, reasoner, saver)
    await run(agent, "my storage node logs EIO5 errors, what is it?", "t-memory-001")
    state, events = await run(agent, "and which firmware version fixes it?", "t-memory-001")

    ctx = steps(events, "contextualize")[0]
    assert ctx["changed"] and "eio" in ctx["standalone"]
    assert [c for c in reasoner.calls if c[0] == "contextualize"][1][2] == 2  # saw turn 1
    assert "4.2.1" in state["answer"]
    history = await agent.history("t-memory-001")
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[2]["standalone"] == ctx["standalone"]
    assert await agent.history("t-other-0001") == []


async def test_memory_survives_a_new_runner_on_the_same_redis(services):
    server = fakeredis.FakeServer()

    def saver():
        return RedisCheckpointSaver(fakeredis.FakeAsyncRedis(server=server, decode_responses=False))

    first = make_agent(services, HeuristicReasoner(), saver())
    await run(first, "what does EIO-5 mean?", "t-restart-01")
    second = make_agent(services, HeuristicReasoner(), saver())  # "process restart"
    history = await second.history("t-restart-01")
    assert len(history) == 2 and history[0]["content"] == "what does EIO-5 mean?"
    _, events = await run(second, "which firmware fixes it?", "t-restart-01")
    assert steps(events, "contextualize")[0]["changed"]
