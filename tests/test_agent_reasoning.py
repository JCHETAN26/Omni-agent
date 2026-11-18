"""Heuristic reasoner behaviour and the LLM reasoner's prompt/parse plumbing."""

import pytest

from backend.agent.llm import LLMError, strict_json_schema
from backend.agent.reasoning import (
    GroundingGrade,
    HeuristicReasoner,
    LLMReasoner,
    RelevanceGrade,
    Rewrite,
    RouteDecision,
    StandaloneQuestion,
    ToolContext,
)

EIO = {
    "document_title": "Storage Node Runbook",
    "section_heading": "Storage Node Runbook > Disk Errors > I/O Timeouts (EIO-5)",
    "text": "Error code EIO-5 means the controller timed out waiting for a block write. "
    "Firmware below 4.2.1 has a known bug where queue depth above 64 triggers "
    "spurious timeouts; set queue_depth=32 as a workaround.",
    "rerank_score": None,
}
PSU = {
    "document_title": "Power and Cooling",
    "section_heading": "Power and Cooling > PSU Redundancy Alerts",
    "text": "A PSU redundancy lost alert means one of the two power supplies has failed.",
    "rerank_score": None,
}


async def collect(gen_fn, *args):
    parts = []

    async def on_token(t):
        parts.append(t)

    gen = await gen_fn(*args, on_token)
    assert "".join(parts) == gen.text  # streamed text == returned text
    return gen


# --- heuristic -----------------------------------------------------------------


async def test_heuristic_contextualizes_follow_ups_only():
    h = HeuristicReasoner()
    history = [
        {
            "role": "user",
            "content": "my node logs EIO5 errors",
            "standalone": "my node logs EIO5 errors",
        },
        {"role": "assistant", "content": "..."},
    ]
    follow = await h.contextualize("and which firmware fixes it?", history)
    assert "firmware" in follow and "eio" in follow
    standalone = "How do I fix TLS handshake failures on the API gateway?"
    assert await h.contextualize(standalone, history) == standalone
    assert await h.contextualize("which firmware fixes it?", []) == "which firmware fixes it?"


@pytest.mark.parametrize(
    "question,tool,argument",
    [
        ("show me recent errors on storage-07", "live_logs", "storage-07"),
        ("what happened on rack-b4 in the logs?", "live_logs", "rack-b4"),
        ("if the queue depth is 64 what's 64/2?", "calculator", "64/2"),
        ("what's 15% of 400?", "calculator", "(15/100*400)"),
        ("what is 7 times 6", "calculator", "7 * 6"),
        ("thanks!", "direct", ""),
        ("what does EIO-5 mean?", "knowledge_base", "what does EIO-5 mean?"),
    ],
)
async def test_heuristic_router(question, tool, argument):
    d = await HeuristicReasoner().route(question, [])
    assert (d.tool, d.argument) == (tool, argument)


async def test_heuristic_relevance_by_term_coverage():
    h = HeuristicReasoner(relevance_threshold=0.5)
    good = await h.grade_relevance("EIO5 controller timeout firmware", [EIO])
    assert good.relevant and good.score == 1.0  # EIO5 is split into eio + 5
    bad = await h.grade_relevance("rotate kafka broker keystore", [PSU, EIO])
    assert not bad.relevant and bad.score == 0.0 and "kafka" in bad.reason
    none = await h.grade_relevance("anything", [])
    assert not none.relevant


async def test_heuristic_relevance_uses_calibrated_cross_encoder_scores():
    h = HeuristicReasoner(rerank_calibrated=True)
    hi = await h.grade_relevance("q", [{**EIO, "rerank_score": 6.0}, {**PSU, "rerank_score": -8.0}])
    lo = await h.grade_relevance("q", [{**EIO, "rerank_score": -4.0}])
    assert hi.relevant and hi.score > 0.99
    assert not lo.relevant and lo.score < 0.05


async def test_heuristic_rewrite_strategies():
    h = HeuristicReasoner()
    q = "hey so my box keeps spitting out EIO5, any idea what's going on??"
    first = await h.rewrite(q, q, 1, "low relevance", [])
    assert first.strategy == "keyword_focus"
    assert "eio 5" in first.query and "hey" not in first.query
    second = await h.rewrite(q, first.query, 2, "low relevance", [EIO])
    assert second.strategy == "pseudo_relevance_feedback"
    assert second.query.startswith(first.query) and len(second.query) > len(first.query)


async def test_heuristic_generation_is_extractive_and_grounded():
    h = HeuristicReasoner()
    ctx = ToolContext("knowledge_base", passages=[EIO, PSU])
    gen = await collect(h.generate, "which firmware has the timeout bug?", ctx, [])
    assert "4.2.1" in gen.text and "[1]" in gen.text and not gen.abstained
    grade = await h.grade_grounding("q", gen.text, ctx)
    assert grade.grounded and grade.score == 1.0


async def test_heuristic_abstains_on_insufficient_context():
    h = HeuristicReasoner()
    ctx = ToolContext("knowledge_base", passages=[PSU], insufficient=True)
    gen = await collect(h.generate, "rotate kafka keystore", ctx, [])
    assert gen.abstained and gen.text.startswith("I couldn't find")


async def test_heuristic_grounding_flags_unsupported_sentences():
    h = HeuristicReasoner()
    ctx = ToolContext("knowledge_base", passages=[EIO])
    answer = (
        "EIO-5 means the controller timed out waiting for a block write. "
        "Upgrading to firmware 5.0.2 from the vendor portal permanently resolves it."
    )
    grade = await h.grade_grounding("q", answer, ctx)
    assert not grade.grounded and grade.score == 0.5
    assert grade.unsupported_claims == [
        "Upgrading to firmware 5.0.2 from the vendor portal permanently resolves it."
    ]


# --- LLM reasoner plumbing (fake model, no network) -----------------------------


class FakeModel:
    name = "fake"

    def __init__(self, json_replies=None, stream_reply="", fail=False):
        self.json_replies = list(json_replies or [])
        self.stream_reply = stream_reply
        self.fail = fail
        self.calls: list[tuple[str, str, list]] = []

    async def complete_json(self, system, messages, schema, effort="low"):
        self.calls.append(("json", schema.__name__, messages))
        if self.fail:
            raise LLMError("boom")
        reply = self.json_replies.pop(0)
        return schema.model_validate(reply)

    async def stream_text(self, system, messages, effort="medium"):
        self.calls.append(("stream", system, messages))
        for word in self.stream_reply.split(" "):
            yield word + " "

    async def close(self):
        pass


def test_strict_schemas_are_closed_and_fully_required():
    for model in (RouteDecision, RelevanceGrade, Rewrite, GroundingGrade, StandaloneQuestion):
        schema = strict_json_schema(model)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    assert strict_json_schema(RouteDecision)["properties"]["tool"]["enum"] == [
        "knowledge_base",
        "live_logs",
        "calculator",
        "direct",
    ]


async def test_llm_reasoner_parses_decisions_and_clamps_scores():
    model = FakeModel(
        [
            {"tool": "live_logs", "argument": "storage-07", "reason": "host named"},
            {"score": 1.7, "relevant": True, "reason": "contains answer"},
            {"score": -0.2, "grounded": False, "unsupported_claims": ["x"], "reason": "made up"},
        ]
    )
    r = LLMReasoner(model)
    route = await r.route("errors on storage-07?", [])
    assert route.tool == "live_logs" and route.argument == "storage-07"
    rel = await r.grade_relevance("q", [EIO])
    assert rel.score == 1.0
    g = await r.grade_grounding("q", "answer", ToolContext("knowledge_base", [EIO]))
    assert g.score == 0.0 and not g.grounded
    assert "[1] Storage Node Runbook" in model.calls[1][2][0]["content"]  # passages numbered


async def test_llm_reasoner_contextualize_skips_llm_without_history():
    model = FakeModel([{"question": "Which firmware fixes EIO-5 timeouts?"}])
    r = LLMReasoner(model)
    assert await r.contextualize("which firmware fixes it?", []) == "which firmware fixes it?"
    assert model.calls == []
    history = [
        {"role": "user", "content": "EIO-5 errors?"},
        {"role": "assistant", "content": "controller timeout"},
    ]
    out = await r.contextualize("which firmware fixes it?", history)
    assert out == "Which firmware fixes EIO-5 timeouts?"
    assert "EIO-5 errors?" in model.calls[0][2][0]["content"]


async def test_llm_reasoner_streams_generation_with_history_and_context():
    model = FakeModel(stream_reply="I couldn't find the keystore procedure.")
    r = LLMReasoner(model, history_messages=2)
    history = [
        {"role": "assistant", "content": "dangling"},
        {"role": "user", "content": "u1", "standalone": "u1"},
        {"role": "assistant", "content": "a1", "tool": "knowledge_base"},
    ]
    gen = await collect(
        r.generate,
        "rotate keystore?",
        ToolContext("knowledge_base", [PSU], insufficient=True),
        history,
    )
    assert gen.abstained
    _, _, messages = model.calls[0]
    assert messages[0] == {"role": "user", "content": "u1"}  # extra keys stripped, starts with user
    assert "did not find clearly relevant" in messages[-1]["content"]
