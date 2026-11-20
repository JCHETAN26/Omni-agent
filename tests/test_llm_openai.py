import json

import httpx
from pydantic import BaseModel

from backend.agent.llm_openai import OpenAICompatibleChatModel
from backend.core import metrics


class Route(BaseModel):
    tool: str


def _tokens(kind: str) -> float:
    return metrics.LLM_TOKENS.labels(kind)._value.get()


def _model(handler, temperature=0.0):
    model = OpenAICompatibleChatModel(
        "https://example.test/v1", "k", "gpt-x", temperature=temperature
    )
    model.client = httpx.AsyncClient(
        base_url="https://example.test/v1", transport=httpx.MockTransport(handler)
    )
    return model


async def test_structured_call_sends_temperature_and_counts_usage():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"tool": "knowledge_base"}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 7},
            },
        )

    before = (_tokens("prompt"), _tokens("completion"))
    model = _model(handler)
    assert (
        await model.complete_json("sys", [{"role": "user", "content": "q"}], Route)
    ).tool == "knowledge_base"
    assert seen["temperature"] == 0.0 and seen["response_format"]["json_schema"]["strict"] is True
    assert (_tokens("prompt") - before[0], _tokens("completion") - before[1]) == (120, 7)
    await model.close()


async def test_stream_handles_the_usage_only_final_event():
    seen = {}
    events = [
        {"choices": [{"delta": {"content": "Check "}}]},
        {"choices": [{"delta": {"content": "canary health."}}]},
        {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 4}},
    ]

    def handler(request):
        seen.update(json.loads(request.content))
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    before = _tokens("prompt")
    model = _model(handler, temperature=None)
    text = "".join([d async for d in model.stream_text("sys", [{"role": "user", "content": "q"}])])
    assert text == "Check canary health."
    assert "temperature" not in seen and seen["stream_options"] == {"include_usage": True}
    assert _tokens("prompt") - before == 300
    await model.close()
