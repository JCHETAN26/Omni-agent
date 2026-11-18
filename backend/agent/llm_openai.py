"""OpenAI-compatible Chat Completions client (OpenAI, vLLM, other compatible servers).

Structured steps use `response_format: json_schema` (vLLM serves this through
guided decoding). Answers stream over SSE.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from backend.agent.llm import Effort, LLMError, Message, T, strict_json_schema


class OpenAICompatibleChatModel:
    def __init__(self, base_url: str, api_key: str | None, model: str, timeout_s: float = 60.0):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )
        self.model = model
        self.name = f"openai-compatible:{model}"

    async def complete_json(
        self, system: str, messages: list[Message], schema: type[T], effort: Effort = "low"
    ) -> T:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": strict_json_schema(schema),
                    "strict": True,
                },
            },
        }
        try:
            r = await self.client.post("/chat/completions", json=body)
            r.raise_for_status()
            choice = r.json()["choices"][0]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"chat completion failed: {exc}") from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("structured output truncated")
        try:
            return schema.model_validate_json(choice["message"]["content"] or "")
        except ValueError as exc:
            raise LLMError(f"invalid structured output: {exc}") from exc

    async def stream_text(
        self, system: str, messages: list[Message], effort: Effort = "medium"
    ) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
        }
        try:
            async with self.client.stream("POST", "/chat/completions", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"chat stream failed: {exc}") from exc

    async def close(self) -> None:
        await self.client.aclose()
