"""Provider-neutral chat-model interface used by the agent's LLM reasoner.

Implementations:
* `backend.agent.llm_anthropic.AnthropicChatModel`: Claude (official SDK)
* `backend.agent.llm_openai.OpenAICompatibleChatModel`: OpenAI / vLLM
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

from backend.core.config import Settings

T = TypeVar("T", bound=BaseModel)
Effort = Literal["low", "medium", "high"]
Message = dict[str, str]  # {"role": "user" | "assistant", "content": "..."}


class LLMError(RuntimeError):
    """Any failure to get a usable answer (network, API error, bad JSON, truncation)."""


class LLMRefusal(LLMError):
    def __init__(self, category: str | None):
        super().__init__(f"model declined the request (category={category})")
        self.category = category


class ChatModel(Protocol):
    name: str

    async def complete_json(
        self, system: str, messages: list[Message], schema: type[T], effort: Effort = "low"
    ) -> T: ...

    def stream_text(
        self, system: str, messages: list[Message], effort: Effort = "medium"
    ) -> AsyncIterator[str]: ...

    async def close(self) -> None: ...


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema -> strict JSON schema (all fields required, no extras)."""

    def fix(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for key in ("title", "default"):
                node.pop(key, None)
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
        return node

    schema = copy.deepcopy(model.model_json_schema())
    if "$defs" in schema:
        raise ValueError(f"{model.__name__}: nested models are not supported in strict schemas")
    return fix(schema)


def chat_model_from_settings(s: Settings) -> ChatModel | None:
    if s.llm_provider == "anthropic":
        from backend.agent.llm_anthropic import AnthropicChatModel

        return AnthropicChatModel(s.anthropic_model, s.anthropic_api_key, s.llm_timeout_s)
    if s.llm_provider == "openai":
        from backend.agent.llm_openai import OpenAICompatibleChatModel

        if not s.llm_model:
            raise ValueError("LLM_PROVIDER=openai requires LLM_MODEL")
        return OpenAICompatibleChatModel(
            s.llm_base_url,
            s.llm_api_key or s.openai_api_key,
            s.llm_model,
            s.llm_timeout_s,
            s.llm_temperature,
        )
    return None
