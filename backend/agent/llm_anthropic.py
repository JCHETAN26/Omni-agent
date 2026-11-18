"""Claude via the official Anthropic SDK (AsyncAnthropic).

* Model: claude-opus-5-5 by default. Thinking is always on for this model; depth
  is controlled with `output_config.effort` (low for routing/grading, medium for
  answers), which is set explicitly on every call.
* Structured steps (router, graders, rewriter) use JSON-schema structured outputs.
* Answers stream with `messages.stream(...).text_stream`.
* Server-side refusal fallbacks are enabled (`fallbacks="default"`): if Claude
  declines on policy grounds, the API re-runs the request on a fallback model
  inside the same call. A refusal of the whole chain surfaces as `LLMRefusal`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import anthropic

from backend.agent.llm import Effort, LLMError, LLMRefusal, Message, T, strict_json_schema

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _refusal_category(message) -> str | None:
    details = getattr(message, "stop_details", None)
    return getattr(details, "category", None) if details else None


class AnthropicChatModel:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ):
        kwargs = {"timeout": timeout_s, "max_retries": max_retries}
        # No key given: the SDK resolves ANTHROPIC_API_KEY / auth token / `ant` profile.
        self.client = (
            anthropic.AsyncAnthropic(api_key=api_key, **kwargs)
            if api_key
            else anthropic.AsyncAnthropic(**kwargs)
        )
        self.model = model
        self.name = f"anthropic:{model}"

    async def complete_json(
        self, system: str, messages: list[Message], schema: type[T], effort: Effort = "low"
    ) -> T:
        try:
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": strict_json_schema(schema)},
                },
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"connection error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMRefusal(_refusal_category(response))
        if response.stop_reason == "max_tokens":
            raise LLMError("structured output truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("no text block in structured response")
        try:
            return schema.model_validate_json(text)
        except ValueError as exc:
            raise LLMError(f"invalid structured output: {exc}") from exc

    async def stream_text(
        self, system: str, messages: list[Message], effort: Effort = "medium"
    ) -> AsyncIterator[str]:
        try:
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=64000,
                system=system,
                messages=messages,
                output_config={"effort": effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"connection error: {exc}") from exc
        if final.stop_reason == "refusal":
            raise LLMRefusal(_refusal_category(final))

    async def close(self) -> None:
        await self.client.close()
