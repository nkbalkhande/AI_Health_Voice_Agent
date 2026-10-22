import asyncio
from typing import Any

import litellm
from litellm import acompletion

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.interfaces.llm_provider import GenerationResult, LLMProvider
from backend.infrastructure.litellm.model_resolver import azure_kwargs, resolve_llm_model

litellm.suppress_debug_info = True
litellm.drop_params = True

log = logger.getChild("infrastructure.llm.litellm")

# Re-exported here so callers/tests can import the contract alongside the
# provider, while the canonical definition lives in the domain layer.
__all__ = ["GenerationResult", "LiteLLMProvider"]

# Transient errors worth a single retry. These are infrastructure hiccups
# (network, throttling, upstream 5xx), not problems with the request itself.
# Seconds to wait before the single retry on a transient failure (R-62).
_RETRY_BACKOFF_SECONDS = 5.0

# litellm.Timeout is handled separately -> status="timeout".
_TRANSIENT_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)

# Statuses that indicate a recoverable, non-deterministic failure. Retried at
# most once. "refused"/"generation_error" are intentionally excluded: a refusal
# is a deliberate model decision and a generation_error means the request was
# rejected (bad request / validation), so retrying is pointless or harmful.
_RETRYABLE_STATUSES = frozenset(
    {"empty_generation", "incomplete_length", "timeout", "infrastructure_error"}
)

# finish_reason values that mean the model ran out of output max_tokens.
_LENGTH_FINISH_REASONS = frozenset({"length", "max_tokens"})


class LiteLLMProvider(LLMProvider):
    """
    Unified LLM via LiteLLM — any supported model via config, no per-vendor classes.

    Supports single-turn (generate) and multi-turn (chat) for threaded conversations.
    Every call returns a :class:`GenerationResult` describing exactly how the
    request ended, with at most one retry for transient/recoverable failures.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> None:
        llm_provider = provider or settings.llm_provider
        llm_model = model or settings.llm_model
        self._model = resolve_llm_model(llm_provider, llm_model)
        self._max_completion_tokens = (
            max_completion_tokens or settings.llm_max_completion_tokens
        )
        self._reasoning_effort = settings.llm_reasoning_effort
        self._timeout = settings.llm_timeout
        self._extra_kwargs = settings.litellm_llm_kwargs

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, max_tokens=max_tokens)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        max_tokens = max_tokens or self._max_completion_tokens

        # At most one retry: attempt 0 (initial) + attempt 1 (single retry).
        result = await self._attempt(messages, max_tokens)
        if result.status in _RETRYABLE_STATUSES:
            # R-62: the retry used to fire immediately with identical
            # parameters, so a 429 window had no time to clear and a
            # length-truncated answer was retried into the same ceiling.
            retry_budget = max_tokens
            if result.status == "incomplete_length" and max_tokens:
                retry_budget = max_tokens * 2
                log.warning(
                    "Generation truncated (max_completion_tokens=%s); retrying "
                    "once with %s.",
                    max_tokens,
                    retry_budget,
                )
            else:
                log.warning(
                    "Generation ended with status=%s; backing off %.1fs then "
                    "retrying once.",
                    result.status,
                    _RETRY_BACKOFF_SECONDS,
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
            result = await self._attempt(messages, retry_budget)
        return result

    async def _attempt(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
    ) -> GenerationResult:
        """One call to the model, always resolving to a GenerationResult.

        Transient infrastructure errors map to ``timeout`` /
        ``infrastructure_error`` so callers (and the refusal scorecard, R-09)
        can distinguish them from blank model replies (``empty_generation``)
        and from refusals; the retry policy still re-attempts them once.
        Deterministic request errors (bad request, validation, auth, context
        window) map to ``generation_error`` and are never retried.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "drop_params": True,
            # R-87: without this, litellm waits 6000s before raising Timeout,
            # so the status="timeout" path below was effectively unreachable.
            "timeout": self._timeout,
            **azure_kwargs(self._model),
            **self._extra_kwargs,
        }
        # R-57: llm.reasoning_effort was read into the provider and never
        # sent. drop_params already guards models that do not accept it.
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        try:
            response = await acompletion(**kwargs)
        except litellm.Timeout as exc:
            log.warning("LLM timeout: %s", exc)
            return GenerationResult(
                text="",
                status="timeout",
                error=str(exc),
            )
        except _TRANSIENT_EXCEPTIONS as exc:
            log.warning("Transient LLM infrastructure error: %s", exc)
            return GenerationResult(
                text="",
                status="infrastructure_error",
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all -> error state
            # BadRequest, ContextWindowExceeded, Authentication, schema errors, ...
            log.error("Non-retryable LLM error: %s", exc)
            return GenerationResult(
                text="",
                status="generation_error",
                error=str(exc),
            )

        return self._build_result(response)

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ):
        max_tokens = max_tokens or self._max_completion_tokens
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "drop_params": True,
            "timeout": self._timeout,
            **azure_kwargs(self._model),
            **self._extra_kwargs,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        response = await acompletion(**kwargs)
        async for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if text:
                yield text

    @staticmethod
    def _build_result(response: Any) -> GenerationResult:
        choice = response.choices[0]
        message = choice.message
        text = (getattr(message, "content", None) or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        reasoning_tokens = None
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

        request_id = getattr(response, "id", None)

        # R-85: litellm can price the response from its model map; a missing
        # or unpriced model must not break generation.
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - pricing is best-effort
            cost = None

        common = {
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "request_id": request_id,
            "cost": cost,
        }

        # 1) Explicit refusal (e.g. GPT-5 safety refusal) — distinct from empty.
        refusal = getattr(message, "refusal", None)
        if refusal:
            return GenerationResult(
                text="", status="refused", refusal=str(refusal), **common
            )

        # 2) No usable text — distinguish "ran out of tokens" from "said nothing".
        if not text:
            status = (
                "incomplete_length"
                if finish_reason in _LENGTH_FINISH_REASONS
                else "empty_generation"
            )
            return GenerationResult(text="", status=status, **common)

        # 3) Truncated but with partial text -> still incomplete.
        if finish_reason in _LENGTH_FINISH_REASONS:
            return GenerationResult(
                text=text, status="incomplete_length", **common
            )

        return GenerationResult(text=text, status="answered", **common)
