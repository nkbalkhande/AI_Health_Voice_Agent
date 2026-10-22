"""Shared judge-model configuration for the evaluation runners.

Both runners talk to the same judge LLM (``evaluation.model`` in
``config/settings.yaml``) but used to resolve its credentials in two different,
partly-wrong ways. This module owns that resolution so ragas and deepeval agree:

* :func:`_missing_judge_credentials` gates on the credentials the *resolved*
  models actually need instead of unconditionally demanding ``OPENAI_API_KEY``
  (R-21 / R-54), and fails fast when Azure credentials are absent rather than
  burning a full pipeline run before the first judge call errors.
* :func:`_resolve_judge_litellm_client_kwargs` wires ``base_url`` and
  ``api_version`` for ``azure/*`` judges — LiteLLM otherwise only resolves the
  endpoint if ``AZURE_API_BASE`` happens to be exported, which nothing in this
  repo sets or documents (R-02) — and applies the configured output ceiling
  (R-03).

``ragas_runner`` imports this module and ``deepeval_runner`` re-exports the
helpers, so ``deepeval_runner`` can keep importing from ``ragas_runner``
without a cycle.
"""

from __future__ import annotations

from typing import Any

from backend.core.config import settings

# Reasoning models reject non-default sampler args and use
# max_completion_tokens instead of max_tokens.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _openai_model_id(model: str) -> str:
    """Strip the LiteLLM provider prefix (e.g. ``openai/gpt-4.1`` -> ``gpt-4.1``)."""
    text = (model or "").strip()
    if "/" in text:
        return text.split("/", 1)[1]
    return text


def _is_azure_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("azure/")


def _is_reasoning_model(model: str) -> bool:
    return _openai_model_id(model).startswith(_REASONING_MODEL_PREFIXES)


def _missing_judge_credentials(
    judge_model: str,
    *,
    embedding_model: str | None = None,
) -> str | None:
    """Return a human-readable message naming missing credentials, else ``None``.

    Gates on what the resolved models need: ``azure/*`` models require the
    Azure key + endpoint, everything else requires ``OPENAI_API_KEY``. The old
    blunt gate demanded ``OPENAI_API_KEY`` even on an all-Azure config (which
    never uses it) while never checking the Azure credentials it does use.
    """
    models = [judge_model] + ([embedding_model] if embedding_model else [])
    missing: list[str] = []
    if any(_is_azure_model(m) for m in models):
        missing.extend(
            name
            for name, value in (
                ("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key),
                ("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
            )
            if not value
        )
    if any(not _is_azure_model(m) for m in models) and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not missing:
        return None
    return (
        f"Missing judge credentials for {judge_model}: " + ", ".join(missing)
    )


def _resolve_judge_litellm_client_kwargs(judge_model: str) -> dict[str, Any]:
    """Build the LiteLLMModel kwargs for *judge_model*.

    Large retrieval contexts produce long per-statement verdict JSON; without
    an explicit ceiling the judge call truncates mid-JSON and the parser has to
    repair a partial object, silently depressing scores. The ceiling is
    ``evaluation.deepeval.evaluation_judge_max_tokens``.
    """
    generation_kwargs: dict[str, Any] = {
        "timeout": settings.deepeval_per_task_timeout_seconds,
        "num_retries": 0,
        "max_completion_tokens": settings.evaluation_judge_max_tokens,
    }
    if _is_azure_model(judge_model):
        generation_kwargs["api_version"] = settings.azure_openai_api_version
        return {
            "api_key": settings.azure_openai_api_key,
            "base_url": settings.azure_openai_endpoint,
            "generation_kwargs": generation_kwargs,
        }
    return {
        "api_key": settings.openai_api_key,
        "base_url": None,
        "generation_kwargs": generation_kwargs,
    }


def _judge_llm_kwargs(model: str) -> dict[str, Any]:
    """llm_factory kwargs for the ragas judge.

    Classic models get deterministic judging (configured temperature + seed)
    and the configured output ceiling. Reasoning models get ``{}`` —
    :func:`_sanitize_reasoning_judge` fixes their ``model_args`` after
    ``llm_factory`` has injected its own defaults.
    """
    if _is_reasoning_model(model):
        return {}
    return {
        "max_tokens": settings.evaluation_judge_max_tokens,
        "temperature": settings.evaluation_temperature,
        "seed": 42,
    }


def _sanitize_reasoning_judge(llm: Any) -> None:
    """Strip sampler args GPT-5/o-series reject; force max_completion_tokens.

    ragas' instructor adapter always merges ``InstructorModelArgs()`` defaults
    (including ``max_tokens=1024``) into ``model_args``, so the ceiling has to
    be re-applied here rather than relying on a pop-with-default.
    """
    model_args = getattr(llm, "model_args", None)
    if not isinstance(model_args, dict):
        return
    model_args.pop("top_p", None)
    model_args.pop("max_tokens", None)
    model_args["temperature"] = 1.0
    model_args["max_completion_tokens"] = settings.evaluation_judge_max_tokens
