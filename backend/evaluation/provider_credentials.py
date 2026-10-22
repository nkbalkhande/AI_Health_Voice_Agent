"""Provider routing and credential validation shared by the evaluation runners.

There is no single "evaluation provider" setting to validate against: the
provider is carried by each LiteLLM model id (``azure/<deployment>`` vs
``openai/<model>``), and a runner may resolve several models independently --
ragas routes its judge and its embeddings separately. Validation therefore
folds over the models actually in use and checks only the credentials those
providers need.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from backend.core.config import settings

_AZURE_PREFIX = "azure/"

# Environment variable name → settings attribute holding its value.
_AZURE_REQUIRED_CREDENTIALS = (
    ("AZURE_OPENAI_API_KEY", "azure_openai_api_key"),
    ("AZURE_OPENAI_ENDPOINT", "azure_openai_endpoint"),
)


def _clean(value: str | None) -> str:
    """Normalize an externally-supplied value; blank and whitespace-only alike."""
    return (value or "").strip()


def is_azure_model(model: str) -> bool:
    """True when a LiteLLM id routes to Azure OpenAI (``azure/<deployment>``)."""
    return _clean(model).lower().startswith(_AZURE_PREFIX)


def model_id(model: str) -> str:
    """Strip the LiteLLM provider prefix (``azure/gpt-5.1`` → ``gpt-5.1``)."""
    text = _clean(model)
    if "/" in text:
        return text.split("/", 1)[1]
    return text


def azure_api_version() -> str | None:
    """API version for Azure clients.

    Prefers ``AZURE_OPENAI_API_VERSION`` (via ``settings.azure_openai_api_version``)
    and falls back to the generic ``OPENAI_API_VERSION`` env var so deployments
    that only set the latter keep working. Returns ``None`` when neither holds
    a non-blank value.

    Note: ``settings.openai_api_version`` is only a back-compat alias for the
    Azure field (R-50), so the env fallback must read ``OPENAI_API_VERSION``
    directly — otherwise clearing the Azure field cannot fall through.
    """
    return (
        _clean(settings.azure_openai_api_version)
        or _clean(os.getenv("OPENAI_API_VERSION"))
        or None
    )


def _describe(models: Mapping[str, str]) -> str:
    """Render ``{"judge": "azure/gpt-5.1"}`` as ``judge (model=azure/gpt-5.1)``."""
    parts = [
        f"{role} (model={_clean(model) or '<unset>'})" for role, model in models.items()
    ]
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def missing_credentials(
    models: Mapping[str, str],
    *,
    runner: str,
    azure_hint: str | None = None,
) -> str | None:
    """Return an error message when the in-use providers lack credentials.

    ``models`` maps a role name (``judge``, ``embeddings``) to its LiteLLM id.
    Only providers that at least one model routes to are validated, so an
    Azure-only deployment never needs ``OPENAI_API_KEY``. Ids without an
    ``azure/`` prefix — including blank ones — route to OpenAI, which is where
    the clients would actually be built.

    Azure is checked first and returns eagerly: when Azure credentials are
    incomplete the caller gets an Azure-specific message and no OpenAI
    validation is performed at all. ``azure_hint`` appends runner-specific
    routing guidance to that message.
    """
    azure_models = {
        role: model for role, model in models.items() if is_azure_model(model)
    }
    openai_models = {
        role: model for role, model in models.items() if not is_azure_model(model)
    }

    if azure_models:
        missing = [
            variable
            for variable, attribute in _AZURE_REQUIRED_CREDENTIALS
            if not _clean(getattr(settings, attribute, None))
        ]
        if missing:
            message = (
                f"{', '.join(missing)} required for the Azure "
                f"{_describe(azure_models)} used by the {runner} evaluator."
            )
            return f"{message} {azure_hint}" if azure_hint else message

    if openai_models and not _clean(settings.openai_api_key):
        return (
            f"OPENAI_API_KEY is required for the {_describe(openai_models)} "
            f"used by the {runner} evaluator."
        )

    return None
