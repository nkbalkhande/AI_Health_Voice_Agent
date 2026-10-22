"""Resolve LiteLLM model strings from config provider + model."""

_LLM_PREFIX = {
    "anthropic": "anthropic",
    "openai": "openai",
    "azure": "azure",
    "azure_openai": "azure",
    "gemini": "gemini",
    "google": "gemini",
    "bedrock": "bedrock",
    "cohere": "cohere",
    "mistral": "mistral",
    "ollama": "ollama",
    "groq": "groq",
    "deepseek": "deepseek",
}

_EMBEDDING_PREFIX = {
    "openai": "openai",
    "azure": "azure",
    "azure_openai": "azure",
    "cohere": "cohere",
    "voyage": "voyage",
    "bedrock": "bedrock",
    "gemini": "gemini",
    "huggingface": "huggingface",
}


def resolve_llm_model(provider: str, model: str) -> str:
    """
    Return a LiteLLM model id.

    Prefer full ids in config (e.g. anthropic/claude-sonnet-4-20250514).
    Legacy: provider=anthropic + model=claude-sonnet-4-20250514.
    """
    if "/" in model:
        return model
    prefix = _LLM_PREFIX.get(provider.lower())
    if prefix:
        return f"{prefix}/{model}"
    return model


def azure_kwargs(model: str) -> dict[str, str]:
    """litellm reads AZURE_API_*, not the AZURE_OPENAI_* names used in .env."""
    if not model.lower().startswith("azure/"):
        return {}
    from backend.core.config import settings

    values = {
        "api_base": settings.azure_openai_endpoint,
        "api_key": settings.azure_openai_api_key,
        "api_version": settings.azure_openai_api_version,
    }
    return {key: value for key, value in values.items() if value}


def resolve_embedding_model(provider: str, model: str) -> str:
    """
    Return a LiteLLM embedding model id.

    Examples: openai/text-embedding-3-small, cohere/embed-english-v3.0
    """
    if "/" in model:
        return model
    prefix = _EMBEDDING_PREFIX.get(provider.lower())
    if prefix:
        return f"{prefix}/{model}"
    return model
