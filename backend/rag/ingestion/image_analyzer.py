"""Per-picture vision transcription for RAG ingestion.

Base64 is used only for the LLM request and is never returned on the
DocumentImageBlock.
"""

from __future__ import annotations

import base64
import io
from typing import Any, cast

from litellm import completion
from litellm.types.utils import Choices, ModelResponse

from backend.core.config import settings
from backend.core.logger import logger
from backend.evaluation.provider_credentials import azure_api_version, is_azure_model, model_id
from backend.infrastructure.litellm.model_resolver import resolve_llm_model
from backend.rag.ingestion.image_analysis_prompt import DEFAULT_IMAGE_ANALYSIS_PROMPT

log = logger.getChild("rag.ingestion.image_analyzer")


def pil_to_base64_png(image: Any) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def picture_page_no(picture: Any) -> int | None:
    """Docling page numbers are 1-based via provenance."""
    prov = getattr(picture, "prov", None) or []
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return int(page_no) if page_no is not None else None


def strip_code_fence(text: str) -> str:
    """Drop a wrapping ```/```markdown fence some models add despite the prompt."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _resolved_model() -> str:
    return resolve_llm_model(settings.llm_provider, settings.llm_model)


def analyze_image(
    *,
    page_content: str,
    image_b64: str,
    page_no: int | None = None,
    caption: str | None = None,
    model: str | None = None,
) -> str:
    """Send page markdown + base64 PNG to a vision LLM; return analysis text.

    Raises on transport / API failure so the caller can log and continue.
    """
    prompt = DEFAULT_IMAGE_ANALYSIS_PROMPT
    caption_line = f"Docling caption: {caption}\n" if caption else ""
    page_label = f"page {page_no}" if page_no is not None else "unknown page"

    user_text = (
        f"{prompt}\n\n"
        f"--- Background page context ({page_label}), NOT a source for the "
        "transcription ---\n"
        f"{caption_line}"
        f"{page_content if page_content.strip() else '[empty page text]'}\n"
        f"--- End page context ---\n\n"
        "Transcribe the attached image only."
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    litellm_model = model or _resolved_model()
    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "max_completion_tokens": 4096,
        "temperature": 0.0,
        "drop_params": True,
        "timeout": settings.llm_timeout,
    }

    if is_azure_model(litellm_model):
        endpoint = (settings.azure_openai_endpoint or "").strip().rstrip("/")
        api_key = (settings.azure_openai_api_key or "").strip()
        version = azure_api_version() or "2025-03-01-preview"
        if not endpoint or not api_key:
            raise RuntimeError(
                "Image analysis requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )
        kwargs["api_base"] = endpoint
        kwargs["api_key"] = api_key
        kwargs["api_version"] = version
        # Ensure deployment id is present for azure/<deployment> ids.
        if "/" not in litellm_model:
            kwargs["model"] = f"azure/{model_id(litellm_model)}"

    log.info("Analyzing image on %s via %s", page_label, kwargs["model"])
    response = cast(ModelResponse, completion(**kwargs))
    choice = cast(Choices, response.choices[0])
    return strip_code_fence((choice.message.content or "").strip())
