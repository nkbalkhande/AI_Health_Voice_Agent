import json
import re
from typing import Any

_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def response_to_json_safe(content: str) -> dict[str, Any]:
    """Parse LLM output into a dict, tolerating fenced JSON."""
    text = (content or "").strip()
    if not text:
        return {}

    fence_match = _JSON_FENCE_PATTERN.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}

    return parsed if isinstance(parsed, dict) else {}


def build_page_context(
    all_pages: list[str],
    page_index: int,
    *,
    current_page_content: str,
    max_chars: int,
    include_neighbors: bool,
) -> str:
    """Build LLM page context from neighboring pages and the current page body."""
    parts: list[str] = []

    # R-29: only the CURRENT page was truncated, so a page with long
    # neighbours resent far more than max_chars on every chunk call. Each
    # neighbour gets a quarter of the budget, taking the tail of the previous
    # page and the head of the next (the text nearest the current page).
    neighbor_chars = max(0, max_chars // 4) if include_neighbors else 0

    if include_neighbors and page_index > 0:
        parts.append(
            all_pages[page_index - 1][-neighbor_chars:] if neighbor_chars else ""
        )

    parts.append(current_page_content[:max_chars])

    if include_neighbors and page_index + 1 < len(all_pages):
        parts.append(all_pages[page_index + 1][:neighbor_chars])

    return "\n\n".join(part for part in parts if part.strip())


def apply_enrichment(chunk: str, enrich: dict[str, Any]) -> str:
    """Merge context_block into chunk using the insert directive."""
    context = str(enrich.get("context_block", "")).strip()
    if not context:
        return chunk

    insert = str(enrich.get("insert", "prefix")).strip().lower()
    if insert == "suffix":
        return f"{chunk}\n\n{context}"
    if insert == "none":
        return chunk
    return f"{context}\n{chunk}"
