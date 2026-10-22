import json
from typing import Any

UNDERSTAND_SYSTEM = """You resolve the latest user message in an ongoing healthcare conversation.

Return one JSON object and nothing else:
{
  "intent": "question | follow_up | correction | greeting | other",
  "standalone_query": "the message rewritten so it can be searched without the chat",
  "needs_retrieval": true,
  "memory_updates": [{"key": "short name", "value": "fact the user stated or corrected"}],
  "summary": "one or two sentences describing the conversation so far"
}

Rules:
- Resolve references such as "it", "that", "the previous one", and "what about treatment" from the history, summary, and facts.
- standalone_query must stand alone. Keep the user's language.
- In standalone_query, fix spelling mistakes and speech-to-text errors in medical words (for example "symptons" becomes "symptoms", "diabities" becomes "diabetes"). Do not change the meaning.
- needs_retrieval is false for greetings, thanks, and replies that the conversation already answers.
- needs_retrieval is true for health questions: symptoms, conditions, causes, tests, treatments, medicines, prevention, diet, and when to see a doctor.
- memory_updates keep durable details the user states about themselves, such as age, symptoms, conditions, medications, allergies, pregnancy, and preferences. A correction replaces the older value under the same key. Use [] when nothing durable changed.
- summary includes the current goal and preferences. Do not invent details the user did not say.
"""


def parse_understanding(raw: str, fallback_query: str) -> dict[str, Any]:
    """Read the model's JSON. Any failure keeps the raw message searchable."""
    data = _load_json_object(raw)
    standalone = str(data.get("standalone_query") or "").strip() or fallback_query
    needs = data.get("needs_retrieval")
    if not isinstance(needs, bool):
        needs = True
    updates = data.get("memory_updates")
    if not isinstance(updates, list):
        updates = []
    summary = data.get("summary")
    return {
        "intent": str(data.get("intent") or "question"),
        "standalone_query": standalone,
        "needs_retrieval": needs,
        "memory_updates": [item for item in updates if isinstance(item, dict)],
        "updated_summary": summary.strip() if isinstance(summary, str) else "",
    }


def merge_facts(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Later values for the same key replace earlier ones."""
    order: list[str] = []
    values: dict[str, str] = {}
    for item in [*existing, *updates]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not key or not value:
            continue
        if key not in values:
            order.append(key)
        values[key] = value
    return [{"key": key, "value": values[key]} for key in order]


def _load_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}
