"""Strip in-answer citation markup before semantic evaluation metrics.

Generated answers often append source/page markers or numeric cite tags.
Those are metadata for humans, not factual content the judge should score.
Reporting and API responses keep the original answer; only scoring inputs
are cleaned via :func:`strip_answer_citations`.
"""

from __future__ import annotations

import re

# (source: Doc.pdf page: 95) / (Source: Doc.pdf, pages 959, 963)
_PAREN_SOURCE_CITATION = re.compile(
    r"\s*\(\s*sources?\s*:\s*[^)]+\)",
    re.IGNORECASE,
)

# [Source: Doc.pdf, 1.200.001 Independence Rule]
_BRACKET_SOURCE_CITATION = re.compile(
    r"\s*\[\s*sources?\s*:\s*[^\]]+\]",
    re.IGNORECASE,
)

# [1], [12], [1], [3], [2][6] — digits only so Medicare [Part A] is preserved
_NUMERIC_CITATION_CLUSTER = re.compile(r"(?:\s*,?\s*\[\s*\d+\s*\])+")

# Optional trailing Evidence: / References: blocks (design-doc format)
_TRAILING_EVIDENCE_BLOCK = re.compile(
    r"(?is)\n\s*(?:Evidence|References|Citation)\s*:\s*(?:\n\s*[-*•].*)*\s*$",
)


def strip_answer_citations(text: str) -> str:
    """Return answer text with citation markup removed.

    Idempotent for answers that already lack citations. Does not mutate
    the caller's string beyond returning a cleaned copy.
    """
    if not text:
        return text

    cleaned = _PAREN_SOURCE_CITATION.sub("", text)
    cleaned = _BRACKET_SOURCE_CITATION.sub("", cleaned)
    cleaned = _NUMERIC_CITATION_CLUSTER.sub("", cleaned)
    cleaned = _TRAILING_EVIDENCE_BLOCK.sub("", cleaned)

    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    return cleaned.strip()
