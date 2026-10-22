import re
from html import unescape
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from backend.domain.interfaces.web_search import WebDocument, WebSearchProvider

_RESULT_LINK = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</(?:a|td|span|div)>',
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")


class DuckDuckGoSearch(WebSearchProvider):
    def __init__(
        self,
        *,
        max_results: int = 5,
        timeout_seconds: float = 12,
        domains: list[str] | None = None,
    ) -> None:
        self._max_results = max_results
        self._timeout = timeout_seconds
        self._domains = [item.lower().lstrip(".") for item in (domains or []) if item]

    async def search(self, query: str, *, limit: int | None = None) -> list[WebDocument]:
        query = (query or "").strip()
        if not query:
            return []
        cap = limit or self._max_results
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as client:
            response = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
            )
            response.raise_for_status()
        documents = parse_duckduckgo_html(response.text)
        if self._domains:
            documents = [item for item in documents if _host_allowed(item.url, self._domains)]
        return documents[:cap]


def parse_duckduckgo_html(html: str) -> list[WebDocument]:
    links = _RESULT_LINK.findall(html or "")
    snippets = [_clean(match) for match in _SNIPPET.findall(html or "")]
    documents: list[WebDocument] = []
    for index, (href, title_html) in enumerate(links):
        url = _unwrap_url(unescape(href))
        title = _clean(title_html)
        snippet = snippets[index] if index < len(snippets) else ""
        if not url or not (title or snippet):
            continue
        documents.append(WebDocument(title=title or url, url=url, snippet=snippet or title))
    return documents


def _unwrap_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    target = parse_qs(parsed.query).get("uddg")
    if target:
        return unquote(target[0])
    return url


def _clean(value: str) -> str:
    return " ".join(unescape(_TAG.sub(" ", value or "")).split())


def _host_allowed(url: str, domains: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith("." + domain) for domain in domains)
