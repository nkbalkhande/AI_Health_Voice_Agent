from urllib.parse import urlparse

import httpx

from backend.core.exceptions import ProviderNotConfiguredError
from backend.core.logger import logger
from backend.domain.interfaces.web_search import WebDocument, WebSearchProvider

log = logger.getChild("web.serpapi")


class SerpApiSearch(WebSearchProvider):
    """Internet search through SerpApi's Google engine."""

    def __init__(
        self,
        api_key: str,
        *,
        max_results: int = 5,
        timeout_seconds: float = 12,
        domains: list[str] | None = None,
        engine: str = "google",
        language: str = "en",
        country: str = "in",
    ) -> None:
        if not (api_key or "").strip():
            raise ProviderNotConfiguredError("SerpApi API key is not configured")
        self._api_key = api_key.strip()
        self._max_results = max_results
        self._timeout = timeout_seconds
        self._domains = [item.lower().lstrip(".") for item in (domains or []) if item]
        self._engine = engine or "google"
        self._language = language
        self._country = country

    async def search(self, query: str, *, limit: int | None = None) -> list[WebDocument]:
        query = (query or "").strip()
        if not query:
            return []
        cap = limit or self._max_results
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": self._engine,
                    "q": self._scoped_query(query),
                    "api_key": self._api_key,
                    "num": cap,
                    # Without hl/gl Google returned unrelated pages for health questions.
                    "hl": self._language,
                    "gl": self._country,
                },
            )
            if not response.is_success:
                log.warning("SerpApi request failed status=%s", response.status_code)
                response.raise_for_status()
            payload = response.json()
        documents = parse_serpapi_results(payload)
        if self._domains:
            documents = [item for item in documents if _host_allowed(item.url, self._domains)]
        return documents[:cap]

    def _scoped_query(self, query: str) -> str:
        if not self._domains:
            return query
        sites = " OR ".join(f"site:{domain}" for domain in self._domains)
        return f"{query} ({sites})"


def parse_serpapi_results(payload: dict) -> list[WebDocument]:
    documents: list[WebDocument] = []
    for item in (payload or {}).get("organic_results") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("link") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not url or not (title or snippet):
            continue
        documents.append(
            WebDocument(title=title or url, url=url, snippet=snippet or title)
        )
    return documents


def _host_allowed(url: str, domains: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith("." + domain) for domain in domains)
