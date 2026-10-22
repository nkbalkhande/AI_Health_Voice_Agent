from backend.core.logger import logger
from backend.domain.interfaces.cache import CacheStore
from backend.domain.interfaces.web_search import WebDocument, WebSearchProvider

log = logger.getChild("knowledge.web")


class WebKnowledgeBranch:
    """Branch B — internet search only (SerpApi).

    Returns web pages as their own type. It never sees document chunks,
    so the search provider, domains, and cache can change without
    touching the RAG branch.
    """

    def __init__(
        self,
        provider: WebSearchProvider | None,
        *,
        cache: CacheStore | None = None,
    ) -> None:
        self._provider = provider
        self._cache = cache

    async def search(self, query: str) -> list[WebDocument]:
        if self._provider is None:
            return []
        cache_key = f"web:{' '.join(query.lower().split())}"
        cached = await self._read_cache(cache_key)
        if cached is not None:
            return cached
        try:
            pages = await self._provider.search(query)
        except Exception:
            log.warning("Web search failed for a turn", exc_info=True)
            return []
        pages = _dedupe(pages)
        if pages:
            await self._write_cache(cache_key, pages)
        return pages

    async def _read_cache(self, key: str) -> list[WebDocument] | None:
        if self._cache is None:
            return None
        cached = await self._cache.get_json(key)
        if not isinstance(cached, list):
            return None
        return [
            WebDocument(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
            )
            for item in cached
            if isinstance(item, dict)
        ]

    async def _write_cache(self, key: str, pages: list[WebDocument]) -> None:
        if self._cache is None:
            return
        await self._cache.set_json(
            key,
            [
                {"title": page.title, "url": page.url, "snippet": page.snippet}
                for page in pages
            ],
        )


def _dedupe(pages: list[WebDocument]) -> list[WebDocument]:
    seen: set[str] = set()
    unique: list[WebDocument] = []
    for page in pages:
        key = page.url or " ".join(page.snippet.lower().split())[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(page)
    return unique
