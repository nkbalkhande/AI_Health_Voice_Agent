import json
from typing import Any

import redis.asyncio as redis

from backend.core.logger import logger
from backend.domain.interfaces.cache import CacheStore

log = logger.getChild("cache.redis")


class RedisCache(CacheStore):
    """Redis for temporary state only. PostgreSQL remains the durable store."""

    def __init__(self, url: str, *, ttl_seconds: int = 300) -> None:
        self._client = redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds

    async def get_json(self, key: str) -> Any | None:
        try:
            raw = await self._client.get(key)
        except Exception:
            log.warning("Redis read failed; treating as a cache miss", exc_info=True)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(self, key: str, value: Any) -> None:
        try:
            await self._client.set(key, json.dumps(value), ex=self._ttl)
        except Exception:
            log.warning("Redis write failed; continuing without cache", exc_info=True)
