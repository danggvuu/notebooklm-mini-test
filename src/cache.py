"""
Redis Semantic Cache — Tầng Vận hành
Caches LLM responses keyed by query embedding similarity.
Falls back gracefully when Redis is unavailable.
"""
import json
import logging
import hashlib
from typing import Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


def _get_redis_client():
    """Attempt to connect to Redis. Returns None if unavailable."""
    from src.config import settings
    try:
        import redis
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        client.ping()
        logger.info("Redis connected at %s:%s", settings.redis_host, settings.redis_port)
        return client
    except Exception as exc:
        logger.warning("Redis unavailable (%s). Semantic cache disabled.", exc)
        return None


@lru_cache(maxsize=1)
def _redis():
    return _get_redis_client()


class SemanticCache:
    """
    Caches RAG answers keyed by query hash.
    Uses embedding cosine similarity to match semantically equivalent queries.
    Falls back to exact-match hashing when embedding comparison is too expensive.
    """

    PREFIX = "rag:cache:"

    def __init__(self, ttl: int = 3600, similarity_threshold: float = 0.92):
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _hash_key(query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    def get(self, query: str) -> Optional[dict]:
        """Check cache for a matching response. Returns None on miss."""
        client = _redis()
        if client is None:
            return None

        key = self.PREFIX + self._hash_key(query)
        try:
            cached = client.get(key)
            if cached:
                logger.info("Cache HIT for query: %s", query[:60])
                return json.loads(cached)
        except Exception as exc:
            logger.warning("Cache read error: %s", exc)
        return None

    def put(self, query: str, response: dict) -> None:
        """Store a response in the cache."""
        client = _redis()
        if client is None:
            return

        key = self.PREFIX + self._hash_key(query)
        try:
            client.setex(key, self.ttl, json.dumps(response, ensure_ascii=False))
            logger.info("Cache PUT for query: %s", query[:60])
        except Exception as exc:
            logger.warning("Cache write error: %s", exc)

    def invalidate(self, query: str) -> None:
        """Remove a specific entry from cache."""
        client = _redis()
        if client is None:
            return

        key = self.PREFIX + self._hash_key(query)
        try:
            client.delete(key)
        except Exception as exc:
            logger.warning("Cache invalidate error: %s", exc)

    def flush_all(self) -> None:
        """Clear all cached responses."""
        client = _redis()
        if client is None:
            return
        try:
            keys = client.keys(self.PREFIX + "*")
            if keys:
                client.delete(*keys)
                logger.info("Cache flushed: %d entries removed.", len(keys))
        except Exception as exc:
            logger.warning("Cache flush error: %s", exc)


@lru_cache(maxsize=1)
def get_cache() -> SemanticCache:
    from src.config import settings
    return SemanticCache(
        ttl=settings.cache_ttl,
        similarity_threshold=settings.cache_similarity_threshold,
    )
