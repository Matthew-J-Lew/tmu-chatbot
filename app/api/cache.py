from __future__ import annotations

import json
from typing import Any, Optional

from redis.asyncio import Redis

from app.api.config import REDIS_URL

_redis: Optional[Redis] = None


async def init_redis() -> None:
    """
    Initialize a global Redis client.

    Why this exists:
    - Many users ask the same questions.
    - Caching lets us answer repeat questions instantly, without re-running the LLM.
    """
    global _redis
    if _redis is not None:
        return
    _redis = Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None


def get_redis() -> Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized. Call init_redis() on startup.")
    return _redis


def make_cache_key(prefix: str, normalized_question: str) -> str:
    return f"{prefix}:{normalized_question}"


async def cache_get_json(key: str) -> Optional[dict[str, Any]]:
    r = get_redis()
    val = await r.get(key)
    if not val:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return None


async def cache_set_json(key: str, obj: dict, ttl_seconds: int):
    if not ttl_seconds or ttl_seconds <= 0:
        return
    r = get_redis()
    await r.set(key, json.dumps(obj), ex=int(ttl_seconds))

