from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

import httpx

from app.api.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_MAX_RETRIES,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
)

_client: Optional[httpx.AsyncClient] = None


async def init_ollama_client() -> None:
    """
    Create a shared HTTP client.

    Why this exists:
    - Creating a brand-new HTTP client for every request is slow.
    - A shared client reuses connections (faster and more stable).
    """
    global _client
    if _client is not None:
        return

    timeout = httpx.Timeout(
        connect=10.0,
        read=float(OLLAMA_TIMEOUT_SECONDS),
        write=10.0,
        pool=10.0,
    )
    _client = httpx.AsyncClient(timeout=timeout)


async def close_ollama_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("Ollama client not initialized. Call init_ollama_client() on startup.")
    return _client


def _payload(prompt: str, stream: bool) -> dict[str, Any]:
    return {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
        },
    }


async def generate(prompt: str) -> str:
    """
    Non-streaming generate call.
    Retries transient request failures.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            resp = await client.post(url, json=_payload(prompt, stream=False))
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
            data = resp.json()
            return data.get("response", "")
        except (httpx.RequestError, httpx.TimeoutException, RuntimeError) as exc:
            last_exc = exc
            if attempt == OLLAMA_MAX_RETRIES:
                raise
            await asyncio.sleep(1.5 * attempt)

    raise RuntimeError(f"Ollama generate failed: {last_exc}")


async def generate_stream(prompt: str) -> AsyncIterator[str]:
    """
    Streaming generate call.
    Yields incremental text chunks.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    client = _get_client()

    async with client.stream("POST", url, json=_payload(prompt, stream=True)) as resp:
        if resp.status_code != 200:
            text = await resp.aread()
            raise RuntimeError(f"Ollama error {resp.status_code}: {text.decode('utf-8', errors='ignore')}")

        async for line in resp.aiter_lines():
            if not line:
                continue
            # Ollama streams JSON lines like {"response":"...","done":false}
            try:
                obj = httpx.Response(200, content=line).json()
            except Exception:
                continue

            chunk = obj.get("response")
            if chunk:
                yield chunk

            if obj.get("done") is True:
                break
