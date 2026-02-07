from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncAzureOpenAI

from app.api.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MAX_TOKENS,
    AZURE_OPENAI_TEMPERATURE,
    AZURE_OPENAI_TIMEOUT_SECONDS,
)

# Messages follow the Chat Completions schema.
ChatMessage = Dict[str, str]

_client: Optional[AsyncAzureOpenAI] = None


async def init_azure_openai_client() -> None:
    """Initialize a shared Azure OpenAI client."""
    global _client
    if _client is not None:
        return

    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. Set it in your .env file before using LLM_PROVIDER=azure."
        )

    # The OpenAI Python SDK uses HTTPX under the hood and supports async usage.
    _client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        timeout=AZURE_OPENAI_TIMEOUT_SECONDS,
    )


async def close_azure_openai_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def _get_client() -> AsyncAzureOpenAI:
    if _client is None:
        raise RuntimeError("Azure OpenAI client not initialized. Call init_azure_openai_client() on startup.")
    return _client


async def generate(messages: List[ChatMessage]) -> str:
    """Non-streaming chat completion."""
    client = _get_client()

    resp = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,  # Azure uses deployment name here
        messages=messages,
        max_tokens=AZURE_OPENAI_MAX_TOKENS,
        temperature=AZURE_OPENAI_TEMPERATURE,
        stream=False,
    )

    try:
        return (resp.choices[0].message.content or "")
    except Exception:
        return ""


async def generate_stream(messages: List[ChatMessage]) -> AsyncIterator[str]:
    """Streaming chat completion. Yields incremental token deltas."""
    client = _get_client()

    stream = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        max_tokens=AZURE_OPENAI_MAX_TOKENS,
        temperature=AZURE_OPENAI_TEMPERATURE,
        stream=True,
    )

    async for chunk in stream:
        # OpenAI streaming chunks provide deltas
        try:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
        except Exception:
            content = None

        if content:
            yield content
