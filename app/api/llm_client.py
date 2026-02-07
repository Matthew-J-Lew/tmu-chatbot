from __future__ import annotations

from typing import AsyncIterator, Dict, List, Optional

from app.api.config import LLM_FALLBACK_PROVIDER, LLM_PROVIDER

from app.api.ollama_client import (
    close_ollama_client,
    generate as ollama_generate,
    generate_stream as ollama_generate_stream,
    init_ollama_client,
)

from app.api.azure_openai_client import (
    close_azure_openai_client,
    generate as azure_generate,
    generate_stream as azure_generate_stream,
    init_azure_openai_client,
)

# Messages follow the Chat Completions schema.
ChatMessage = Dict[str, str]


def _normalize_provider(p: Optional[str]) -> str:
    p = (p or "").strip().lower()
    return "azure" if p == "azure" else "ollama"


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    """
    Convert Chat Completions messages into a single prompt string.

    This keeps Ollama compatibility while allowing Azure OpenAI to use native
    system/user separation.
    """
    parts: List[str] = []

    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue

        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"USER:\n{content}")
        elif role == "assistant":
            parts.append(f"ASSISTANT:\n{content}")
        else:
            parts.append(f"{role.upper()}:\n{content}")

    return "\n\n".join(parts).strip()


async def init_llm_client() -> None:
    """Initialize the selected LLM provider."""
    provider = _normalize_provider(LLM_PROVIDER)
    if provider == "azure":
        await init_azure_openai_client()
    else:
        await init_ollama_client()


async def close_llm_client() -> None:
    """Close any initialized clients (safe to call regardless of provider)."""
    await close_azure_openai_client()
    await close_ollama_client()


async def generate(messages: List[ChatMessage]) -> str:
    """Generate a full response (non-streaming)."""
    provider = _normalize_provider(LLM_PROVIDER)
    fallback = _normalize_provider(LLM_FALLBACK_PROVIDER) if LLM_FALLBACK_PROVIDER else None

    try:
        if provider == "azure":
            return await azure_generate(messages)

        prompt = _messages_to_prompt(messages)
        return await ollama_generate(prompt)

    except Exception:
        if not fallback or fallback == provider:
            raise

        # Try fallback once
        if fallback == "azure":
            await init_azure_openai_client()
            return await azure_generate(messages)

        await init_ollama_client()
        prompt = _messages_to_prompt(messages)
        return await ollama_generate(prompt)


async def generate_stream(messages: List[ChatMessage]) -> AsyncIterator[str]:
    """Generate a response as a stream of incremental text chunks."""
    provider = _normalize_provider(LLM_PROVIDER)
    fallback = _normalize_provider(LLM_FALLBACK_PROVIDER) if LLM_FALLBACK_PROVIDER else None

    async def _stream_from(p: str) -> AsyncIterator[str]:
        if p == "azure":
            async for chunk in azure_generate_stream(messages):
                yield chunk
        else:
            prompt = _messages_to_prompt(messages)
            async for chunk in ollama_generate_stream(prompt):
                yield chunk

    yielded = False
    try:
        async for c in _stream_from(provider):
            yielded = True
            yield c
        return

    except Exception:
        # Only attempt fallback if the primary provider failed before yielding anything.
        if yielded or not fallback or fallback == provider:
            raise

        async for c in _stream_from(fallback):
            yield c
