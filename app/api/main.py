"""
Production API entrypoint for the TMU Faculty of Arts chatbot.

High-level flow (non-technical explanation):
1) A user asks a question.
2) We look in a specialized database (RAG DB) to find the best official TMU text snippets.
3) We build a strict prompt that tells the AI: "use only these snippets and cite them."
4) We call the language model (Azure OpenAI or Ollama) to generate an answer.
5) We return JSON: answer + sources + performance timings.

Production upgrades included:
- Async Postgres pooling (fast DB queries)
- Redis caching (instant answers for repeat questions)
- Shared HTTP client for Ollama (connection reuse)
- Concurrency limiting (prevents overload)
- Prompt + output limits (reduces tail latency)
- Optional streaming endpoint (/api/chat/stream)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import re
from time import perf_counter
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pathlib import Path

from app.api.config import (
    RAG_TOP_K,
    RAG_NUM_CANDIDATES,
    MAX_CHUNK_CHARS,
    MAX_CONTEXT_CHARS,
    CACHE_TTL_RESPONSE,
    CACHE_TTL_RETRIEVAL,
    MAX_CONCURRENT_LLM,
    CORS_ALLOW_ORIGINS,
    RERANK_ENABLED,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    AZURE_OPENAI_DEPLOYMENT,
)
from app.api.db import init_db_pool, close_db_pool, get_pool
from app.api.cache import init_redis, close_redis, make_cache_key, cache_get_json, cache_set_json
from app.api.llm_client import init_llm_client, close_llm_client, generate, generate_stream

from app.rag.retrieval import retrieve


app = FastAPI(title="TMU Faculty of Arts Chatbot API", version="3.0-prod")

# CORS is required for the web widget deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    # NOTE: allow GET so the widget JS can be loaded cross-origin if needed.
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _find_widget_dir() -> Path:
    """Locate app/frontend/widget in both dev and Docker layouts.

    This repo runs FastAPI in Docker by copying app/api/main.py to /app/main.py.
    In that layout, __file__ == /app/main.py and the widget lives at /app/app/frontend/widget.

    In local/dev layouts, __file__ may be .../app/api/main.py and the widget lives at
    .../app/frontend/widget.
    """
    here = Path(__file__).resolve()

    candidates = [
        # Docker layout: /app/main.py -> /app/app/frontend/widget
        here.parent / "app" / "frontend" / "widget",
        # Dev layout: .../app/api/main.py -> .../app/frontend/widget
        here.parent.parent / "frontend" / "widget",
    ]

    for p in candidates:
        if p.exists() and p.is_dir():
            return p

    # Last resort: relative to CWD
    p = Path.cwd() / "app" / "frontend" / "widget"
    return p


# Serve the web widget assets (versioned).
_WIDGET_DIR = _find_widget_dir()
if _WIDGET_DIR.exists():
    app.mount("/widget", StaticFiles(directory=str(_WIDGET_DIR), html=True), name="widget")

# Concurrency limiter: prevents too many simultaneous LLM calls (stabilizes latency)
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)


# -----------------------------
# Request / Response Models
# -----------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question for the TMU Arts chatbot.")


class SourceItem(BaseModel):
    id: int
    url: str
    title: str
    section: Optional[str] = None


class TimingBreakdown(BaseModel):
    cache_ms: int
    retrieve_ms: int
    prompt_ms: int
    llm_ms: int
    total_ms: int


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    latency_ms: int
    timings: TimingBreakdown
    cached: bool


class ChatParams(BaseModel):
    """Optional tuning knobs for admin/debug calls.

    These are intentionally limited for safety.
    """

    top_k: Optional[int] = Field(None, ge=1, le=20)
    num_candidates: Optional[int] = Field(None, ge=1, le=50)


class AdminChatRequest(BaseModel):
    question: str = Field(..., description="User's question for the TMU Arts chatbot.")
    params: Optional[ChatParams] = None


class IntentInfo(BaseModel):
    label: str
    confidence: float


class RetrievalDebugItem(BaseModel):
    rank: int
    url: str
    section: Optional[str] = None
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    snippet: str


class RetrievalDebug(BaseModel):
    top_k: int
    num_candidates: int
    rerank_enabled: bool
    items: List[RetrievalDebugItem]


class ModelDebug(BaseModel):
    provider: str
    name: str


class AdminDebug(BaseModel):
    intent: IntentInfo
    retrieval: RetrievalDebug
    model: ModelDebug


class AdminChatResponse(ChatResponse):
    debug: AdminDebug


# -----------------------------
# Startup / Shutdown
# -----------------------------

@app.on_event("startup")
async def _startup() -> None:
    # Keep long-lived resources ready for fast performance.
    await init_db_pool()
    await init_redis()
    await init_llm_client()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await close_llm_client()
    await close_redis()
    await close_db_pool()


# -----------------------------
# Helpers
# -----------------------------

def normalize_question(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def sanitize_question(q: str) -> str:
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(q) > 500:
        raise HTTPException(status_code=400, detail="Question is too long (max 500 characters).")
    return q


def redact_pii(text: str) -> str:
    # Basic regex-based PII redaction for safety (not perfect, but helps).
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone = re.compile(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)\d{3}[\s\-]?\d{4}")
    student_id = re.compile(r"\b\d{9}\b")  # simple 9-digit pattern

    text = email.sub("[REDACTED_EMAIL]", text)
    text = phone.sub("[REDACTED_PHONE]", text)
    text = student_id.sub("[REDACTED_ID]", text)
    return text


def detect_intent(question: str) -> IntentInfo:
    """Heuristic intent detection (cheap + deterministic).

    This is intentionally simple for the MVP. It can be replaced later with
    an LLM classifier or hybrid approach.
    """
    q = normalize_question(question)

    # Strong list / directory requests
    if re.search(r"\b(list|show)\b.*\b(all|every)\b", q) or re.search(r"\b(list|catalog|directory)\b", q):
        return IntentInfo(label="LIST_REQUEST", confidence=0.92)

    if re.search(r"\b(programs?|departments?|contacts?|emails?|phone numbers?)\b", q) and re.search(r"\b(all|list|show)\b", q):
        return IntentInfo(label="LIST_REQUEST", confidence=0.85)

    # Procedural / how-to
    if re.search(r"\b(how do i|how to|steps?|procedure|process)\b", q) or re.search(r"\b(apply|register|enroll|drop|withdraw|submit|request)\b", q):
        return IntentInfo(label="PROCEDURAL", confidence=0.80)

    # Factual / quick facts
    if re.search(r"^(what|when|where|who|how many)\b", q):
        return IntentInfo(label="FACT", confidence=0.70)

    return IntentInfo(label="RAG_QA", confidence=0.55)


def model_debug_info() -> ModelDebug:
    provider = (LLM_PROVIDER or "").strip().lower() or "unknown"
    name = "unknown"
    if provider == "ollama":
        name = OLLAMA_MODEL
    elif provider == "azure":
        name = AZURE_OPENAI_DEPLOYMENT
    return ModelDebug(provider=provider, name=name)


def build_messages_and_sources(
    question: str, chunks: List[Dict[str, Any]]
) -> tuple[List[Dict[str, str]], List[SourceItem]]:
    """
    Build a strict prompt with numbered context passages.

    Production principle:
    - Smaller prompt => faster inference.
    So we cap chunk text length and total context length.
    """
    sources: List[SourceItem] = []
    context_lines: List[str] = []

    total_chars = 0
    for i, c in enumerate(chunks, start=1):
        url = c.get("chunk_url") or c.get("source_url") or ""
        section = c.get("section")
        chunk_text = (c.get("chunk") or "").strip()

        # Per-chunk cap
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS].rstrip() + "…"

        block = f"[{i}] URL: {url}\nSection: {section}\nText: {chunk_text}\n"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        total_chars += len(block)
        context_lines.append(block)
        sources.append(SourceItem(id=i, url=url, title=url, section=section))

    system_instructions = (
        "You are a helpful assistant for Toronto Metropolitan University's Faculty of Arts.\n"
        "Answer the user's question using ONLY the context passages provided.\n"
        "If the answer is not in the context, say you do not know.\n"
        "Do not guess or invent details.\n"
        "When you state a fact, cite it using [1], [2], etc.\n"
        "Keep the answer concise and factual.\n"
    )

    user_content = (
        f"USER QUESTION:\n{question}\n\n"
        f"CONTEXT PASSAGES:\n{''.join(context_lines)}\n"
        f"FINAL ANSWER (with citations):\n"
    )

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_content},
    ]

    return messages, sources


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    start_total = perf_counter()

    q = sanitize_question(req.question)
    nq = normalize_question(q)

    # ---- Cache lookup (full response cache) ----
    start_cache = perf_counter()
    resp_key = make_cache_key("resp", nq)
    cached_resp = await cache_get_json(resp_key)
    cache_ms = int((perf_counter() - start_cache) * 1000)

    if cached_resp:
        # Return cached response instantly
        cached_resp["timings"]["cache_ms"] = cache_ms
        cached_resp["timings"]["total_ms"] = int((perf_counter() - start_total) * 1000)
        cached_resp["latency_ms"] = cached_resp["timings"]["total_ms"]
        cached_resp["cached"] = True
        return ChatResponse(**cached_resp)

    # ---- Retrieval (cache retrieval separately) ----
    pool = get_pool()
    start_retrieve = perf_counter()

    ret_key = make_cache_key("ret", nq)
    cached_ret = await cache_get_json(ret_key)

    if cached_ret and "chunks" in cached_ret:
        chunks = cached_ret["chunks"]
    else:
        chunks = await retrieve(
            pool=pool,
            query=q,
            k=RAG_TOP_K,
            num_candidates=RAG_NUM_CANDIDATES,
        )
        await cache_set_json(ret_key, {"chunks": chunks}, CACHE_TTL_RETRIEVAL)

    retrieve_ms = int((perf_counter() - start_retrieve) * 1000)

    # ---- Prompt building ----
    start_prompt = perf_counter()
    messages, sources = build_messages_and_sources(q, chunks)
    prompt_ms = int((perf_counter() - start_prompt) * 1000)

    # ---- LLM call (rate-limited) ----
    start_llm = perf_counter()
    async with _llm_semaphore:
        try:
            raw = await generate(messages)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")

    llm_ms = int((perf_counter() - start_llm) * 1000)

    answer = redact_pii(raw).strip()
    total_ms = int((perf_counter() - start_total) * 1000)

    payload = {
        "answer": answer,
        "sources": [s.dict() for s in sources],
        "latency_ms": total_ms,
        "timings": {
            "cache_ms": cache_ms,
            "retrieve_ms": retrieve_ms,
            "prompt_ms": prompt_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
        },
        "cached": False,
    }

    # Cache the final response
    await cache_set_json(resp_key, payload, CACHE_TTL_RESPONSE)

    return ChatResponse(**payload)


@app.post("/admin/tools/chat", response_model=AdminChatResponse)
async def admin_chat(req: AdminChatRequest) -> AdminChatResponse:
    """Admin/debug chat endpoint.

    This endpoint bypasses the response cache and returns extra debug
    information (intent + retrieval details) to power internal tooling.
    """
    start_total = perf_counter()

    q = sanitize_question(req.question)

    # Params (fallback to env defaults)
    top_k = (req.params.top_k if req.params and req.params.top_k is not None else RAG_TOP_K)
    num_candidates = (
        req.params.num_candidates if req.params and req.params.num_candidates is not None else RAG_NUM_CANDIDATES
    )

    intent = detect_intent(q)

    # ---- Retrieval ----
    pool = get_pool()
    start_retrieve = perf_counter()
    chunks = await retrieve(pool=pool, query=q, k=top_k, num_candidates=num_candidates)
    retrieve_ms = int((perf_counter() - start_retrieve) * 1000)

    # ---- Prompt building ----
    start_prompt = perf_counter()
    messages, sources = build_messages_and_sources(q, chunks)
    prompt_ms = int((perf_counter() - start_prompt) * 1000)

    # ---- LLM call (rate-limited) ----
    start_llm = perf_counter()
    async with _llm_semaphore:
        try:
            raw = await generate(messages)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
    llm_ms = int((perf_counter() - start_llm) * 1000)

    answer = redact_pii(raw).strip()
    total_ms = int((perf_counter() - start_total) * 1000)

    # Debug retrieval items
    items: List[RetrievalDebugItem] = []
    for i, c in enumerate(chunks, start=1):
        url = c.get("chunk_url") or c.get("source_url") or ""
        section = c.get("section")
        chunk_text = (c.get("chunk") or "").strip()
        snippet = chunk_text[:300].replace("\n", " ").strip()
        if len(chunk_text) > 300:
            snippet += "…"
        items.append(
            RetrievalDebugItem(
                rank=i,
                url=str(url),
                section=section,
                vector_score=float(c["vector_score"]) if c.get("vector_score") is not None else None,
                text_score=float(c["text_score"]) if c.get("text_score") is not None else None,
                hybrid_score=float(c["hybrid_score"]) if c.get("hybrid_score") is not None else None,
                rerank_score=float(c["rerank_score"]) if c.get("rerank_score") is not None else None,
                snippet=snippet,
            )
        )

    payload = {
        "answer": answer,
        "sources": [s.dict() for s in sources],
        "latency_ms": total_ms,
        "timings": {
            "cache_ms": 0,
            "retrieve_ms": retrieve_ms,
            "prompt_ms": prompt_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
        },
        "cached": False,
        "debug": {
            "intent": intent.dict(),
            "retrieval": {
                "top_k": top_k,
                "num_candidates": num_candidates,
                "rerank_enabled": bool(RERANK_ENABLED),
                "items": [it.dict() for it in items],
            },
            "model": model_debug_info().dict(),
        },
    }

    return AdminChatResponse(**payload)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming endpoint:
    - begins returning text quickly
    - improves perceived latency (important for production UX)
    - still uses retrieval and prompt rules
    """
    q = sanitize_question(req.question)
    nq = normalize_question(q)

    # If we already have a cached final response, stream it instantly
    resp_key = make_cache_key("resp", nq)
    cached_resp = await cache_get_json(resp_key)
    if cached_resp and "answer" in cached_resp:
        async def _cached_iter():
            # Safety: ensure cached responses are also redacted.
            # (The non-streaming endpoint already redacts, but this keeps
            # the streaming path consistent if the cache is ever populated
            # by another caller.)
            yield redact_pii(str(cached_resp["answer"]))
        return StreamingResponse(_cached_iter(), media_type="text/plain")

    pool = get_pool()
    chunks = await retrieve(pool=pool, query=q, k=RAG_TOP_K, num_candidates=RAG_NUM_CANDIDATES)
    messages, _sources = build_messages_and_sources(q, chunks)

    async def _iter():
        async with _llm_semaphore:
            try:
                async for piece in generate_stream(messages):
                    # Chunk-level redaction; may miss patterns spanning boundaries.
                    yield redact_pii(piece)
            except Exception as e:
                yield f"\n\n[ERROR] LLM request failed: {e}"

    return StreamingResponse(_iter(), media_type="text/plain")
