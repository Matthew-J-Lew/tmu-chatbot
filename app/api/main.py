"""
main.py

This file defines the FastAPI application for the TMU Faculty of Arts chatbot.

This file does the following:
- Exposing HTTP endpoints that other services (or a frontend) can call.
- Connecting a user question to the RAG pipeline:
  1) Validate and clean the user question.
  2) Retrieve the most relevant chunks of information from our TMU database.
  3) Build a clear prompt that contains those chunks plus instructions for the AI model.
  4) Call the local AI model (via Ollama) to generate an answer.
  5) Perform basic redaction of sensitive information (emails, phone numbers, IDs).
  6) Return a JSON response that includes:
       - the answer,
       - the sources used (for citations),
       - and the total latency in milliseconds.
"""

from typing import List, Optional, Dict, Any
import os
import re
from time import perf_counter

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.retrieval import retrieve


# --------------------------------------------------------------------------------------
# Pydantic models (define the input and output shapes of our /api/chat endpoint)
# --------------------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """
    This represents the JSON body that a client sends to /api/chat.

    Example:
    {
        "question": "How do I apply to the Faculty of Arts?"
    }
    """
    question: str = Field(
        ...,
        max_length=2048,
        description="The natural-language question asked by the user.",
    )


class ChatSource(BaseModel):
    """
    One source used in the answer.

    - id:      A small integer used in citations like [1], [2] inside the answer.
    - url:     The page or document URL where the information came from.
    - title:   Optional short title or section name for readability.
    - section: Optional section heading from the original document.
    """
    id: int
    url: str
    title: Optional[str] = None
    section: Optional[str] = None


class ChatResponse(BaseModel):
    """
    This is what the API returns to the caller.

    - answer:      The final answer text, ready to show to the user.
    - sources:     A list of 2–6 sources that back up the answer.
    - latency_ms:  Total time spent from receiving the request to producing
                   the answer (in milliseconds).
    """
    answer: str
    sources: List[ChatSource]
    latency_ms: int


# --------------------------------------------------------------------------------------
# Global configuration (model + host)
# --------------------------------------------------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# System instructions for the AI model.
# These tell the model how to behave and what rules to follow.
SYSTEM_INSTRUCTIONS = """
You are a helpful assistant for the Toronto Metropolitan University (TMU) Faculty of Arts.

Your job is to answer questions using ONLY the information provided in the context below.
If the context does not contain enough information to answer the question, you MUST say
that you do not know based on the available information.

Important rules:
- Do NOT make up policies, dates, or program details.
- Do NOT invent URLs or email addresses.
- When you use a piece of information from the context, refer to it using a citation
  like [1], [2], etc., matching the source numbers provided in the context.
"""


# --------------------------------------------------------------------------------------
# PII redaction helpers
# (Basic, intentionally conservative patterns)
# --------------------------------------------------------------------------------------

# Very simple patterns for obvious personal information.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,2}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{4}\b"
)
# Naive 9-digit "student ID"-like pattern.
STUDENT_ID_RE = re.compile(r"\b\d{9}\b")


def redact_pii(text: str) -> str:
    """
    Replace obvious personal information in the text with placeholders.

    This is NOT a perfect or exhaustive solution. It is a basic safeguard for:
    - Email addresses
    - Phone numbers
    - 9-digit ID-like numbers

    Example:
      "Contact me at alice@example.com" -> "Contact me at [REDACTED_EMAIL]"
    """
    if not text:
        return text

    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = STUDENT_ID_RE.sub("[REDACTED_ID]", text)
    return text


# --------------------------------------------------------------------------------------
# Prompt construction
# (turn retrieved chunks into a single string the model can work with)
# --------------------------------------------------------------------------------------


def build_prompt_and_sources(
    question: str, chunks: List[Dict[str, Any]]
) -> tuple[str, List[ChatSource]]:
    """
    Turn the user's question + retrieved database chunks into:

    1) A single prompt string that we will send to the AI model.
    2) A list of ChatSource objects used to display citations and metadata.

    High-level idea:
    - Each distinct source_url is assigned an ID: [1], [2], [3], ...
    - For each chunk, we include a context block like:
        [1] Undergraduate Programs
        <chunk text here>

    - The model will then answer the question and refer to these blocks using
      [1], [2], etc. in its answer.
    """
    # Map each unique source URL to a numeric citation id.
    source_url_to_id: Dict[str, int] = {}
    id_to_source: Dict[int, ChatSource] = {}
    next_id = 1

    context_blocks: List[str] = []

    for c in chunks:
        source_url = c.get("source_url") or c.get("chunk_url")
        if not source_url:
            # If we don't know where this chunk came from, skip it.
            continue

        # Assign a new citation ID if this source_url hasn't been seen yet.
        if source_url not in source_url_to_id:
            cid = next_id
            source_url_to_id[source_url] = cid
            next_id += 1

            # Use the section as a human-friendly title if available.
            section = c.get("section")
            title = section or source_url

            id_to_source[cid] = ChatSource(
                id=cid,
                url=source_url,
                title=title,
                section=section,
            )
        else:
            cid = source_url_to_id[source_url]

        # Build the text block for this chunk of context.
        section = c.get("section") or ""
        chunk_text = c.get("chunk") or ""
        block = f"[{cid}] {section}\n{chunk_text}"
        context_blocks.append(block)

    # Combine all context blocks into one big "CONTEXT" section.
    context_text = "\n\n".join(context_blocks)

    # The final prompt that goes to the AI model.
    prompt = f"""{SYSTEM_INSTRUCTIONS.strip()}

CONTEXT:
{context_text}

USER QUESTION:
{question}
"""

    # Sort sources by their numeric id so they appear in order [1], [2], ...
    sources = [id_to_source[cid] for cid in sorted(id_to_source.keys())]

    return prompt, sources


# --------------------------------------------------------------------------------------
# Ollama client
# (talks to the local LLM server to get an answer)
# --------------------------------------------------------------------------------------


async def call_ollama(prompt: str) -> str:
    """
    Send the prepared prompt to the local Ollama server and return the model's answer.

    Technical notes:
    - We use Ollama's /api/generate endpoint in non-streaming mode.
    - The model name and host are configured via environment variables.
    - If the Ollama server is down or returns an error, we raise HTTPException(502),
      which surfaces as a "Bad Gateway" error to the client.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # For milestone 3 we keep it simple: wait for the full answer.
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
    except httpx.RequestError as exc:
        # Network or connection error when talking to Ollama.
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}",
        )

    if response.status_code != 200:
        # Ollama responded but with an error.
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned error {response.status_code}: {response.text}",
        )

    data = response.json()
    # In /api/generate, the main text is in the "response" field.
    answer = data.get("response", "")
    if not answer:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty response.",
        )

    return answer


# --------------------------------------------------------------------------------------
# FastAPI application and routes
# --------------------------------------------------------------------------------------

app = FastAPI(
    title="TMU Faculty of Arts Chatbot API",
    description="Backend service for answering questions about TMU's Faculty of Arts.",
    version="0.3.0",  # milestone 3
)


@app.get("/healthz")
async def health_check() -> dict:
    """
    Simple health check endpoint.

    This allows DevOps / monitoring systems (and you) to verify that:
    - The FastAPI server is running.
    - The container is reachable.

    It does NOT check the database or LLM; it's deliberately lightweight.
    """
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for milestone 3.

    High-level flow:
    1) Clean and validate the incoming question.
    2) Use our retrieval pipeline to fetch relevant context from the TMU database.
    3) Build a strict prompt that includes:
         - system instructions,
         - the retrieved context chunks,
         - and the original user question.
    4) Send the prompt to the local LLM (Ollama) and get an answer.
    5) Redact obvious personal information from the answer.
    6) Return the answer, the list of sources, and the total latency.
    """
    start_time = perf_counter()

    # 1) Sanitize input
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # 2) Retrieve top chunks from the RAG database
    #    - k=6: we want up to 6 chunks in the final answer
    #    - num_candidates=30: we let the reranker choose the best among more options
    chunks = retrieve(question, k=6, num_candidates=30)

    # 3) Build prompt string and source metadata (for citations)
    prompt, sources = build_prompt_and_sources(question, chunks)

    # 4) Call the local LLM via Ollama
    raw_answer = await call_ollama(prompt)

    # 5) Basic PII redaction
    cleaned_answer = redact_pii(raw_answer)

    # 6) Measure total latency
    latency_ms = int((perf_counter() - start_time) * 1000)

    return ChatResponse(
        answer=cleaned_answer,
        sources=sources,
        latency_ms=latency_ms,
    )
