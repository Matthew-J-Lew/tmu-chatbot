# TMU Faculty of Arts Chatbot

A small, Docker-first **Retrieval-Augmented Generation (RAG)** stack for answering questions about **Toronto Metropolitan University (TMU) Faculty of Arts** content using:

- **Postgres + pgvector** for storage and hybrid retrieval (keyword + embeddings)
- **A crawler + ingestion pipeline** to discover/ingest official pages (HTML/PDF/CSV/XLSX)
- **A FastAPI service** that retrieves context, optionally reranks, and calls an LLM
- **Redis** for response/retrieval caching

This repo supports **two interchangeable LLM backends**:

- **Azure OpenAI (recommended)** — fast, hosted, production-oriented
- **Ollama (optional)** — local fallback for offline dev / emergency back-pocket use

---

## Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [LLM configuration (Azure OpenAI vs Ollama)](#llm-configuration-azure-openai-vs-ollama)
- [Ingesting data](#ingesting-data)
- [Calling the API](#calling-the-api)
- [Debug / inspection tools](#debug--inspection-tools)
- [Tuning knobs](#tuning-knobs)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## Architecture

High-level flow:

1. **Crawl**: discover in-scope URLs and store them in Postgres (`crawl_targets`)
2. **Ingest**: fetch approved pages/files → extract text → chunk → embed → store in Postgres (`sources`, `chunks`)
3. **Ask a question**: embed query → hybrid search in Postgres → (optional) rerank → build prompt → **call the configured LLM provider** → return answer + citations

```
User -> FastAPI (/api/chat)
          |-> retrieve (pgvector + full-text) -> candidates
          |-> optional rerank (cross-encoder)
          |-> prompt builder (strict "use only context")
          |-> LLM generate (Azure OpenAI OR Ollama)
          `-> JSON: answer + sources + timings (+ caching via Redis)
```

---

## Prerequisites

### Required
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose** (the modern `docker compose` command)

### Recommended
- 8–16GB RAM (more helps ingestion + reranking)
- If you plan to use **Ollama locally**, more RAM/CPU helps (model-dependent)

---

## Quickstart

### 1) Clone the repo
```bash
git clone https://github.com/Matthew-J-Lew/tmu-chatbot.git
cd tmu-chatbot
```

### 2) Create your environment file
Docker Compose automatically reads a `.env` file in the project root for variable substitution.

```bash
cp .env.example .env
```

Then open `.env` and set at least:
- `AZURE_OPENAI_API_KEY=...` (do **not** commit this)
- `LLM_PROVIDER=azure` (recommended)

### 3) Start the core services
This starts Postgres (pgvector), Redis, the API, and Adminer (DB UI).
Ollama also starts by default, but it is only *used* if `LLM_PROVIDER=ollama`.

```bash
docker compose up -d --build
```

Check status:
```bash
docker compose ps
```

### 4) Put some data in the database
Follow [Ingesting data](#ingesting-data). Once chunks exist, you can start asking questions via the API.

---

## LLM configuration (Azure OpenAI vs Ollama)

The API reads these environment variables (see `.env.example`):

### Provider selection
- `LLM_PROVIDER=azure|ollama`
  - `azure` uses Azure OpenAI Chat Completions (recommended)
  - `ollama` uses the local Ollama container
- Optional fallback:
  - `LLM_FALLBACK_PROVIDER=ollama|azure`
  - If the primary provider fails, the API will try the fallback **once**.

### Azure OpenAI (recommended)
Set these in `.env`:

- `AZURE_OPENAI_ENDPOINT` (defaults to the Faculty of Arts resource endpoint)
- `AZURE_OPENAI_API_KEY` (**required**)
- `AZURE_OPENAI_DEPLOYMENT` (deployment name, e.g. `gpt-4o-mini`)
- `AZURE_OPENAI_API_VERSION` (default `2024-10-21`)

Cost-control defaults:
- `AZURE_OPENAI_MAX_TOKENS=512` (hard-capped at 512 in config)
- `AZURE_OPENAI_TEMPERATURE=0.1`

**Swap models:** in Azure, you typically switch **deployments**, so changing
`AZURE_OPENAI_DEPLOYMENT` is the normal way to “swap models” without code changes.

### Ollama (optional / offline fallback)
To use Ollama instead:

1) Set in `.env`:
```env
LLM_PROVIDER=ollama
```

2) Pull the model you want (inside the Ollama container):
```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama list
```

3) If you change the model, restart the API:
```bash
docker compose up -d --build api
```

---

## Ingesting data

### Option A: crawl then ingest from the DB

This is the “production-style” flow:
1) crawl URLs into `crawl_targets`
2) ingest approved targets from the DB

**Run the crawler** (uses `app/crawler/profiles.yaml`):
```bash
docker compose --profile crawl run --rm crawler   python -m app.crawler.crawl --profile arts
```

**Then ingest from DB targets**:
```bash
docker compose --profile ingest run --rm ingestion   python -m app.ingestion.ingest --mode db --profile arts --limit 200
```

Verify:
```bash
docker compose exec pg psql -U rag -d ragdb -c "SELECT status, COUNT(*) FROM crawl_targets GROUP BY status ORDER BY COUNT(*) DESC;"
docker compose exec pg psql -U rag -d ragdb -c "SELECT COUNT(*) FROM chunks;"
```

### Option B: run the scheduler (crawl+ingest on an interval)

This runs the pipeline continuously inside a container.
It’s convenient for a Docker-only deployment; in production you might prefer cron/CI.

Start it:
```bash
docker compose --profile pipeline up -d --build pipeline
```

Stop it:
```bash
docker compose --profile pipeline stop pipeline
```

Scheduler knobs live in `docker-compose.yml` under `pipeline.environment`:
- `PIPELINE_PROFILES`
- `PIPELINE_INTERVAL_SECONDS`
- `CRAWL_RPS`, `CRAWL_ENABLE_SITEMAPS`
- `INGEST_LIMIT`

---

## Calling the API

### Health check
```bash
curl http://localhost:8000/healthz
```

### Ask a question (macOS/Linux)
```bash
curl -s http://localhost:8000/api/chat   -H "Content-Type: application/json"   -d '{"question":"What undergraduate programs are offered at the Faculty of Arts?"}' | jq
```

### Ask a question (Windows PowerShell)
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" ` -ContentType "application/json" ` -Body (@{question="How many undergraduate programs are there in the faculty of arts? Can you list them all?"} | ConvertTo-Json) ` | ConvertTo-Json -Depth 10
```

### What you get back
A JSON response containing:
- `answer` (with citations like `[1]`)
- `sources` (URL list)
- `timings` + `latency_ms`
- `cached` (whether Redis returned a cached response)

---

## Web widget (v1)

The API serves a **dependency-free, embeddable web widget** at:
- `GET /widget/v1/widget.js`

There is also a simple demo page (useful for smoke testing):
- `GET /widget/v1/demo.html`

### Embed via script tag

```html
<div id="tmu-arts-chat"></div>

<script src="https://YOUR_API_HOST/widget/v1/widget.js" defer></script>
<script>
  window.TMUChatbot.init({
    container: '#tmu-arts-chat',
    apiBaseUrl: 'https://YOUR_API_HOST',
    mode: 'public',
    title: 'TMU Arts Chat',
    enableCitations: true,
    initialPrompt: "Hi! Ask me anything about TMU Faculty of Arts. I'll cite official sources when available."
  });
</script>
```

### Admin/debug mode

Admin mode uses `POST /admin/tools/chat` and enables a per-message debug drawer (intent + retrieval details):

```html
<div id="tmu-arts-chat-admin"></div>

<script src="https://YOUR_API_HOST/widget/v1/widget.js" defer></script>
<script>
  window.TMUChatbot.init({
    container: '#tmu-arts-chat-admin',
    apiBaseUrl: 'https://YOUR_API_HOST',
    mode: 'admin',
    title: 'TMU Arts Chat (Admin)',
    enableCitations: true,
    enableDebug: true,
    defaultParams: { top_k: 6, num_candidates: 20 }
  });
</script>
```

---

## Debug / inspection tools

Open a shell in the API container:
```bash
docker compose run --rm api bash
```

Print the **retrieved chunks** (what RAG is feeding into the prompt):
```bash
python -m app.tools.inspect_retrieval "How do I apply to the Faculty of Arts?" 6 20 1200
```

Print the **full exact prompt** sent to the LLM:
```bash
python -m app.tools.inspect_prompt "What undergraduate programs are offered at the Faculty of Arts?" 4 12
```

Inspect recent crawl/ingest runs and URL status counts:
```bash
python -m app.tools.inspect_pipeline_stats
```

---

## Tuning knobs

Most knobs can be adjusted by editing `docker-compose.yml` (recommended for simple deployments).
Anything that changes embeddings/chunking usually requires **re-ingesting**.

### LLM provider selection
In `.env`:
- `LLM_PROVIDER` — `azure` or `ollama`
- `LLM_FALLBACK_PROVIDER` — optional, try this provider once if the primary fails

### Azure OpenAI generation + cost control
In `.env` / `docker-compose.yml`:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT` — **deployment name**
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_MAX_TOKENS` — capped at 512
- `AZURE_OPENAI_TEMPERATURE` — recommended `0.1` for grounded RAG
- `AZURE_OPENAI_TIMEOUT_SECONDS`

### Ollama generation + latency
In `docker-compose.yml` → `api.environment`:
- `OLLAMA_MODEL` — which model to call (must be pulled in Ollama)
- `OLLAMA_NUM_PREDICT` — max tokens to generate
- `OLLAMA_TEMPERATURE`, `OLLAMA_TOP_P` — creativity vs determinism
- `OLLAMA_TIMEOUT_SECONDS`, `OLLAMA_MAX_RETRIES` — reliability under load
- `MAX_CONCURRENT_LLM` — caps concurrent LLM calls to stabilize latency

Also in `docker-compose.yml` → `ollama.environment`:
- `OLLAMA_KEEP_ALIVE` — keeps models loaded longer to reduce cold starts

### Retrieval quality vs speed
In `docker-compose.yml` → `api.environment`:
- `RAG_TOP_K` — how many chunks make it into the final prompt
- `RAG_NUM_CANDIDATES` — how many candidates are pulled from Postgres before rerank
- `RERANK_ENABLED` — `"true"`/`"false"` to enable the cross-encoder reranker
- `RERANK_MODEL` — reranker model name (defaults in `app/rag/reranker.py`)

Rule of thumb:
- Increase `RAG_NUM_CANDIDATES` for better recall (but reranking gets slower)
- Increase `RAG_TOP_K` if answers need more context (prompt gets bigger/slower)

### Prompt/context sizing
In `docker-compose.yml` → `api.environment`:
- `MAX_CHUNK_CHARS` — per-chunk cap in prompt (smaller = faster, less context)
- `MAX_CONTEXT_CHARS` — total prompt context cap (smaller = faster)

If you see answers getting cut off or missing citations, these caps are often the reason.

### CORS (web widget)
In `.env` / `docker-compose.yml`:
- `CORS_ALLOW_ORIGINS` — comma-separated list (e.g. `https://www.torontomu.ca,http://localhost:3000`)

### Caching (Redis)
In `docker-compose.yml` → `api.environment`:
- `CACHE_TTL_RESPONSE` — cache full model responses (seconds)
- `CACHE_TTL_RETRIEVAL` — cache retrieval results (seconds)

Set either TTL to `0` to effectively disable that cache.

### Ingestion chunking + embeddings (requires re-ingest if changed)
Ingestion reads env vars (see `app/ingestion/ingest.py`):
- `CHUNK_TOKEN_SIZE` (default 250)
- `CHUNK_TOKEN_OVERLAP` (default 50)
- `EMBED_MODEL_NAME` (default `sentence-transformers/all-MiniLM-L6-v2`)

**Important:** the embedding model must match runtime query embedding in `app/rag/embeddings.py`
(currently hard-coded to `sentence-transformers/all-MiniLM-L6-v2`, 384 dims).

If you change embedding dimension:
- update `VECTOR(384)` in `app/ingestion/schema.sql`
- re-create the embedding index
- re-ingest everything

### Crawler scope and politeness
Crawler knobs:
- `app/crawler/profiles.yaml` — seeds, allowed domains, deny patterns, max depth/pages
- `CRAWL_RPS` — requests per second per host (politeness)
- `CRAWL_ENABLE_SITEMAPS` — whether to use sitemaps
- `CRAWL_TIMEOUT_SECONDS`, `CRAWL_USER_AGENT`

---

## Troubleshooting

### “Azure OpenAI calls fail”
Common issues:
- **401/403**: `AZURE_OPENAI_API_KEY` missing/incorrect.
- **404**: wrong `AZURE_OPENAI_DEPLOYMENT` (deployment name must exist in Azure).
- **429**: rate limit hit; reduce concurrency (`MAX_CONCURRENT_LLM`) and/or request sizes.

### “The API is running but answers are bad / irrelevant”
- You probably haven’t ingested enough relevant pages yet.
- Try: crawl + ingest more pages (`--limit 500`, increase `max_pages`, etc.)
- Inspect what’s being retrieved:
  ```bash
  docker compose run --rm api python -m app.tools.inspect_retrieval "YOUR QUESTION"
  ```

### “Ollama model not found”
Pull the model:
```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
```
Then restart API:
```bash
docker compose up -d --build api
```

### “Responses are slow”
Common causes:
- Reranker is enabled and CPU-only (set `RERANK_ENABLED=false` to test)
- Prompt is too large (reduce `MAX_CONTEXT_CHARS` / `MAX_CHUNK_CHARS`)
- Too much concurrency (reduce `MAX_CONCURRENT_LLM`)
- Using a large local Ollama model for your machine (try a smaller model)

### “Start fresh” (wipe DB + models)
⚠️ This deletes Postgres data and local Ollama models stored in Docker volumes.
```bash
docker compose down -v
docker compose up -d --build
```

### View the database in a browser (Adminer)
Adminer runs at:
- http://localhost:8080

Use:
- System: `PostgreSQL`
- Server: `pg`
- Username: `rag`
- Password: `rag`
- Database: `ragdb`

---

## Project layout

```
app/
  api/          FastAPI service + Redis cache + asyncpg + (Azure OpenAI / Ollama) client
  crawler/      URL discovery into Postgres (crawl_profiles, crawl_targets)
  ingestion/    Fetch/parse/chunk/embed -> Postgres (sources, chunks)
  pipeline/     Optional scheduler that runs crawl+ingest repeatedly
  rag/          Embeddings, hybrid retrieval, optional reranker
  tools/        Small CLI utilities for debugging retrieval/prompt/pipeline stats
docker-compose.yml
.env.example
```

---

## Notes
- In case the commands/instructions in this file don't work, refer to README_OLD.md for commands
- For some of the tuning knobs, there may be multiple instances of value assignment for them (one in docker-compose.yml, and another maybe in some python file). If updating the docker compose does not immediately change a tuning knob, search the repo for any other local instances.

## Major features todo:
- Make the frontend widget to be placed on TMU webpages.
- Make goldset faq, undergrad programs, graduate programs, departments, staff contact information tables and then the infrastructure for them
- Adding an analytics table + dashboard, tracking question, intent, sources retrieved, answer given, confidence score, satisfaction: shows us what we're getting and what we need

## Minor features todo:
- Change app/crawler/profiles.yaml to ingest more TMU webpages, not just the arts pages.
- Experiment with different prompt sizes, number of chunks, and chunk sizes, etc. to find out what the best balance is for each tuning knob in our specific case/dataset.
- Add an answer formatting layer, tables for lists, numbered steps for procedures, bullet points for requirements, short summaries first, details after, sources at the end

## proposed todo:
- Detect when a question asks for specific lists or information and either increase retrieved context, or fallback to tables.
- Work on “smart ingestion”: chunk information based on headings / lists instead of raw text count.
- Graceful escalation "I may not have complete information, you may want to contact"

## Current task:
Chat Widget:
- Build it with a debug drawer that can be toggled on and off
- A basic <ChatWidget apiBaseUrl="" mode="public|admin"/>
- Takes: apiBaseUrl, title, initial prompt, enable citations, enable debug, default params
- Adapter layer: sendMessage(query, sessionId, options) calls: /chat (public), /admin/tools/chat (debug mode)
- MVP: Message list (user + assistant), loading indicator/streaming, citations (links), "copy answer", "reset chat" button
- Admin: debug drawer on each assistant message showing: intent + confidence, retrieval top-k sources + scores, reranker on/off, latency + model used
- be able to just stick it in a div, no global css, 
- Backend: fastapi (what we already have), frontend: vanilla JS, dependency free web component with shadow DOM, a small init() api, and versioned script URLs

Analytics Dashboard:
1. Quality + trust section (is it answering well?):
  - % of answers that include verified citations
  - % low-confidence rate (how often retrieval is weak)
  - zero-hit queries (number of queries with no chunks retrieved)
  - downvoted queries (could be for any reason, wrong/outdated/missing)
  - hallucination risk flags (answers without citations, answers when confidence is below threshold, answer contradicts retrieved text)
2. Content gaps (what do we need to add?):
  - top unanswered/low confidence questions (last 7-30 days)
  - queries that should become FAQ goldset
  - queries that should become structured table entries (program lists, contacts)
  - most requested departments/programs
3. Ops + performance (does it survive traffic?)
  - latency avg
  - cache hit rate
  - error rate/timeouts
  - token usage / cost estimates
4. Retrieval health (does RAG work?)
  - top sources used
  - bad sources, sources that coorelate with downvotes
  - duplicate retrieval rate (how often top-k hcunks are basically the same page)
5. Drilldown Table (performance at a glance):
  - query, intent, retrieved chunks, final answer, confidence score breakdown, feedback 
6. Playground:
  a. Retrieval inspector: 
  Inputs:
    - query textbox
    - controls (top_k, candidates, max_chars, use_reranker, use_query_rewrite, use_decomposition)
    - filters (only tmu.ca, only faculty of arts sites)
  Outputs:
    - rank, title+url, similarity score, rerank score, chunk text preview, page metadata
  Implement using an admin endpoint that wraps the same retrieval pipeline our cli uses but returns a json POST /admin/tools/inspect_retrieval
  b. Chat bot
    - Real conversation UI/widget shows: final answer, citations, confidence score, latency, model used
    - Debug drawer: detected intent + confidence, rewritten/decomposed queries, retrieved chunks list + scores, prompt metadata, cache hit/miss
  Implement using current endpoint but add a debug flag
