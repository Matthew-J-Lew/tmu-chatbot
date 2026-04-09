# TMU Faculty of Arts Chatbot

A Docker-first **Retrieval-Augmented Generation (RAG)** chatbot for answering questions about **Toronto Metropolitan University (TMU) Faculty of Arts** content using official TMU sources.

The stack includes:
- **Postgres + pgvector** for chunk storage and hybrid retrieval
- **Crawler + ingestion pipeline** for discovering and ingesting TMU pages
- **FastAPI** for retrieval, prompt construction, answer generation, and API endpoints
- **Redis** for caching
- **Azure OpenAI** (recommended) or **Ollama** (optional local fallback)
- **Embeddable web widget** in self-contained vanilla JavaScript

---

## Contents
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Most common commands](#most-common-commands)
- [LLM configuration](#llm-configuration)
- [Ingesting data](#ingesting-data)
- [Calling the API](#calling-the-api)
- [Web widget](#web-widget)
- [Debug / inspection tools](#debug--inspection-tools)
- [Tuning knobs](#tuning-knobs)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Handoff notes](#handoff-notes)

---

## Architecture

High-level flow:

1. **Crawl** approved TMU pages into Postgres (`crawl_targets`)
2. **Ingest** approved pages into `sources` and `chunks`
3. **Retrieve** the most relevant chunks using hybrid search (keyword + embeddings)
4. **Optionally rerank** the retrieved chunks
5. **Build a grounded prompt** using only retrieved context
6. **Generate an answer** with Azure OpenAI or Ollama
7. **Return JSON** with answer, sources, timings, and cache info

```text
User -> FastAPI (/api/chat)
          |-> turn prep / lightweight session context
          |-> retrieval (pgvector + full-text)
          |-> optional rerank
          |-> grounded prompt builder
          |-> Azure OpenAI OR Ollama
          `-> JSON answer + citations + timings
```

---

## Prerequisites

### Required
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose plugin** (`docker compose`)

### Recommended
- 8–16 GB RAM
- More RAM/CPU if using **Ollama** locally

---

## Quickstart

### 1) Clone the repo
```bash
git clone https://github.com/Matthew-J-Lew/tmu-chatbot.git
cd tmu-chatbot
```

### 2) Create your environment file
```bash
cp .env.example .env
```

Open `.env` and set at least:
- `AZURE_OPENAI_API_KEY=...`
- `LLM_PROVIDER=azure`
- optionally `LLM_FALLBACK_PROVIDER=ollama`

> For most deployments, **`.env` is the main place to change system behavior**. `docker-compose.yml` mostly passes those variables through to the containers.

### 3) Start the core services
```bash
docker compose up -d --build pg redis ollama api adminer
```

Check status:
```bash
docker compose ps
```

### 4) Verify the API is healthy
```bash
curl http://localhost:8000/healthz
```

### 5) Crawl and ingest data
```bash
docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile arts
docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile arts_calendar
docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile tmu_core

docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile arts --limit 500
docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile arts_calendar --limit 500
docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile tmu_core --limit 500
```

### 6) Ask a test question
macOS/Linux:
```bash
curl -s http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What undergraduate programs are offered at the Faculty of Arts?"}'
```

Windows PowerShell:
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" `
  -ContentType "application/json" `
  -Body (@{question="What undergraduate programs are offered at the Faculty of Arts?"} | ConvertTo-Json) |
  ConvertTo-Json -Depth 10
```

---

## Most common commands

### Start the main stack
```bash
docker compose up -d --build pg redis ollama api adminer
```

### Follow API logs
```bash
docker compose logs -f api
```

### Fresh rebuild / recrawl / reingest
Use this when the environment is in a bad state or after major ingestion/schema work.

```bash
docker compose down -v --remove-orphans

docker builder prune -af
docker image prune -af

docker compose build --no-cache
docker compose up -d pg redis ollama api adminer

docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile arts
docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile arts_calendar
docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile tmu_core

docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile arts --limit 500
docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile arts_calendar --limit 500
docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile tmu_core --limit 500
```

### Reset Redis cache only
```bash
docker compose exec redis redis-cli FLUSHALL
```

### Check chunk count
```bash
docker compose exec pg psql -U rag -d ragdb -c "SELECT COUNT(*) FROM chunks;"
```

### View the database in Adminer
- URL: `http://localhost:8080`
- System: `PostgreSQL`
- Server: `pg`
- Username: `rag`
- Password: `rag`
- Database: `ragdb`

---

## LLM configuration

This repo supports two interchangeable LLM backends.

### Azure OpenAI (recommended)
Set in `.env`:
- `LLM_PROVIDER=azure`
- `AZURE_OPENAI_API_KEY=...`
- `AZURE_OPENAI_ENDPOINT=...`
- `AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini`
- `AZURE_OPENAI_API_VERSION=2024-10-21`

Common generation knobs:
- `AZURE_OPENAI_MAX_TOKENS`
- `AZURE_OPENAI_TEMPERATURE`
- `AZURE_OPENAI_TIMEOUT_SECONDS`

### Ollama (optional local dev / fallback)
Set in `.env`:
```env
LLM_PROVIDER=ollama
```

Pull a model:
```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama list
```

If you change the model in `.env`, restart the API:
```bash
docker compose up -d --build api
```

### Optional provider fallback
If the primary provider fails, the API can try one fallback once:
```env
LLM_PROVIDER=azure
LLM_FALLBACK_PROVIDER=ollama
```

---

## Ingesting data

### Recommended flow: crawl to DB, then ingest from DB

Run the crawler:
```bash
docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile arts
docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile arts_calendar
docker compose --profile crawl run --rm crawler python -m app.crawler.crawl --profile tmu_core
```

Then ingest approved targets:
```bash
docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile arts --limit 500
docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile arts_calendar --limit 500
docker compose --profile ingest run --rm ingestion python -m app.ingestion.ingest --mode db --profile tmu_core --limit 500
```

Verify crawl status counts:
```bash
docker compose exec pg psql -U rag -d ragdb -c "SELECT p.name, t.status, COUNT(*) FROM crawl_targets t JOIN crawl_profiles p ON p.id = t.profile_id GROUP BY p.name, t.status ORDER BY p.name, COUNT(*) DESC;"
```

Verify chunks exist:
```bash
docker compose exec pg psql -U rag -d ragdb -c "SELECT COUNT(*) FROM chunks;"
```

### Optional: pipeline scheduler
The optional pipeline container can run crawl + ingest on an interval.

Start it:
```bash
docker compose --profile pipeline up -d --build pipeline
```

Stop it:
```bash
docker compose --profile pipeline stop pipeline
```

Scheduler knobs are usually set in `.env`:
- `PIPELINE_PROFILES`
- `PIPELINE_INTERVAL_SECONDS`
- `CRAWL_RPS`
- `CRAWL_ENABLE_SITEMAPS`
- `INGEST_LIMIT`

---

## Calling the API

### Health check
```bash
curl http://localhost:8000/healthz
```

### Chat endpoint
```bash
curl -s http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I apply to the Faculty of Arts?"}'
```

### Streaming endpoint
```bash
curl -N http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I apply to the Faculty of Arts?"}'
```

### Session reset endpoint
```bash
curl -X POST http://localhost:8000/api/session/reset \
  -H "Content-Type: application/json" \
  -d '{"session_id":"YOUR_SESSION_ID"}'
```

### What the API returns
Typical responses include:
- `answer`
- `sources`
- `timings`
- `latency_ms`
- `cached`

---

## Web widget

The API serves a dependency-free widget from:
- `/widget/v1/widget.js`
- `/widget/v1/demo.html`
- `/widget/v1/demo-public.html`

### Public embed example
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

### Admin/debug mode example
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

Admin mode uses `POST /admin/tools/chat` and exposes debug-oriented retrieval details.

---

## Debug / inspection tools

Open a shell in the API container:
```bash
docker compose run --rm api bash
```

Inspect retrieved chunks:
```bash
python -m app.tools.inspect_retrieval "How do I apply to the Faculty of Arts?" 6 20 1200
```

Inspect the full exact prompt:
```bash
python -m app.tools.inspect_prompt "What undergraduate programs are offered at the Faculty of Arts?" 4 12
```

Inspect crawl/ingest stats:
```bash
python -m app.tools.inspect_pipeline_stats
```

When debugging bad answers, check in this order:
1. retrieval
2. prompt
3. LLM output

---

## Tuning knobs

### Important note
For this repo, **most important tuning knobs now live in `.env` / `.env.example`**. `docker-compose.yml` mainly passes them through into the containers. If a config change does not seem to apply, check the Python defaults in `app/api/config.py` or ingestion/runtime code.

### Highest-impact runtime knobs
In `.env`:
- `RAG_NUM_CANDIDATES`
- `RAG_TOP_K`
- `RERANK_ENABLED`
- `RERANK_MODEL`
- `HYBRID_WEIGHT_VECTOR`
- `HYBRID_WEIGHT_TEXT`
- `MAX_CHUNK_CHARS`
- `MAX_CONTEXT_CHARS`
- `LLM_PROVIDER`
- `LLM_FALLBACK_PROVIDER`
- `AZURE_OPENAI_MAX_TOKENS`
- `AZURE_OPENAI_TEMPERATURE`
- `OLLAMA_MODEL`
- `OLLAMA_NUM_PREDICT`
- `CACHE_TTL_RESPONSE`
- `CACHE_TTL_RETRIEVAL`
- `MAX_CONCURRENT_LLM`

### Ingestion / indexing knobs
Changing these usually requires **re-ingesting** the data:
- `EMBED_MODEL_NAME`
- `CHUNK_TOKEN_SIZE`
- `CHUNK_TOKEN_OVERLAP`
- `INGEST_USE_PLAYWRIGHT`
- `INGEST_PLAYWRIGHT_ALWAYS`
- `INGEST_PLAYWRIGHT_FALLBACK`
- `INGEST_MIN_EXTRACTED_CHARS`
- `PLAYWRIGHT_EXPAND_ACCORDIONS`

### Crawl / ops knobs
Usually set in `.env`:
- `CRAWL_PROFILE`
- `CRAWL_RPS`
- `CRAWL_ENABLE_SITEMAPS`
- `CRAWL_CALENDAR_ALLOWED_YEARS`
- `INGEST_LIMIT`
- `PIPELINE_PROFILES`
- `PIPELINE_INTERVAL_SECONDS`

### Rule of thumb
- **Bad retrieval** -> check chunking, query handling, candidate count, reranking, and crawl scope
- **Missing context** -> increase `RAG_TOP_K`, `RAG_NUM_CANDIDATES`, or context caps carefully
- **Slow responses** -> reduce reranking, context size, or concurrency
- **Chunking / embedding changes** -> re-ingest

---

## Troubleshooting

### Azure OpenAI fails
Common causes:
- `401/403` -> bad or missing `AZURE_OPENAI_API_KEY`
- `404` -> wrong `AZURE_OPENAI_DEPLOYMENT`
- `429` -> rate limits; reduce load or request size

### Answers are weak or irrelevant
Common causes:
- not enough relevant data ingested
- chunking quality is poor
- retrieval is returning noisy pages
- context caps are too small

Start by inspecting retrieval:
```bash
docker compose run --rm api python -m app.tools.inspect_retrieval "YOUR QUESTION"
```

### Ollama model not found
```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose up -d --build api
```

### Responses are slow
Common causes:
- reranker on CPU
- prompt is too large
- too much concurrency
- local Ollama model is too large for the machine

### Start fresh
Warning: this deletes Postgres data and local Ollama models stored in Docker volumes.
```bash
docker compose down -v --remove-orphans
docker compose up -d --build
```

---

## Project layout

```text
app/
  api/          FastAPI service, session/context handling, answer generation, provider routing
  crawler/      URL discovery into Postgres
  frontend/     Embeddable chat widget
  ingestion/    Fetch / parse / chunk / embed into Postgres
  pipeline/     Optional scheduler for repeated crawl + ingest
  rag/          Embeddings, hybrid retrieval, reranker
  tools/        CLI inspection utilities
  tests/        Retrieval, turn prep, crawl, and ranking tests

docker-compose.yml
.env.example
README.md
```

---



