# TMU Faculty of Arts Chatbot

A small, Docker-first **Retrieval-Augmented Generation (RAG)** stack for answering questions about **Toronto Metropolitan University (TMU) Faculty of Arts** content using:

- **Postgres + pgvector** for storage and hybrid retrieval (keyword + embeddings)
- **A crawler + ingestion pipeline** to discover/ingest official pages (HTML/PDF/CSV/XLSX)
- **A FastAPI service** that retrieves context, optionally reranks, and calls a local LLM
- **Redis** for response/retrieval caching

---

## Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
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
3. **Ask a question**: embed query → hybrid search in Postgres → (optional) rerank → build prompt → call Ollama → return answer + citations

```
User -> FastAPI (/api/chat)
          |-> retrieve (pgvector + full-text) -> candidates
          |-> optional rerank (cross-encoder)
          |-> prompt builder (strict "use only context")
          |-> LLM generate
          `-> JSON: answer + sources + timings (+ caching via Redis)
```

---

## Prerequisites

### Required
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose** (the modern `docker compose` command)

### Recommended
- 8–16GB RAM (LLMs can be memory-hungry depending on the model)
- More CPU cores helps ingestion + reranking latency

---

## Quickstart

### 1) Clone the repo
```bash
git clone https://github.com/Matthew-J-Lew/tmu-chatbot.git
cd tmu-chatbot
```

### 2) Start the core services
This starts Postgres (pgvector), Redis, Ollama, the API, and Adminer (DB UI):
```bash
docker compose up -d --build
```

Check status:
```bash
docker compose ps
```

### 3) Pull an Ollama model
The default model is set in `docker-compose.yml` under `api.environment.OLLAMA_MODEL`.

Pull it (or any model you want) inside the Ollama container:
```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama list
```

If you change `OLLAMA_MODEL`, restart the API:
```bash
docker compose up -d --build api
```

### 4) Put some data in the database

## Ingesting data

### Option A: crawl then ingest from the DB

This is the “production-style” flow:
1) crawl URLs into `crawl_targets`
2) ingest approved targets from the DB

**Run the crawler** (uses `app/crawler/profiles.yaml`):
```bash
docker compose --profile crawl run --rm crawler \
  python -m app.crawler.crawl --profile arts
```

**Then ingest from DB targets**:
```bash
docker compose --profile ingest run --rm ingestion \
  python -m app.ingestion.ingest --mode db --profile arts --limit 200
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
curl -s http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What undergraduate programs are offered at the Faculty of Arts?"}' | jq
```

### Ask a question (Windows PowerShell)
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/chat" `
  -ContentType "application/json" `
  -Body (@{question="What undergraduate programs are offered at the Faculty of Arts?"} | ConvertTo-Json) | ConvertTo-Json -Depth 10
```

### What you get back
A JSON response containing:
- `answer` (with citations like `[1]`)
- `sources` (URL list)
- `timings` + `latency_ms`
- `cached` (whether Redis returned a cached response)

---

## Debug / inspection tools

These are helpful when tuning retrieval and prompt size.

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

### LLM generation + latency
In `docker-compose.yml` → `api.environment`:
- `OLLAMA_MODEL` — which model to call (must be pulled in Ollama)
- `OLLAMA_NUM_PREDICT` — max tokens to generate (higher = longer answers, slower)
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
- Model is too large for your machine (try a smaller model)
- Reranker is enabled and CPU-only (set `RERANK_ENABLED=false` to test)
- Prompt is too large (reduce `MAX_CONTEXT_CHARS` / `MAX_CHUNK_CHARS`)
- Too much concurrency (reduce `MAX_CONCURRENT_LLM`)

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
  api/          FastAPI service + Redis cache + asyncpg + Ollama client
  crawler/      URL discovery into Postgres (crawl_profiles, crawl_targets)
  ingestion/    Fetch/parse/chunk/embed -> Postgres (sources, chunks)
  pipeline/     Optional scheduler that runs crawl+ingest repeatedly
  rag/          Embeddings, hybrid retrieval, optional reranker
  tools/        Small CLI utilities for debugging retrieval/prompt/pipeline stats
docker-compose.yml
```

---

## Notes
- In case the commands/instructions in this file don't work, refer to README_OLD.md for commands
- For some of the tuning knobs, there may be multiple instances of value assignment for them (one in docker-compose.yml, and another maybe in some python file). If updating the docker compose does not immediately change a tuning knob, search the repo for any other local instances.

## TODO:
- Experiment with different prompt sizes, number of chunks, and chunk sizes, etc. to find out what the best balance is for each tuning knob in our specific case/dataset.
- Because local OLLAMA LLM is relatively slow even with smaller models, we need to implement using a hosted LLM API to improve answer quality and increase speed.
- Change app/crawler/profiles.yaml to ingest more TMU webpages, not just the arts pages.
- Make the frontend widget to be placed on TMU webpages.
- Make a "Goldset/FAQ" document so common questions can easily be answered instead of relying on hard-to-find webpage content
- Due to beautiful soup having trouple parsing JavaScript-heavy pages, we need to look into another crawler or ingestion tool.