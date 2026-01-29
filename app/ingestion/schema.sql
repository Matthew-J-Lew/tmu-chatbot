-- =============================================================
-- Arts AI Chatbot — RAG Database Schema (PostgreSQL + pgvector)
-- File: app/ingestion/schema.sql
-- -------------------------------------------------------------
-- WHAT THIS DB DOES (high level)
-- 1) Stores SOURCES (web pages, PDFs, etc.) you ingest.
-- 2) Splits each source into CHUNKS (small passages) for retrieval.
-- 3) Creates TEXT (tsvector) and VECTOR (embedding) indexes so a
--    user question can find the most relevant chunks quickly.
-- 4) Exposes a helper function for HYBRID search (keyword + vector).
-- =============================================================


-- -------------------------
-- Required extensions
-- -------------------------
-- 'vector' provides the pgvector type and ANN indexes for embeddings
-- 'unaccent' helps normalize text before building tsvector for better keyword search.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;


-- =============================================================
-- TABLE: sources
-- -------------------------------------------------------------
-- One row per original file/page you ingested (HTML page, PDF, etc.)
-- We keep minimal fetch metadata here; text content itself will be
-- chunked and stored in the 'chunks' table (better for retrieval).
-- -------------------------------------------------------------
-- id            = basic id
-- url           = where the source came from, could be a webpage, file, or filepath
-- content_type  = html, pdf, docx, etc.
-- title         = short title for the source, e.g. "Undergraduate programs - faculty of arts"

-- meta          = Optional metadata (headers, crawl info, custom tags) 

-- http_status   = e.g. 200, 404
-- etag          = useful for change detection
-- content_sha1  = strong dedude if we compute a SHA1 of the raw file
-- content_bytes = size of the raw file in bytes

-- fetched_at    = when we first fetched the file
-- created_at    = when the file was first created
-- updated_at    = when the file was last updated  
-- =============================================================
CREATE TABLE IF NOT EXISTS sources (

  id            BIGSERIAL PRIMARY KEY,
  url           TEXT        NOT NULL,
  content_type  TEXT        NOT NULL,
  title         TEXT        NOT NULL,

  meta          JSONB,

  http_status   INTEGER,
  etag          TEXT,
  content_sha1  CHAR(40), 
  content_bytes INTEGER,   

  fetched_at    TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()

);


-- -------------------------
-- Indexes for the Sources table
-- -------------------------
-- Used for common lookups and prevent duplicates of the same original file/page
-- 1. We look up by URL fairly often
-- 2. If you compute content_sha1, this prevents storing the same file twice
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources (url);
CREATE UNIQUE INDEX IF NOT EXISTS ux_sources_content_sha1 ON sources (content_sha1);


-- =============================================================
-- TABLE: chunks
-- -------------------------------------------------------------
-- Each source is split into multiple "chunks" (small passages).
-- Retrieval engines work best on smaller units (~300–400 tokens)
-- with a little overlap. We store embeddings and text features here.
-- -------------------------------------------------------------
-- id             = basic ID
-- source_id      = Link back to the original source document/page, references the source table id
-- section        = the specific section inside a given source (like H2, H3 html tags)
-- url            = the canonical URL that this chunk belongs to (for citations)
-- chunk          = the chunk text used for retrieval
-- chunk_tokens   = optional: approximate token count (useful for stats and budgeting)
-- embedding      = embedding vector for the semantic search (dimension must match embedding model)
-- ts             = full text search vector (keywrod search), auto built using trigger defined below
-- chunk_md5      = a stable ID for "this exact chunk from this URL"
-- created_at     = when this chunk was created in the db
-- updated_at     = when this chunk was last updated in the db
-- =============================================================
CREATE TABLE IF NOT EXISTS chunks (

  id            BIGSERIAL PRIMARY KEY,
  source_id     BIGINT      NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  section       TEXT,
  url           TEXT        NOT NULL,
  chunk         TEXT        NOT NULL,
  chunk_tokens  INTEGER,
  embedding     VECTOR(384) NOT NULL,
  ts            TSVECTOR    NOT NULL,

  chunk_md5     CHAR(32)    NOT NULL,

  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()

);


-- -------------------------
-- Indexes for the chunks table
-- -------------------------
-- 1. Fast join filter between chunks and sources
-- 2. Prevent duplicate chunks from being stored repeatedly on re-ingest
-- 3. Full-text search index for keyword queries (BM25-ish ranking with ts_rank)
CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks (source_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_url_md5 ON chunks (url, chunk_md5);
CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks USING GIN (ts);

-- Approximate nearest neighbor index over embeddings (cosine similarity).
-- 'lists' controls speed/recall tradeoff. Start with 100; tune later for larger corpora.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
  ON chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Helpful for citations lookups by URL
CREATE INDEX IF NOT EXISTS idx_chunks_url ON chunks (url);


-- =============================================================
-- TRIGGER: keep 'ts' (tsvector) and 'updated_at' fresh for chunks
-- -------------------------------------------------------------
-- This ensures that whenever we insert/update a chunk, we build/refresh
-- the full-text vector using 'unaccent' for better matching.
-- =============================================================
CREATE OR REPLACE FUNCTION chunks_ensure_ts_fn()
RETURNS TRIGGER AS $$
BEGIN
  -- Build a keyword-search vector from section + chunk text (unaccented)
  NEW.ts := to_tsvector(
              'english',
              unaccent(COALESCE(NEW.section,'') || ' ' || NEW.chunk)
            );
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_ensure_ts ON chunks;
CREATE TRIGGER trg_chunks_ensure_ts
BEFORE INSERT OR UPDATE OF section, chunk
ON chunks
FOR EACH ROW
EXECUTE FUNCTION chunks_ensure_ts_fn();

-- Backfill ts for any existing rows (if you ever load data before creating the trigger)
UPDATE chunks
SET ts = to_tsvector('english', unaccent(COALESCE(section,'') || ' ' || chunk))
WHERE ts IS NULL;


-- =============================================================
-- VIEW: quick peek at chunks without dumping full text
-- -------------------------------------------------------------
CREATE OR REPLACE VIEW v_chunks_basic AS
SELECT
  c.id,
  c.source_id,
  s.url AS source_url,
  c.url  AS chunk_url,
  left(c.chunk, 600) AS chunk_preview,
  c.created_at
FROM chunks c
JOIN sources s ON s.id = c.source_id;


-- =============================================================
-- HYBRID SEARCH HELPER (optional)
-- -------------------------------------------------------------
-- Many apps do the hybrid scoring in application code, but this SQL helper
-- lets you prototype quickly. It blends:
--   - semantic similarity (vector cosine via pgvector)
--   - keyword relevance (ts_rank on tsvector)
-- weights are tunable (default 0.7 semantic, 0.3 keyword).
-- Return top k results with their individual and combined scores.
-- =============================================================
CREATE OR REPLACE FUNCTION rag_hybrid_search(
  query_text TEXT,
  query_embedding VECTOR,
  k INT DEFAULT 10,
  weight_vector DOUBLE PRECISION DEFAULT 0.7,
  weight_text   DOUBLE PRECISION DEFAULT 0.3
)
RETURNS TABLE(
  id BIGINT,
  chunk_url TEXT,
  source_url TEXT,
  section TEXT,
  chunk TEXT,
  vector_score DOUBLE PRECISION,
  text_score DOUBLE PRECISION,
  hybrid_score DOUBLE PRECISION
)
LANGUAGE sql
AS $$
  SELECT
    c.id,
    c.url  AS chunk_url,
    s.url  AS source_url,
    c.section,
    c.chunk,
    (1 - (c.embedding <=> query_embedding))                         AS vector_score, -- cosine similarity via 1 - cosine_distance
    ts_rank(c.ts, plainto_tsquery('english', query_text))            AS text_score,
    (weight_vector * (1 - (c.embedding <=> query_embedding))
    + weight_text * ts_rank(c.ts, plainto_tsquery('english', query_text))) AS hybrid_score
  FROM chunks c
  JOIN sources s ON s.id = c.source_id
  ORDER BY hybrid_score DESC
  LIMIT k
$$;


-- =============================================================
-- NOTES / HOW THIS FITS THE RAG PIPELINE
-- -------------------------------------------------------------
-- INGESTION PHASE (Python script):
--   1) Fetch a source (web page, PDF, etc.) → insert/UPSERT into 'sources'.
--   2) Extract clean text → split into 300–400 token chunks (with 50–80 overlap).
--   3) Create an embedding vector per chunk (e.g., 384 dims from SBERT/BGE).
--   4) For each chunk:
--        - compute chunk_md5 = md5(url || '#' || chunk)
--        - INSERT into 'chunks' with (source_id, url, section, chunk, tokens, embedding).
--        - The trigger auto-builds 'ts' for keyword search.
--
-- RUNTIME (when a user asks a question):
--   A) Embed the user query with the same embedding model (normalized).
--   B) Fetch top 30 candidates via HYBRID (vector + text). Use this function
--      or equivalent logic in app code.
--   C) Rerank those 30 with a cross-encoder (in Python) → take top 6–8 chunks.
--   D) Build your prompt from those chunks → call the LLM → cite URLs.
--
-- TUNING TIPS:
--   - If you switch to a 768-dim embedding model, change VECTOR(384) → VECTOR(768)
--     and recreate the 'idx_chunks_embedding' index.
--   - Adjust 'lists' in IVFFLAT as corpus grows (larger = faster recall, more memory).
--   - VACUUM ANALYZE periodically; monitor index bloat.
--   - Keep chunks small; lots of small, focused passages beat a few long ones.
-- =============================================================
