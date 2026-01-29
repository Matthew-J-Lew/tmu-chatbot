"""TMU Chatbot ingestion pipeline.

This script fetches documents (HTML/PDF/tabular) and stores them in Postgres as sources + chunks.
For production crawling, it can read approved URLs from the crawl_targets table instead of a YAML allowlist.
"""

import argparse
import hashlib
import io
import os
import re
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import psycopg2
import requests
import tiktoken
import yaml
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer

PGHOST = os.getenv("PGHOST", "pg")
PGUSER = os.getenv("PGUSER", "rag")
PGPASSWORD = os.getenv("PGPASSWORD", "rag")
PGDATABASE = os.getenv("PGDATABASE", "ragdb")

# 384-dim model to match schema VECTOR(384)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking knobs (token-based) for balancing answer quality vs. retrieval precision
CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE", "250"))
CHUNK_TOKEN_OVERLAP = int(os.getenv("CHUNK_TOKEN_OVERLAP", "50"))

USER_AGENT = os.getenv("INGEST_USER_AGENT", "TMU-FOA-RAG-Ingest/0.2 (+https://www.torontomu.ca/arts/)")

# Model + tokenizer are initialized once to amortize startup cost per ingestion run
model = SentenceTransformer(EMBED_MODEL_NAME)
enc = tiktoken.get_encoding("cl100k_base")


# -----------------------
# Utilities
# -----------------------

def approx_token_len(text: str) -> int:
    """Return an approximate token count for budgeting chunk sizes."""
    return len(enc.encode(text))


def chunk_text(text: str, size: int = CHUNK_TOKEN_SIZE, overlap: int = CHUNK_TOKEN_OVERLAP) -> Iterable[str]:
    """Yield overlapping token-window chunks from an input text."""
    tokens = enc.encode(text)
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        yield enc.decode(chunk_tokens)
        if end == len(tokens):
            break
        start = end - overlap


def sha1_bytes(b: bytes) -> str:
    """Compute a stable SHA1 for change detection."""
    return hashlib.sha1(b).hexdigest()


def md5_text(s: str) -> str:
    """Compute md5 used as a stable per-chunk identifier."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def guess_type_from_url(url: str, content_type_header: str = "") -> str:
    """Infer a coarse content type (html/pdf/tabular) using URL and HTTP headers."""
    ct = (content_type_header or "").lower()
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(".pdf") or "pdf" in ct:
        return "pdf"
    if path.endswith(".csv") or "text/csv" in ct:
        return "tabular"
    if path.endswith(".xlsx") or "officedocument.spreadsheetml.sheet" in ct:
        return "tabular"
    return "html"


# -----------------------
# Fetchers / loaders
# -----------------------

def fetch_http(
    url: str,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
) -> Tuple[Optional[bytes], requests.Response]:
    """Fetch a URL with optional conditional headers for cheap change detection."""
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    resp = requests.get(url, headers=headers, timeout=45, allow_redirects=True)
    if resp.status_code == 304:
        return None, resp
    resp.raise_for_status()
    return resp.content, resp


def read_file_url(file_url: str) -> Tuple[bytes, dict]:
    """Read a local file:// URL (useful for offline datasets)."""
    path = urllib.parse.urlparse(file_url).path
    if os.name == "nt" and path.startswith("/"):
        path = path[1:]
    with open(path, "rb") as f:
        data = f.read()
    meta = {"final_url": file_url, "headers": {}, "status_code": 200}
    return data, meta


# -----------------------
# Parsers
# -----------------------

def clean_html(html: str, base_url: str) -> Tuple[str, str]:
    """Extract readable text from HTML while stripping common page chrome."""
    soup = BeautifulSoup(html, "lxml")

    for sel in ["nav", "footer", "aside", "script", "style", "noscript", "form"]:
        for tag in soup.find_all(sel):
            tag.decompose()

    noisy = ["cookie", "banner", "subscribe", "social", "share", "breadcrumb", "sidebar", "search"]
    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", []))
        ident = (tag.get("id") or "")
        if any(k in classes.lower() or k in ident.lower() for k in noisy):
            tag.decompose()

    title_text = None
    if soup.title and soup.title.string:
        try:
            title_text = soup.title.string.strip()
        except Exception:
            title_text = None
    title = title_text or (urllib.parse.urlparse(base_url).path.strip("/") or "Untitled")

    main = soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from a PDF using pdfminer."""
    with io.BytesIO(content) as f:
        return pdf_extract_text(f)


def load_tabular(url: str, fmt: str = "auto") -> pd.DataFrame:
    """Load a CSV/XLSX dataset from http(s) or file:// into a DataFrame."""
    if url.startswith("file://"):
        path = urllib.parse.urlparse(url).path
        if os.name == "nt" and path.startswith("/"):
            path = path[1:]
        if fmt == "auto":
            fmt = "xlsx" if path.endswith(".xlsx") else "csv"
        if fmt == "xlsx":
            return pd.read_excel(path)
        return pd.read_csv(path)

    raw, resp = fetch_http(url)
    assert raw is not None
    cth = resp.headers.get("Content-Type", "")
    if fmt == "auto":
        path = urllib.parse.urlparse(resp.url).path.lower()
        if path.endswith(".xlsx") or "officedocument.spreadsheetml.sheet" in cth:
            fmt = "xlsx"
        else:
            fmt = "csv"
    if fmt == "xlsx":
        return pd.read_excel(io.BytesIO(raw))
    return pd.read_csv(io.StringIO(raw.decode("utf-8", errors="ignore")))


# -----------------------
# DB helpers
# -----------------------

def connect():
    """Open a Postgres connection (psycopg2)."""
    return psycopg2.connect(host=PGHOST, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE)


def get_profile_id(cur, profile_name: str) -> int:
    """Resolve a crawl profile name to its DB id."""
    cur.execute("SELECT id FROM crawl_profiles WHERE name=%s", (profile_name,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"crawl_profiles.name={profile_name!r} not found (run the crawler/bootstrap first)")
    return int(row[0])


def select_targets_for_ingest(cur, profile_id: int, limit: int) -> List[dict]:
    """Select the next batch of approved URLs to ingest, ordered by priority."""
    cur.execute(
        """
        SELECT id, url, etag, last_modified
        FROM crawl_targets
        WHERE profile_id=%s
          AND status IN ('approved','failed','crawled')
          AND (next_crawl_at IS NULL OR next_crawl_at <= NOW())
        ORDER BY priority DESC, last_crawled_at NULLS FIRST, discovered_at ASC
        LIMIT %s;
        """,
        (profile_id, limit),
    )
    out: List[dict] = []
    for rid, url, etag, last_modified in cur.fetchall():
        out.append({"crawl_target_id": rid, "url": url, "etag": etag, "last_modified": last_modified})
    return out


def create_ingest_run(cur, profile_id: int, meta: dict) -> int:
    """Create an ingest_runs row to capture per-run observability."""
    cur.execute(
        "INSERT INTO ingest_runs (profile_id, meta) VALUES (%s, %s) RETURNING id;",
        (profile_id, Json(meta)),
    )
    return int(cur.fetchone()[0])


def finish_ingest_run(cur, run_id: int, selected: int, ingested: int, skipped: int, failed: int, meta: dict) -> None:
    """Finalize an ingest_runs row with counts and a finish timestamp."""
    cur.execute(
        """
        UPDATE ingest_runs
        SET finished_at=NOW(), selected_count=%s, ingested_count=%s, skipped_count=%s, failed_count=%s, meta=%s
        WHERE id=%s;
        """,
        (selected, ingested, skipped, failed, Json(meta), run_id),
    )


def update_target_status(
    cur,
    target_id: int,
    *,
    status: str,
    http_status: Optional[int] = None,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    content_sha1: Optional[str] = None,
    content_bytes: Optional[int] = None,
    error: Optional[str] = None,
    next_crawl_at: Optional[datetime] = None,
    set_last_crawled: bool = True,
    set_last_ingested: bool = False,
) -> None:
    """Update crawl_targets row after a crawl/ingest attempt."""
    cur.execute(
        """
        UPDATE crawl_targets
        SET status=%s,
            last_http_status=COALESCE(%s, last_http_status),
            etag=COALESCE(%s, etag),
            last_modified=COALESCE(%s, last_modified),
            content_sha1=COALESCE(%s, content_sha1),
            content_bytes=COALESCE(%s, content_bytes),
            error=%s,
            next_crawl_at=%s,
            last_crawled_at=CASE WHEN %s THEN NOW() ELSE last_crawled_at END,
            last_ingested_at=CASE WHEN %s THEN NOW() ELSE last_ingested_at END,
            updated_at=NOW()
        WHERE id=%s;
        """,
        (
            status,
            http_status,
            etag,
            last_modified,
            content_sha1,
            content_bytes,
            error,
            next_crawl_at,
            set_last_crawled,
            set_last_ingested,
            target_id,
        ),
    )


def get_existing_source(cur, url: str) -> Tuple[Optional[int], Optional[str]]:
    """Return (source_id, content_sha1) for an existing source URL, if present."""
    cur.execute("SELECT id, content_sha1 FROM sources WHERE url=%s", (url,))
    row = cur.fetchone()
    if not row:
        return None, None
    return int(row[0]), (row[1] or None)


def upsert_source(
    cur,
    url: str,
    content_type: str,
    title: str,
    meta: dict,
    http_status: int,
    etag: Optional[str],
    content_sha1: Optional[str],
    content_bytes: Optional[int],
) -> int:
    """Upsert a source row using URL as the stable identity key."""
    cur.execute(
        """
        INSERT INTO sources (url, content_type, title, meta, http_status, etag, content_sha1, content_bytes, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO UPDATE SET
          content_type = EXCLUDED.content_type,
          title = EXCLUDED.title,
          meta = EXCLUDED.meta,
          http_status = EXCLUDED.http_status,
          etag = EXCLUDED.etag,
          content_sha1 = EXCLUDED.content_sha1,
          content_bytes = EXCLUDED.content_bytes,
          fetched_at = EXCLUDED.fetched_at,
          updated_at = NOW()
        RETURNING id;
        """,
        (
            url,
            content_type,
            title,
            Json(meta),
            http_status,
            etag,
            content_sha1,
            content_bytes,
            datetime.now(timezone.utc),
        ),
    )
    return int(cur.fetchone()[0])


def delete_chunks_for_source(cur, source_id: int) -> None:
    """Delete all chunks for a given source id so we can replace them on re-ingest."""
    cur.execute("DELETE FROM chunks WHERE source_id=%s;", (source_id,))


def insert_chunk(cur, source_id: int, url: str, section: Optional[str], chunk_text_: str, tokens: int, embedding) -> None:
    """Insert a single chunk (with embedding) while deduping by (url, chunk_md5)."""
    emb = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
    chunk_md5 = md5_text(f"{url}#{chunk_text_}")
    cur.execute(
        """
        INSERT INTO chunks (source_id, section, url, chunk, chunk_tokens, embedding, ts, chunk_md5)
        VALUES (%s, %s, %s, %s, %s, %s::vector, to_tsvector('english', %s), %s)
        ON CONFLICT (url, chunk_md5) DO NOTHING;
        """,
        (source_id, section, url, chunk_text_, tokens, emb, chunk_text_, chunk_md5),
    )


def embed_texts(texts: List[str]):
    """Embed many texts with a single batched call for speed."""
    return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)


# -----------------------
# Ingestion paths
# -----------------------

def ingest_html_or_pdf(
    cur,
    raw: bytes,
    final_url: str,
    content_type_header: str,
    status_code: int,
    etag: Optional[str],
    last_modified: Optional[str],
    *,
    replace_chunks: bool = True,
) -> Tuple[int, bool]:
    """Ingest an HTML/PDF document, returning (chunks_added, content_changed)."""
    ctype = guess_type_from_url(final_url, content_type_header)
    if ctype == "pdf":
        text = extract_text_from_pdf(raw)
        title = final_url.split("/")[-1] or "PDF"
        content_type = "pdf"
    else:
        title, text = clean_html(raw.decode("utf-8", errors="ignore"), final_url)
        content_type = "html"

    meta = {
        "final_url": final_url,
        "headers": {"Content-Type": content_type_header, "ETag": etag, "Last-Modified": last_modified},
    }
    content_bytes = len(raw)
    content_sha1 = sha1_bytes(raw)

    existing_id, existing_sha1 = get_existing_source(cur, final_url)
    changed = existing_sha1 != content_sha1

    source_id = upsert_source(
        cur,
        url=final_url,
        content_type=content_type,
        title=title,
        meta=meta,
        http_status=status_code,
        etag=etag,
        content_sha1=content_sha1,
        content_bytes=content_bytes,
    )

    # If content didn't change, keep existing chunks to avoid needless churn.
    if existing_id is not None and not changed:
        return 0, False

    if replace_chunks:
        delete_chunks_for_source(cur, source_id)

    sections = list(chunk_text(text))
    if not sections:
        return 0, changed

    embs = embed_texts(sections)
    inserted = 0
    for chunk_str, emb in zip(sections, embs):
        tokens = approx_token_len(chunk_str)
        insert_chunk(cur, source_id, final_url, section=None, chunk_text_=chunk_str, tokens=tokens, embedding=emb)
        inserted += 1

    return inserted, changed


def ingest_tabular(
    cur,
    df: pd.DataFrame,
    dataset_url: str,
    question_col: str = "Question",
    answer_col: str = "Answer",
    url_col: str = "SourceURL",
) -> Tuple[int, bool]:
    """Ingest a Q/A dataset (CSV/XLSX) into chunks for retrieval."""
    raw_bytes = df.to_csv(index=False).encode("utf-8")
    content_sha1 = sha1_bytes(raw_bytes)

    existing_id, existing_sha1 = get_existing_source(cur, dataset_url)
    changed = existing_sha1 != content_sha1

    meta = {"final_url": dataset_url, "headers": {}, "columns": list(df.columns)}
    title = (dataset_url.split("/")[-1] or "FAQ/Tabular").strip()
    source_id = upsert_source(
        cur,
        url=dataset_url,
        content_type="tabular",
        title=title,
        meta=meta,
        http_status=200,
        etag=None,
        content_sha1=content_sha1,
        content_bytes=len(raw_bytes),
    )

    if existing_id is not None and not changed:
        return 0, False

    delete_chunks_for_source(cur, source_id)

    rows_text: List[str] = []
    rows_url: List[str] = []
    rows_section: List[Optional[str]] = []

    q_col = question_col if question_col in df.columns else None
    a_col = answer_col if answer_col in df.columns else None
    s_col = url_col if url_col in df.columns else None

    for _, row in df.iterrows():
        q = str(row[q_col]).strip() if q_col else ""
        a = str(row[a_col]).strip() if a_col else ""
        if not a and not q:
            continue
        text = f"Q: {q}\nA: {a}".strip()
        cite_url = str(row[s_col]).strip() if s_col and pd.notna(row[s_col]) else dataset_url

        rows_text.append(text)
        rows_url.append(cite_url)
        rows_section.append(q[:240] if q else None)

    if not rows_text:
        return 0, changed

    embs = embed_texts(rows_text)
    inserted = 0
    for txt, emb, u, sec in zip(rows_text, embs, rows_url, rows_section):
        tokens = approx_token_len(txt)
        insert_chunk(cur, source_id, u, section=sec, chunk_text_=txt, tokens=tokens, embedding=emb)
        inserted += 1

    return inserted, changed


def ingest_entry(cur, entry: dict) -> Tuple[int, bool]:
    """Ingest a single entry dict and return (chunks_added, content_changed)."""
    url = entry["url"]

    if url.startswith("file://"):
        if entry.get("type") == "tabular":
            df = load_tabular(url, fmt=entry.get("fmt", "auto"))
            return ingest_tabular(
                cur,
                df,
                dataset_url=url,
                question_col=entry.get("question_col", "Question"),
                answer_col=entry.get("answer_col", "Answer"),
                url_col=entry.get("url_col", "SourceURL"),
            )

        raw, meta = read_file_url(url)
        return ingest_html_or_pdf(
            cur,
            raw,
            final_url=url,
            content_type_header="",
            status_code=int(meta.get("status_code", 200)),
            etag=None,
            last_modified=None,
        )

    # Remote
    if entry.get("type") == "tabular":
        df = load_tabular(url, fmt=entry.get("fmt", "auto"))
        return ingest_tabular(
            cur,
            df,
            dataset_url=url,
            question_col=entry.get("question_col", "Question"),
            answer_col=entry.get("answer_col", "Answer"),
            url_col=entry.get("url_col", "SourceURL"),
        )

    raw, resp = fetch_http(url, etag=entry.get("etag"), last_modified=entry.get("last_modified"))

    # Conditional fetch hit: nothing changed.
    if resp.status_code == 304:
        return 0, False

    assert raw is not None
    cth = resp.headers.get("Content-Type", "")
    return ingest_html_or_pdf(
        cur,
        raw,
        final_url=resp.url,
        content_type_header=cth,
        status_code=resp.status_code,
        etag=resp.headers.get("ETag"),
        last_modified=resp.headers.get("Last-Modified"),
    )


# -----------------------
# Config helpers
# -----------------------

def normalize_sources(cfg: dict) -> List[dict]:
    """Normalize legacy allowlist.yaml formats into a list of entry dicts."""
    out: List[dict] = []

    for u in cfg.get("urls", []):
        out.append({"url": u})

    for ent in cfg.get("sources", []):
        out.append(ent)

    for e in out:
        if "type" not in e:
            e["type"] = guess_type_from_url(e["url"])

    return out


# -----------------------
# Main entrypoints
# -----------------------

def run_yaml_mode(cur, allowlist_path: str, sleep_seconds: float) -> None:
    """Ingest sources from a YAML allowlist (dev + backwards compatible)."""
    if not os.path.exists(allowlist_path):
        raise FileNotFoundError(f"allowlist.yaml not found at {allowlist_path}")

    with open(allowlist_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    entries = normalize_sources(config)
    if not entries:
        print("No sources found in allowlist.yaml")
        return

    total = 0
    for ent in entries:
        try:
            print(f"Ingesting: {ent['url']} (type={ent.get('type')})")
            added, changed = ingest_entry(cur, ent)
            total += added
            print(f"  -> {added} chunks (changed={changed})")
        except Exception as e:
            raise RuntimeError(f"Failed ingest for {ent['url']}: {e}") from e
        time.sleep(sleep_seconds)

    print(f"Done. Total chunks added this run: {total}")


def backoff_after_failure(retries: int) -> datetime:
    """Compute the next crawl time for a failed URL using exponential backoff."""
    seconds = min(24 * 3600, 60 * (2 ** max(0, retries)))  # cap at 24h
    return datetime.now(timezone.utc) + pd.to_timedelta(seconds, unit="s")


def run_db_mode(cur, profile_name: str, limit: int, sleep_seconds: float) -> None:
    """Ingest sources from crawl_targets (production path)."""
    profile_id = get_profile_id(cur, profile_name)
    run_id = create_ingest_run(cur, profile_id, meta={"profile": profile_name, "limit": limit})

    selected = select_targets_for_ingest(cur, profile_id, limit)
    print(f"Selected {len(selected)} targets for profile={profile_name}")

    ingested = 0
    skipped = 0
    failed = 0

    for ent in selected:
        target_id = ent["crawl_target_id"]
        url = ent["url"]
        try:
            update_target_status(cur, target_id, status="queued", error=None, set_last_crawled=False)
            print(f"Ingesting (db): {url}")

            # Fetch with conditional headers so unchanged pages are cheap.
            raw, resp = fetch_http(url, etag=ent.get("etag"), last_modified=ent.get("last_modified"))

            # Conditional fetch hit: nothing changed.
            if resp.status_code == 304:
                skipped += 1
                update_target_status(
                    cur,
                    target_id,
                    status="crawled",
                    http_status=304,
                    etag=resp.headers.get("ETag") or ent.get("etag"),
                    last_modified=resp.headers.get("Last-Modified") or ent.get("last_modified"),
                    error=None,
                    set_last_crawled=True,
                    set_last_ingested=False,
                )
                print("  -> 304 Not Modified")
                time.sleep(sleep_seconds)
                continue

            assert raw is not None
            final_url = resp.url
            cth = resp.headers.get("Content-Type", "")

            added, changed = ingest_html_or_pdf(
                cur,
                raw,
                final_url=final_url,
                content_type_header=cth,
                status_code=resp.status_code,
                etag=resp.headers.get("ETag"),
                last_modified=resp.headers.get("Last-Modified"),
            )

            # Keep crawl_targets fresh with change detection signals.
            content_sha1 = sha1_bytes(raw)
            content_bytes = len(raw)

            if changed:
                ingested += 1
                update_target_status(
                    cur,
                    target_id,
                    status="ingested",
                    http_status=resp.status_code,
                    etag=resp.headers.get("ETag"),
                    last_modified=resp.headers.get("Last-Modified"),
                    content_sha1=content_sha1,
                    content_bytes=content_bytes,
                    error=None,
                    set_last_crawled=True,
                    set_last_ingested=True,
                )
            else:
                skipped += 1
                update_target_status(
                    cur,
                    target_id,
                    status="crawled",
                    http_status=resp.status_code,
                    etag=resp.headers.get("ETag"),
                    last_modified=resp.headers.get("Last-Modified"),
                    content_sha1=content_sha1,
                    content_bytes=content_bytes,
                    error=None,
                    set_last_crawled=True,
                    set_last_ingested=False,
                )

            print(f"  -> {added} chunks (changed={changed})")

        except Exception as e:
            failed += 1
            # Increment retries and compute backoff.
            cur.execute("SELECT retries FROM crawl_targets WHERE id=%s", (target_id,))
            rrow = cur.fetchone()
            retries = int(rrow[0]) + 1 if rrow else 1
            next_at = backoff_after_failure(retries)
            cur.execute("UPDATE crawl_targets SET retries=%s WHERE id=%s", (retries, target_id))
            update_target_status(
                cur,
                target_id,
                status="failed",
                error=str(e)[:1000],
                next_crawl_at=next_at,
                set_last_crawled=True,
                set_last_ingested=False,
            )
            print(f"Failed: {url} ({e})")

        time.sleep(sleep_seconds)

    finish_ingest_run(cur, run_id, selected=len(selected), ingested=ingested, skipped=skipped, failed=failed, meta={"profile": profile_name})
    print(f"Done. selected={len(selected)} ingested={ingested} skipped={skipped} failed={failed}")


def main() -> None:
    """CLI entrypoint for running ingestion in yaml or db mode."""
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG database.")
    parser.add_argument("--mode", choices=["yaml", "db"], default=os.getenv("INGEST_MODE", "yaml"))
    parser.add_argument("--allowlist", default=os.getenv("ALLOWLIST_PATH", "/app/allowlist.yaml"))
    parser.add_argument("--profile", default=os.getenv("CRAWL_PROFILE", "arts"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("INGEST_LIMIT", "200")))
    parser.add_argument("--sleep", type=float, default=float(os.getenv("INGEST_SLEEP_SECONDS", "0.3")))
    args = parser.parse_args()

    conn = connect()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            if args.mode == "yaml":
                run_yaml_mode(cur, args.allowlist, args.sleep)
            else:
                run_db_mode(cur, args.profile, args.limit, args.sleep)
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
