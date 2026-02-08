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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import psycopg2
import requests
import tiktoken
import yaml
from bs4 import BeautifulSoup

# Optional JS-rendering support for modern sites
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
except Exception:  # pragma: no cover
    sync_playwright = None
    PlaywrightTimeoutError = Exception
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


def _env_bool(name: str, default: str = "false") -> bool:
    v = os.getenv(name, default)
    return str(v).lower() in {"1", "true", "yes", "y", "on"}

# Playwright controls (JS-rendering during ingestion)
INGEST_USE_PLAYWRIGHT = _env_bool("INGEST_USE_PLAYWRIGHT", "false")
INGEST_PLAYWRIGHT_ALWAYS = _env_bool("INGEST_PLAYWRIGHT_ALWAYS", "false")
INGEST_PLAYWRIGHT_FALLBACK = _env_bool("INGEST_PLAYWRIGHT_FALLBACK", "true")
INGEST_MIN_EXTRACTED_CHARS = int(os.getenv("INGEST_MIN_EXTRACTED_CHARS", "400"))
PLAYWRIGHT_EXPAND_ACCORDIONS = _env_bool("PLAYWRIGHT_EXPAND_ACCORDIONS", "true")
PLAYWRIGHT_NAV_TIMEOUT_MS = int(os.getenv("PLAYWRIGHT_NAV_TIMEOUT_MS", "45000"))
PLAYWRIGHT_WAIT_UNTIL = os.getenv("PLAYWRIGHT_WAIT_UNTIL", "domcontentloaded")

# Model + tokenizer are initialized once to amortize startup cost per ingestion run
model = SentenceTransformer(EMBED_MODEL_NAME)
enc = tiktoken.get_encoding("cl100k_base")


def _as_dict(maybe_mapping: Any) -> Dict[str, Any]:
    """Best-effort conversion of possibly-None/Mapping-ish values into a dict.

    We occasionally see libraries return `headers=None` or other mapping-like objects.
    This helper keeps the ingestion pipeline resilient.
    """
    if not maybe_mapping:
        return {}
    if isinstance(maybe_mapping, dict):
        return maybe_mapping
    try:
        return dict(maybe_mapping)
    except Exception:
        return {}


def _looks_like_html(raw: bytes) -> bool:
    head = (raw or b"")[:4096].lstrip().lower()
    return head.startswith(b"<!doctype") or b"<html" in head


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



class PlaywrightFetcher:
    """Lightweight, reusable Playwright fetcher for JS-rendered pages.

    We keep a single browser instance per ingestion run and create a fresh context per URL.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and sync_playwright is not None
        self._p = None
        self._browser = None

    def __enter__(self):
        if self.enabled:
            self._p = sync_playwright().start()
            # Chromium tends to be the most compatible choice for campus websites.
            self._browser = self._p.chromium.launch(headless=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._p:
                self._p.stop()
        except Exception:
            pass
        self._browser = None
        self._p = None

    def fetch_html(self, url: str) -> Tuple[bytes, dict]:
        if not self.enabled or self._browser is None:
            raise RuntimeError("Playwright is not enabled/available")

        start = time.time()
        ctx = self._browser.new_context(user_agent=USER_AGENT)
        page = ctx.new_page()
        resp = None
        try:
            resp = page.goto(url, wait_until=PLAYWRIGHT_WAIT_UNTIL, timeout=PLAYWRIGHT_NAV_TIMEOUT_MS)

            # Many pages render key content into <main>.
            try:
                page.wait_for_selector("main", timeout=3000)
            except Exception:
                pass

            if PLAYWRIGHT_EXPAND_ACCORDIONS:
                # Best-effort: expand common accordion patterns before extracting.
                js = """() => {
                  try {
                    document.querySelectorAll('details').forEach(d => d.setAttribute('open',''));
                  } catch (e) {}
                  const btns = Array.from(document.querySelectorAll('[aria-expanded="false"]'))
                    .filter(el => (el.tagName === 'BUTTON' || el.getAttribute('role') === 'button'));
                  for (const b of btns) {
                    try { b.click(); } catch (e) {}
                  }
                }"""
                try:
                    page.evaluate(js)
                    page.wait_for_timeout(400)
                except Exception:
                    pass

            html = page.content()
            final_url = page.url
            status_code = int(resp.status) if resp is not None else 200
            headers = dict(resp.headers) if resp is not None else {}
            duration_ms = int((time.time() - start) * 1000)
            meta = {"final_url": final_url, "status_code": status_code, "headers": headers, "duration_ms": duration_ms}
            return html.encode("utf-8", errors="ignore"), meta
        finally:
            try:
                ctx.close()
            except Exception:
                pass


def _truncate(s: str, n: int = 500) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def _classify_exception(e: Exception) -> str:
    if isinstance(e, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(e, requests.exceptions.ConnectionError):
        return "connection_error"
    if isinstance(e, requests.exceptions.HTTPError):
        return "http_error"
    if isinstance(e, PlaywrightTimeoutError):
        return "playwright_timeout"
    if isinstance(e, requests.exceptions.RequestException):
        return "request_error"
    return e.__class__.__name__


def fetch_remote_with_fallback(
    url: str,
    *,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    pw: Optional[PlaywrightFetcher] = None,
) -> dict:
    """Fetch a remote URL, optionally using Playwright for JS-heavy pages.

    Returns a dict with keys:
      - raw: Optional[bytes]
      - status_code: int
      - final_url: str
      - headers: dict
      - fetcher: 'requests' | 'playwright'
      - parsed_title/parsed_text: optional, for HTML pages
      - extracted_chars: optional
      - duration_ms: int
    """

    started = time.time()

    def _make_result(*, raw: Optional[bytes], status_code: int, final_url: str, headers: dict, fetcher: str,
                     parsed_title: Optional[str] = None, parsed_text: Optional[str] = None,
                     extracted_chars: Optional[int] = None, note: Optional[str] = None) -> dict:
        duration_ms = int((time.time() - started) * 1000)
        out = {
            "raw": raw,
            "status_code": int(status_code),
            "final_url": final_url,
            "headers": dict(headers or {}),
            "fetcher": fetcher,
            "duration_ms": duration_ms,
        }
        if parsed_title is not None:
            out["parsed_title"] = parsed_title
        if parsed_text is not None:
            out["parsed_text"] = parsed_text
        if extracted_chars is not None:
            out["extracted_chars"] = int(extracted_chars)
        if note:
            out["note"] = note
        return out

    # If configured, go straight to Playwright for HTML pages (slow but reliable).
    if INGEST_USE_PLAYWRIGHT and INGEST_PLAYWRIGHT_ALWAYS and pw and pw.enabled and not url.lower().endswith(".pdf"):
        raw_pw, meta_pw = pw.fetch_html(url)
        meta_pw = meta_pw or {}
        headers_pw = meta_pw.get("headers") or {}
        # Playwright/servers sometimes vary header casing.
        ctype = headers_pw.get("content-type") or headers_pw.get("Content-Type") or ""
        final_url = meta_pw.get("final_url") or url
        # Pre-parse HTML to reuse later
        title, text = clean_html(raw_pw.decode("utf-8", errors="ignore"), final_url)
        return _make_result(
            raw=raw_pw,
            status_code=int(meta_pw.get("status_code") or 200),
            final_url=final_url,
            headers=headers_pw or {"Content-Type": ctype},
            fetcher="playwright",
            parsed_title=title,
            parsed_text=text,
            extracted_chars=len(text),
            note="playwright_always",
        )

    # Default: requests first
    try:
        raw, resp = fetch_http(url, etag=etag, last_modified=last_modified)
        if resp.status_code == 304:
            return _make_result(raw=None, status_code=304, final_url=resp.url, headers=resp.headers, fetcher="requests", note="not_modified")

        assert raw is not None
        cth = resp.headers.get("Content-Type", "")
        final_url = resp.url

        # Decide whether we should render with Playwright.
        wants_playwright = False
        parsed_title = parsed_text = None
        extracted_chars = None

        if INGEST_USE_PLAYWRIGHT and INGEST_PLAYWRIGHT_FALLBACK and pw and pw.enabled:
            # Only attempt for likely-HTML.
            if guess_type_from_url(final_url, cth) == "html":
                try:
                    parsed_title, parsed_text = clean_html(raw.decode("utf-8", errors="ignore"), final_url)
                    extracted_chars = len(parsed_text)
                    if extracted_chars < INGEST_MIN_EXTRACTED_CHARS:
                        wants_playwright = True
                    # Heuristic: common JS-required interstitials
                    low = raw[:20000].decode("utf-8", errors="ignore").lower()
                    if "enable javascript" in low or "please enable javascript" in low:
                        wants_playwright = True
                except Exception:
                    # If parsing fails, try Playwright as a last resort.
                    wants_playwright = True

        if wants_playwright:
            try:
                raw_pw, meta_pw = pw.fetch_html(url)
                meta_pw = meta_pw or {}
                headers_pw = meta_pw.get("headers") or {}
                final_pw = meta_pw.get("final_url") or url
                title_pw, text_pw = clean_html(raw_pw.decode("utf-8", errors="ignore"), final_pw)
                if len(text_pw) >= (extracted_chars or 0):
                    return _make_result(
                        raw=raw_pw,
                        status_code=int(meta_pw.get("status_code") or 200),
                        final_url=final_pw,
                        headers=headers_pw,
                        fetcher="playwright",
                        parsed_title=title_pw,
                        parsed_text=text_pw,
                        extracted_chars=len(text_pw),
                        note="playwright_fallback",
                    )
            except Exception:
                # If fallback fails, keep requests result but annotate via note.
                pass

        return _make_result(
            raw=raw,
            status_code=resp.status_code,
            final_url=final_url,
            headers=resp.headers,
            fetcher="requests",
            parsed_title=parsed_title,
            parsed_text=parsed_text,
            extracted_chars=extracted_chars,
        )

    except requests.exceptions.RequestException as e:
        # Requests failed (timeout, 403, etc). Try Playwright once if enabled.
        if INGEST_USE_PLAYWRIGHT and INGEST_PLAYWRIGHT_FALLBACK and pw and pw.enabled and not url.lower().endswith(".pdf"):
            raw_pw, meta_pw = pw.fetch_html(url)
            meta_pw = meta_pw or {}
            headers_pw = meta_pw.get("headers") or {}
            final_pw = meta_pw.get("final_url") or url
            title_pw, text_pw = clean_html(raw_pw.decode("utf-8", errors="ignore"), final_pw)
            return _make_result(
                raw=raw_pw,
                status_code=int(meta_pw.get("status_code") or 200),
                final_url=final_pw,
                headers=headers_pw,
                fetcher="playwright",
                parsed_title=title_pw,
                parsed_text=text_pw,
                extracted_chars=len(text_pw),
                note=f"playwright_after_requests_error:{_classify_exception(e)}",
            )
        raise

# -----------------------
# Parsers
# -----------------------

def _naive_html_to_text(html: str, base_url: str) -> Tuple[str, str]:
    """Very defensive HTML->text fallback (no DOM assumptions)."""
    # Title (best effort)
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title_text = None
    if m:
        title_text = re.sub(r"\s+", " ", m.group(1)).strip()

    # Drop script/style/noscript blocks, then strip tags.
    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<noscript[^>]*>.*?</noscript>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = htmlmod.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace(" ", " ")

    title = title_text or (urllib.parse.urlparse(base_url).path.strip("/") or "Untitled")
    return title, cleaned


def clean_html(html: str, base_url: str) -> Tuple[str, str]:
    """Extract readable text from HTML while stripping common page chrome.

    This should never raise: if parsing fails for any reason, we fall back to a
    very naive tag-stripper to keep ingestion moving.
    """
    try:
        soup = BeautifulSoup(html, "lxml")

        for sel in ["nav", "footer", "aside", "script", "style", "noscript", "form"]:
            for tag in soup.find_all(sel):
                tag.decompose()

        noisy = ["cookie", "banner", "subscribe", "social", "share", "breadcrumb", "sidebar", "search"]
        for tag in soup.find_all(True):
            # Defensive: some parsers / malformed markup can yield unexpected nodes.
            if not hasattr(tag, "get"):
                continue
            try:
                cls_val = tag.get("class") or []
                if isinstance(cls_val, (list, tuple)):
                    classes = " ".join(str(x) for x in cls_val)
                else:
                    classes = str(cls_val)
                ident = str(tag.get("id") or "")
            except Exception:
                continue

            if any(k in classes.lower() or k in ident.lower() for k in noisy):
                try:
                    tag.decompose()
                except Exception:
                    pass

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
    except Exception:
        return _naive_html_to_text(html, base_url)


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
    meta_patch: Optional[dict] = None,
) -> None:
    """Update a crawl_targets row after a crawl/ingest attempt.

    We merge meta_patch into the existing meta JSONB so crawler reasons and ingest diagnostics coexist.
    """
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
            meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb,
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
            Json(meta_patch or {}),
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
    requested_url: Optional[str] = None,
    fetcher: str = "requests",
    parsed_title: Optional[str] = None,
    parsed_text: Optional[str] = None,
) -> Tuple[int, bool, int]:
    """Ingest an HTML/PDF document.

    Returns (chunks_added, content_changed, extracted_chars).

    Note: parsed_title/parsed_text are optional pre-parsed results (used to avoid parsing twice
    when we already inspected the page to decide whether to use Playwright).
    """
    ctype = guess_type_from_url(final_url, content_type_header)

    if ctype == "pdf":
        text = extract_text_from_pdf(raw)
        title = final_url.split("/")[-1] or "PDF"
        content_type = "pdf"
    else:
        if parsed_title is not None and parsed_text is not None:
            title, text = parsed_title, parsed_text
        else:
            title, text = clean_html(raw.decode("utf-8", errors="ignore"), final_url)
        content_type = "html"

    extracted_chars = len(text)
    extracted_tokens = approx_token_len(text) if text else 0

    meta = {
        "requested_url": requested_url or final_url,
        "final_url": final_url,
        "fetcher": fetcher,
        "headers": {"Content-Type": content_type_header, "ETag": etag, "Last-Modified": last_modified},
        "extracted": {"chars": extracted_chars, "tokens": extracted_tokens},
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
        return 0, False, extracted_chars

    if replace_chunks:
        delete_chunks_for_source(cur, source_id)

    sections = list(chunk_text(text))
    if not sections:
        return 0, changed, extracted_chars

    embs = embed_texts(sections)
    inserted = 0
    for chunk_str, emb in zip(sections, embs):
        tokens = approx_token_len(chunk_str)
        insert_chunk(cur, source_id, final_url, section=None, chunk_text_=chunk_str, tokens=tokens, embedding=emb)
        inserted += 1

    return inserted, changed, extracted_chars



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


def ingest_entry(cur, entry: dict, pw: Optional[PlaywrightFetcher] = None) -> Tuple[int, bool]:
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
        added, changed, _chars = ingest_html_or_pdf(
            cur,
            raw,
            final_url=url,
            content_type_header="",
            status_code=int(meta.get("status_code", 200)),
            etag=None,
            last_modified=None,
            requested_url=url,
            fetcher="file",
        )
        return added, changed

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

    res = fetch_remote_with_fallback(
        url,
        etag=entry.get("etag"),
        last_modified=entry.get("last_modified"),
        pw=pw,
    )

    # Conditional fetch hit: nothing changed.
    if res["status_code"] == 304:
        return 0, False

    raw = res["raw"]
    assert raw is not None

    added, changed, _chars = ingest_html_or_pdf(
        cur,
        raw,
        final_url=res["final_url"],
        content_type_header=(res.get("headers") or {}).get("Content-Type", (res.get("headers") or {}).get("content-type", "")),
        status_code=int(res["status_code"]),
        etag=(res.get("headers") or {}).get("ETag"),
        last_modified=(res.get("headers") or {}).get("Last-Modified"),
        requested_url=url,
        fetcher=res.get("fetcher", "requests"),
        parsed_title=res.get("parsed_title"),
        parsed_text=res.get("parsed_text"),
    )
    return added, changed


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
    with PlaywrightFetcher(INGEST_USE_PLAYWRIGHT) as pw:
        for ent in entries:
            try:
                print(f"Ingesting: {ent['url']} (type={ent.get('type')})")
                added, changed = ingest_entry(cur, ent, pw=pw)
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

    with PlaywrightFetcher(INGEST_USE_PLAYWRIGHT) as pw:
        if INGEST_USE_PLAYWRIGHT and sync_playwright is None:
            print("WARNING: INGEST_USE_PLAYWRIGHT=true but Playwright is not installed; proceeding without JS rendering")

        for ent in selected:
            target_id = ent["crawl_target_id"]
            url = ent["url"]
            started = time.time()

            # Cheap guardrail: some pages contain email addresses in the path (often not real content pages)
            # e.g. /contact/name@torontomu.ca. These almost always 404 and just add noise.
            try:
                if "@" in (urllib.parse.urlparse(url).path or ""):
                    skipped += 1
                    update_target_status(
                        cur,
                        target_id,
                        status="blocked",
                        error="blocked:email_in_path",
                        set_last_crawled=False,
                        set_last_ingested=False,
                        meta_patch={
                            "reason": "email_in_path",
                            "last_ingest": {
                                "requested_url": url,
                                "stage": "blocked",
                                "note": "email_in_path",
                                "at": datetime.now(timezone.utc).isoformat(),
                            },
                        },
                    )
                    print(f"Blocked (email-like URL): {url}")
                    time.sleep(sleep_seconds)
                    continue
            except Exception:
                # Never fail ingestion because of this guardrail.
                pass

            try:
                update_target_status(
                    cur,
                    target_id,
                    status="queued",
                    error=None,
                    set_last_crawled=False,
                    meta_patch={"last_ingest": {"stage": "queued", "requested_url": url, "at": datetime.now(timezone.utc).isoformat()}},
                )

                print(f"Ingesting (db): {url}")

                res = fetch_remote_with_fallback(
                    url,
                    etag=ent.get("etag"),
                    last_modified=ent.get("last_modified"),
                    pw=pw,
                )

                # Conditional fetch hit: nothing changed.
                if res["status_code"] == 304:
                    skipped += 1
                    update_target_status(
                        cur,
                        target_id,
                        status="crawled",
                        http_status=304,
                        etag=(res.get("headers") or {}).get("ETag") or ent.get("etag"),
                        last_modified=(res.get("headers") or {}).get("Last-Modified") or ent.get("last_modified"),
                        error=None,
                        set_last_crawled=True,
                        set_last_ingested=False,
                        meta_patch={
                            "last_ingest": {
                                "requested_url": url,
                                "final_url": res.get("final_url", url),
                                "fetcher": res.get("fetcher", "requests"),
                                "http_status": 304,
                                "duration_ms": res.get("duration_ms"),
                                "note": res.get("note"),
                                "at": datetime.now(timezone.utc).isoformat(),
                            }
                        },
                    )
                    print("  -> 304 Not Modified")
                    time.sleep(sleep_seconds)
                    continue

                raw = res["raw"]
                assert raw is not None

                final_url = res.get("final_url") or url
                headers = res.get("headers") or {}
                cth = headers.get("Content-Type", headers.get("content-type", ""))

                added, changed, extracted_chars = ingest_html_or_pdf(
                    cur,
                    raw,
                    final_url=final_url,
                    content_type_header=cth,
                    status_code=int(res["status_code"]),
                    etag=headers.get("ETag"),
                    last_modified=headers.get("Last-Modified"),
                    requested_url=url,
                    fetcher=res.get("fetcher", "requests"),
                    parsed_title=res.get("parsed_title"),
                    parsed_text=res.get("parsed_text"),
                )

                # Keep crawl_targets fresh with change detection signals.
                content_sha1 = sha1_bytes(raw)
                content_bytes = len(raw)

                status = "ingested" if changed else "crawled"
                if changed:
                    ingested += 1
                else:
                    skipped += 1

                update_target_status(
                    cur,
                    target_id,
                    status=status,
                    http_status=int(res["status_code"]),
                    etag=headers.get("ETag"),
                    last_modified=headers.get("Last-Modified"),
                    content_sha1=content_sha1,
                    content_bytes=content_bytes,
                    error=None,
                    set_last_crawled=True,
                    set_last_ingested=bool(changed),
                    meta_patch={
                        "last_ingest": {
                            "requested_url": url,
                            "final_url": final_url,
                            "fetcher": res.get("fetcher", "requests"),
                            "http_status": int(res["status_code"]),
                            "content_type": cth,
                            "content_bytes": content_bytes,
                            "extracted_chars": int(extracted_chars),
                            "duration_ms": res.get("duration_ms"),
                            "note": res.get("note"),
                            "stage": status,
                            "at": datetime.now(timezone.utc).isoformat(),
                        }
                    },
                )

                print(f"  -> {added} chunks (changed={changed}, extracted_chars={extracted_chars})")

            except Exception as e:
                failed += 1

                # Increment retries and compute backoff.
                cur.execute("SELECT retries FROM crawl_targets WHERE id=%s", (target_id,))
                rrow = cur.fetchone()
                retries = int(rrow[0]) + 1 if rrow else 1
                next_at = backoff_after_failure(retries)
                cur.execute("UPDATE crawl_targets SET retries=%s WHERE id=%s", (retries, target_id))

                # Best-effort extract status code if this was an HTTPError.
                http_status = None
                if isinstance(e, requests.exceptions.HTTPError) and getattr(e, "response", None) is not None:
                    try:
                        http_status = int(e.response.status_code)
                    except Exception:
                        http_status = None

                err_type = _classify_exception(e)
                update_target_status(
                    cur,
                    target_id,
                    status="failed",
                    http_status=http_status,
                    error=_truncate(str(e), 1000),
                    next_crawl_at=next_at,
                    set_last_crawled=True,
                    set_last_ingested=False,
                    meta_patch={
                        "last_ingest": {
                            "requested_url": url,
                            "error_type": err_type,
                            "error": _truncate(str(e), 1000),
                            "http_status": http_status,
                            "duration_ms": int((time.time() - started) * 1000),
                            "stage": "failed",
                            "at": datetime.now(timezone.utc).isoformat(),
                        }
                    },
                )
                print(f"Failed: {url} ({err_type}: {e})")
                if os.environ.get("INGEST_TRACEBACK", "0") == "1":
                    import traceback

                    traceback.print_exc()

            time.sleep(sleep_seconds)

    finish_ingest_run(
        cur,
        run_id,
        selected=len(selected),
        ingested=ingested,
        skipped=skipped,
        failed=failed,
        meta={"profile": profile_name},
    )
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
