import hashlib
import io
import os
import re
import time
import urllib.parse
import yaml
import json
from datetime import datetime, timezone
from typing import Iterable, List, Tuple

import psycopg2
from psycopg2.extras import Json
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from sentence_transformers import SentenceTransformer
import tiktoken
import pandas as pd

PGHOST = os.getenv("PGHOST", "pg")
PGUSER = os.getenv("PGUSER", "rag")
PGPASSWORD = os.getenv("PGPASSWORD", "rag")
PGDATABASE = os.getenv("PGDATABASE", "ragdb")

# 384-dim model to match schema VECTOR(384)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKEN_SIZE = 500
CHUNK_TOKEN_OVERLAP = 80
USER_AGENT = "TMU-FOA-RAG-Ingest/0.1 (+https://www.torontomu.ca/arts/)"

model = SentenceTransformer(EMBED_MODEL_NAME)
enc = tiktoken.get_encoding("cl100k_base")

# -----------------------
# Utilities
# -----------------------
def approx_token_len(text: str) -> int:
    return len(enc.encode(text))

def chunk_text(text: str, size=CHUNK_TOKEN_SIZE, overlap=CHUNK_TOKEN_OVERLAP) -> Iterable[str]:
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
    return hashlib.sha1(b).hexdigest()

def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def guess_type_from_url(url: str, content_type_header: str = "") -> str:
    ct = (content_type_header or "").lower()
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(".pdf") or "pdf" in ct:
        return "pdf"
    if path.endswith(".csv") or "text/csv" in ct:
        return "tabular"
    if path.endswith(".xlsx") or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in ct:
        return "tabular"
    return "html"

# -----------------------
# Fetchers / loaders
# -----------------------
def fetch_http(url: str) -> Tuple[bytes, requests.Response]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    resp = requests.get(url, headers=headers, timeout=45, allow_redirects=True)
    resp.raise_for_status()
    return resp.content, resp

def read_file_url(file_url: str) -> Tuple[bytes, dict]:
    # file:///shared/xxx.csv
    path = urllib.parse.urlparse(file_url).path
    if os.name == "nt" and path.startswith("/"):  # windows container path quirk
        path = path[1:]
    with open(path, "rb") as f:
        data = f.read()
    meta = {"final_url": file_url, "headers": {}, "status_code": 200}
    return data, meta

# -----------------------
# Parsers
# -----------------------
def clean_html(html: str, base_url: str) -> Tuple[str, str]:
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

    # Guard against NoneType title edge cases
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
    with io.BytesIO(content) as f:
        return pdf_extract_text(f)

def load_tabular(url: str, fmt: str = "auto") -> pd.DataFrame:
    # Supports http(s) and file://
    if url.startswith("file://"):
        path = urllib.parse.urlparse(url).path
        if os.name == "nt" and path.startswith("/"):
            path = path[1:]
        if fmt == "auto":
            fmt = "xlsx" if path.endswith(".xlsx") else "csv"
        if fmt == "xlsx":
            return pd.read_excel(path)
        return pd.read_csv(path)
    else:
        raw, resp = fetch_http(url)
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
    return psycopg2.connect(
        host=PGHOST, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE
    )

def upsert_source(cur, url, content_type, title, meta, http_status, etag, content_sha1, content_bytes):
    cur.execute("""
        INSERT INTO sources (url, content_type, title, meta, http_status, etag, content_sha1, content_bytes, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_sha1) DO UPDATE SET
          url = EXCLUDED.url,
          title = EXCLUDED.title,
          meta = EXCLUDED.meta,
          http_status = EXCLUDED.http_status,
          etag = EXCLUDED.etag,
          content_bytes = EXCLUDED.content_bytes,
          fetched_at = EXCLUDED.fetched_at,
          updated_at = NOW()
        RETURNING id;
    """, (url, content_type, title, Json(meta), http_status, etag, content_sha1, content_bytes, datetime.now(timezone.utc)))
    return cur.fetchone()[0]

def insert_chunk(cur, source_id, url, section, chunk_text, tokens, embedding):
    emb = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
    chunk_md5 = md5_text(f"{url}#{chunk_text}")
    cur.execute("""
        INSERT INTO chunks (source_id, section, url, chunk, chunk_tokens, embedding, ts, chunk_md5)
        VALUES (%s, %s, %s, %s, %s, %s::vector, to_tsvector('english', %s), %s)
        ON CONFLICT (url, chunk_md5) DO NOTHING;
    """, (source_id, section, url, chunk_text, tokens, emb, chunk_text, chunk_md5))

def embed_texts(texts: List[str]):
    return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

# -----------------------
# Ingestion paths
# -----------------------
def ingest_html_or_pdf(cur, raw: bytes, final_url: str, content_type_header: str, status_code: int, etag: str):
    ctype = guess_type_from_url(final_url, content_type_header)
    if ctype == "pdf":
        text = extract_text_from_pdf(raw)
        title = final_url.split("/")[-1] or "PDF"
        content_type = "pdf"
    else:
        # encoding handled by requests; fallback utf-8
        title, text = clean_html(raw.decode("utf-8", errors="ignore"), final_url)
        content_type = "html"

    meta = {"final_url": final_url, "headers": {"Content-Type": content_type_header, "ETag": etag}}
    content_bytes = len(raw)
    content_sha1 = sha1_bytes(raw)

    source_id = upsert_source(
        cur,
        url=final_url, content_type=content_type, title=title,
        meta=meta, http_status=status_code, etag=etag,
        content_sha1=content_sha1, content_bytes=content_bytes
    )

    sections = list(chunk_text(text))
    if not sections:
        return 0

    embs = embed_texts(sections)
    inserted = 0
    for chunk_str, emb in zip(sections, embs):
        tokens = approx_token_len(chunk_str)
        insert_chunk(cur, source_id, final_url, section=None, chunk_text=chunk_str, tokens=tokens, embedding=emb)
        inserted += 1
    return inserted

def ingest_tabular(cur, df: pd.DataFrame, dataset_url: str,
                   question_col: str = "Question", answer_col: str = "Answer", url_col: str = "SourceURL"):
    # Build a stable raw representation to dedupe 'sources'
    raw_bytes = df.to_csv(index=False).encode("utf-8")
    content_sha1 = sha1_bytes(raw_bytes)
    meta = {"final_url": dataset_url, "headers": {}, "columns": list(df.columns)}
    title = (dataset_url.split("/")[-1] or "FAQ/Tabular").strip()
    source_id = upsert_source(
        cur,
        url=dataset_url, content_type="tabular", title=title,
        meta=meta, http_status=200, etag=None,
        content_sha1=content_sha1, content_bytes=len(raw_bytes)
    )

    rows_text: List[str] = []
    rows_url: List[str] = []
    rows_section: List[str] = []

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
        rows_section.append(q[:240] if q else None)  # use question as section to boost FTS

    if not rows_text:
        return 0

    embs = embed_texts(rows_text)
    inserted = 0
    for txt, emb, u, sec in zip(rows_text, embs, rows_url, rows_section):
        tokens = approx_token_len(txt)
        insert_chunk(cur, source_id, u, section=sec, chunk_text=txt, tokens=tokens, embedding=emb)
        inserted += 1
    return inserted

def ingest_entry(cur, entry: dict) -> int:
    """
    entry:
      - if using old format: {"url": "..."} with optional inferred type
      - rich format: {"type": "html|pdf|tabular", "url": "...", "fmt": "auto|csv|xlsx", "question_col": "...", "answer_col": "...", "url_col": "..."}
    """
    url = entry["url"]
    if url.startswith("file://"):
        raw, meta = read_file_url(url)
        # Tabular local files handled via pandas path directly, so only use this if not tabular
        if entry.get("type") == "tabular":
            df = load_tabular(url, fmt=entry.get("fmt", "auto"))
            return ingest_tabular(cur, df, dataset_url=url,
                                  question_col=entry.get("question_col", "Question"),
                                  answer_col=entry.get("answer_col", "Answer"),
                                  url_col=entry.get("url_col", "SourceURL"))
        else:
            return ingest_html_or_pdf(cur, raw, final_url=url, content_type_header="", status_code=200, etag=None)
    else:
        # Remote
        if entry.get("type") == "tabular":
            df = load_tabular(url, fmt=entry.get("fmt", "auto"))
            return ingest_tabular(cur, df, dataset_url=url,
                                  question_col=entry.get("question_col", "Question"),
                                  answer_col=entry.get("answer_col", "Answer"),
                                  url_col=entry.get("url_col", "SourceURL"))
        else:
            raw, resp = fetch_http(url)
            cth = resp.headers.get("Content-Type", "")
            return ingest_html_or_pdf(cur, raw, final_url=resp.url,
                                      content_type_header=cth,
                                      status_code=resp.status_code,
                                      etag=resp.headers.get("ETag"))

# -----------------------
# Config + main
# -----------------------
def normalize_sources(cfg: dict) -> List[dict]:
    out = []
    # Back-compat: simple list
    for u in cfg.get("urls", []):
        out.append({"url": u})
    # Rich format
    for ent in cfg.get("sources", []):
        out.append(ent)
    # Infer types where missing
    for e in out:
        if "type" not in e:
            e["type"] = guess_type_from_url(e["url"])
    return out

def main():
    cfg_path = "/app/allowlist.yaml"
    if not os.path.exists(cfg_path):
        print("allowlist.yaml not found at /app/allowlist.yaml")
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    entries = normalize_sources(config)
    if not entries:
        print("No sources found in allowlist.yaml")
        return

    conn = connect()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            total = 0
            for ent in entries:
                try:
                    print(f"Ingesting: {ent['url']} (type={ent.get('type')})")
                    added = ingest_entry(cur, ent)
                    total += added
                    conn.commit()
                    print(f"  -> {added} chunks")
                except Exception as e:
                    conn.rollback()
                    print(f"Failed: {ent['url']} ({e})")
                time.sleep(0.3)
        print(f"Done. Total chunks added this run: {total}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
