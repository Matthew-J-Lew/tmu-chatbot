# app/rag/retrieval.py (replace the DEFAULT_DSN + _get_dsn + get_db_connection)

import os
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

from .embeddings import embed_query
from .reranker import rerank


def _build_dsn_from_env() -> str:
    """
    Construct a Postgres DSN from the standard PG* environment variables
    that are already set in docker-compose.yml for the api service.
    """
    host = os.getenv("PGHOST", "pg")
    port = os.getenv("PGPORT", "5432")
    db   = os.getenv("PGDATABASE", "ragdb")
    user = os.getenv("PGUSER", "rag")
    pwd  = os.getenv("PGPASSWORD", "rag")

    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def get_db_connection():
    """
    Open a new Postgres connection.

    In production we may need to use a connection pool, but a simple connection per call is fine for dev.
    """
    dsn = os.getenv("DATABASE_URL") or _build_dsn_from_env()
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    return conn


def retrieve_candidates(query: str, k: int = 30) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval (vector + keyword) using the rag_hybrid_search SQL helper.

    Args:
        query: The user's natural language question.
        k: Number of candidates to return from Postgres (before reranking).

    Returns:
        A list of dicts, each with at least:
            {
              "id": int,
              "chunk_url": str,
              "source_url": str,
              "section": str | None,
              "chunk": str,
              "vector_score": float,
              "text_score": float,
              "hybrid_score": float,
            }
    """
    # 1) Embed the user query
    query_embedding = embed_query(query)  # list[float] length 384

    # 2) Call rag_hybrid_search() in Postgres
    sql = """
        SELECT
          id,
          chunk_url,
          source_url,
          section,
          chunk,
          vector_score,
          text_score,
          hybrid_score
        FROM rag_hybrid_search(%s, %s::vector, %s);
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query, query_embedding, k))
            rows = cur.fetchall()

    return list(rows)


def retrieve(
    query: str,
    k: int = 8,
    num_candidates: int = 30,
) -> List[Dict[str, Any]]:
    """
    Main retrieval entrypoint for the chatbot.

    Args:
        query: The user's question.
        k: Final number of chunks to return after reranking.
        num_candidates: Number of hybrid-search candidates to pull
                        from Postgres before reranking.

    Returns:
        A list of chunk dicts (top-k after reranking), each including:
            - all the fields from retrieve_candidates()
            - "rerank_score": float
    """
    # Step 1: hybrid retrieval from Postgres
    candidates = retrieve_candidates(query, k=num_candidates)

    # Step 2: cross-encoder reranking in Python
    top_chunks = rerank(query, candidates, top_k=k)

    return top_chunks
