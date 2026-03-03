from __future__ import annotations

from typing import Any, Dict, List

import asyncpg

from app.rag.embeddings import embed_query
from app.rag.reranker import rerank

from app.api.config import HYBRID_WEIGHT_TEXT, HYBRID_WEIGHT_VECTOR, RERANK_ENABLED

# NOTE: RERANK_ENABLED and hybrid weights come from app.api.config so they can be
# centrally validated and controlled via .env / docker-compose.


async def retrieve_candidates(
    pool: asyncpg.Pool,
    query: str,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Hybrid search in Postgres:
      1) embed the query
      2) call rag_hybrid_search(query_text, query_embedding, k)
      3) return candidate chunks

    NOTE (important):
    - asyncpg does NOT automatically adapt Python List[float] into pgvector's 'vector' type.
    - pgvector accepts a text literal like: "[0.1, 0.2, ...]" and can cast it to vector.
    """

    q_emb = embed_query(query)

    # Convert Python list[float] -> pgvector text literal
    # Example: "[0.123, -0.456, ...]"
    q_emb_str = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

    rows = await pool.fetch(
        """
        SELECT
          id,
          chunk_url,
          source_url,
          section,
          chunk,
          vector_score,
          text_score,
          hybrid_score
        FROM rag_hybrid_search($1, $2::vector, $3, $4, $5)
        """,
        query,
        q_emb_str,  # <- IMPORTANT: string, not list
        k,
        HYBRID_WEIGHT_VECTOR,
        HYBRID_WEIGHT_TEXT,
    )

    return [dict(r) for r in rows]



async def retrieve(
    pool: asyncpg.Pool,
    query: str,
    k: int = 4,
    num_candidates: int = 12,
) -> List[Dict[str, Any]]:
    """
    Main retrieval entrypoint for the API.

    Production goals:
    - Keep this fast and stable.
    - Reduce reranker load (num_candidates defaults lower than dev).
    - Allow reranking to be disabled via env var.
    """
    candidates = await retrieve_candidates(pool, query, k=num_candidates)

    if not RERANK_ENABLED:
        return candidates[:k]

    # Cross-encoder rerank
    return rerank(query, candidates, top_k=k)
