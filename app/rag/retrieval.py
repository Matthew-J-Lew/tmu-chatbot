from __future__ import annotations

from typing import Any, Dict, List
import re

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



def _normalize_query(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_arts_undergrad_program_list_query(query: str) -> bool:
    q = _normalize_query(query)
    if "graduate" in q:
        return False
    asks_undergrad_programs = "undergraduate program" in q or "undergraduate programs" in q
    asks_for_list = any(token in q for token in ("list", "every", "all", "which", "what are", "include each program name", "include every program name"))
    return asks_undergrad_programs and asks_for_list


def _program_list_bonus(query: str, cand: Dict[str, Any], use_rerank: bool = False) -> float:
    if not _is_arts_undergrad_program_list_query(query):
        return 0.0

    url = (cand.get("chunk_url") or cand.get("source_url") or "").lower()
    section = (cand.get("section") or "").lower()
    chunk = (cand.get("chunk") or "").lower()

    bonus = 0.0
    if "/arts/undergraduate/programs" in url:
        bonus += 6.0
    if section in {"explore program options", "undergraduate programs"}:
        bonus += 2.5
    if "faculty of arts" in chunk and "undergraduate program" in chunk:
        bonus += 1.0
    if len(re.findall(r"\b\d+\.\s", chunk)) >= 8 and "/arts/undergraduate/programs" in url:
        bonus += 4.0

    # Penalize unrelated list pages that now share similar wording.
    if "official program list" in chunk and "/arts/undergraduate/programs" not in url:
        bonus -= 5.0
    if any(noisy in url for noisy in (
        "/student-financial-assistance/",
        "/admissions/undergraduate/requirements/",
        "/admissions/undergraduate/apply/document-submission/faq/",
    )):
        bonus -= 2.5
    if "/curriculum-advising/" in url and "/arts/undergraduate/programs" not in url:
        bonus -= 1.5

    # After rerank, keep the heuristic effect a bit smaller but still decisive.
    return bonus * (0.5 if use_rerank else 1.0)


def _apply_query_specific_boosts(query: str, candidates: List[Dict[str, Any]], use_rerank: bool = False) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    score_key = "rerank_score" if use_rerank else "hybrid_score"
    for cand in candidates:
        base = float(cand.get(score_key) or 0.0)
        bonus = _program_list_bonus(query, cand, use_rerank=use_rerank)
        cand["query_bonus"] = bonus
        cand["adjusted_score"] = base + bonus

    candidates.sort(key=lambda x: x.get("adjusted_score", 0.0), reverse=True)
    return candidates


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
    candidates = _apply_query_specific_boosts(query, candidates, use_rerank=False)

    if not RERANK_ENABLED:
        return candidates[:k]

    # Cross-encoder rerank
    reranked = rerank(query, candidates, top_k=len(candidates))
    reranked = _apply_query_specific_boosts(query, reranked, use_rerank=True)
    return reranked[:k]
