from __future__ import annotations

from typing import Any, Dict, List, Sequence
import re

import asyncpg

from app.rag.embeddings import embed_query
from app.rag.reranker import rerank

from app.api.config import HYBRID_WEIGHT_TEXT, HYBRID_WEIGHT_VECTOR, RERANK_ENABLED


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
        q_emb_str,
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



def _is_arts_grad_program_list_query(query: str) -> bool:
    q = _normalize_query(query)
    asks_grad_programs = "graduate program" in q or "graduate programs" in q
    asks_for_list = any(token in q for token in ("list", "every", "all", "which", "what are", "include each program name", "include every program name"))
    return asks_grad_programs and asks_for_list



def _is_program_list_query(query: str) -> bool:
    return _is_arts_undergrad_program_list_query(query) or _is_arts_grad_program_list_query(query)



def _count_numbered_items(text: str) -> int:
    return len(re.findall(r"(?:^|\n)\s*\d+\.\s", text))



def _count_program_name_markers(text: str) -> int:
    return len(re.findall(r"\b[A-Z][A-Za-z&,'()\-/ ]+\s-\sB[A-Za-z ]+\b", text))



def _is_answer_bearing_program_list_chunk(cand: Dict[str, Any]) -> bool:
    chunk = cand.get("chunk") or ""
    return _count_numbered_items(chunk) >= 6 or _count_program_name_markers(chunk) >= 6



def _is_count_only_program_chunk(cand: Dict[str, Any]) -> bool:
    chunk = (cand.get("chunk") or "").lower()
    if _is_answer_bearing_program_list_chunk(cand):
        return False
    return "undergraduate programs" in chunk or "graduate programs" in chunk



def _preferred_program_list_url(query: str) -> str | None:
    if _is_arts_undergrad_program_list_query(query):
        return "/arts/undergraduate/programs"
    if _is_arts_grad_program_list_query(query):
        return "/arts/graduate/graduate-programs"
    return None



def _program_list_bonus(query: str, cand: Dict[str, Any], use_rerank: bool = False) -> float:
    preferred_url = _preferred_program_list_url(query)
    if not preferred_url:
        return 0.0

    url = (cand.get("chunk_url") or cand.get("source_url") or "").lower()
    source_url = (cand.get("source_url") or "").lower()
    section = (cand.get("section") or "").lower()
    chunk = (cand.get("chunk") or "").lower()

    bonus = 0.0
    is_preferred_page = preferred_url in url or preferred_url in source_url
    if is_preferred_page:
        bonus += 4.0
    if section in {"explore program options", "undergraduate programs", "graduate programs"}:
        bonus += 1.5
    if is_preferred_page and _is_answer_bearing_program_list_chunk(cand):
        bonus += 8.0
    if is_preferred_page and _is_count_only_program_chunk(cand):
        bonus += 2.0

    # Prefer chunks whose sections or text mention Faculty of Arts explicitly.
    if "faculty of arts" in chunk or "faculty of arts" in section:
        bonus += 0.5

    # Penalize unrelated list pages that happen to contain many numbered items.
    if any(noisy in url for noisy in (
        "/student-financial-assistance/",
        "/admissions/undergraduate/requirements/",
        "/admissions/undergraduate/apply/document-submission/faq/",
    )):
        bonus -= 3.0
    if "/curriculum-advising/" in url and not is_preferred_page:
        bonus -= 2.0
    if not is_preferred_page and _count_numbered_items(chunk) >= 3:
        bonus -= 2.5

    return bonus * (0.6 if use_rerank else 1.0)



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



def _pick_diverse_chunks(candidates: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_ids = set()
    for cand in candidates:
        cid = cand.get("id")
        if cid in seen_ids:
            continue
        selected.append(cand)
        seen_ids.add(cid)
        if len(selected) >= k:
            break
    return selected



def _pick_program_list_chunks(query: str, candidates: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    preferred_url = _preferred_program_list_url(query)
    if not preferred_url:
        return _pick_diverse_chunks(candidates, k)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    per_source: dict[str, int] = {}
    max_from_same_source = 3

    def add_candidate(cand: Dict[str, Any]) -> bool:
        cid = cand.get("id")
        if cid in selected_ids:
            return False
        source = cand.get("source_url") or cand.get("chunk_url") or ""
        if per_source.get(source, 0) >= max_from_same_source:
            return False
        selected.append(cand)
        selected_ids.add(cid)
        per_source[source] = per_source.get(source, 0) + 1
        return True

    preferred_candidates = [
        cand for cand in candidates
        if preferred_url in ((cand.get("source_url") or cand.get("chunk_url") or "").lower())
    ]

    # First, lock in the answer-bearing list chunk from the preferred page.
    for cand in preferred_candidates:
        if _is_answer_bearing_program_list_chunk(cand):
            add_candidate(cand)
            break

    # Then add up to two more supporting chunks from the same page (count/overview + strong page context).
    for cand in preferred_candidates:
        if len(selected) >= min(k, max_from_same_source):
            break
        add_candidate(cand)

    # Fill the rest using the overall reranked list, still allowing a small amount of same-source clustering.
    for cand in candidates:
        if len(selected) >= k:
            break
        add_candidate(cand)

    return selected


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
    - For list/composite questions, allow a small number of chunks from the same
      official source page so the model can see the answer-bearing list and its context.
    """
    candidates = await retrieve_candidates(pool, query, k=num_candidates)
    candidates = _apply_query_specific_boosts(query, candidates, use_rerank=False)

    if not RERANK_ENABLED:
        if _is_program_list_query(query):
            return _pick_program_list_chunks(query, candidates, k)
        return _pick_diverse_chunks(candidates, k)

    reranked = rerank(query, candidates, top_k=len(candidates))
    reranked = _apply_query_specific_boosts(query, reranked, use_rerank=True)
    if _is_program_list_query(query):
        return _pick_program_list_chunks(query, reranked, k)
    return _pick_diverse_chunks(reranked, k)
