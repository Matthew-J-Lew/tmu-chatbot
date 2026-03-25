from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import re

import asyncpg

from app.rag.embeddings import embed_query
from app.rag.reranker import rerank

from app.api.config import HYBRID_WEIGHT_TEXT, HYBRID_WEIGHT_VECTOR, RERANK_ENABLED
from app.api.retrieval_policy import RetrievalPolicy


async def retrieve_candidates(
    pool: asyncpg.Pool,
    query: str,
    k: int,
) -> List[Dict[str, Any]]:
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
    if "faculty of arts" in chunk or "faculty of arts" in section:
        bonus += 0.5
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



def _policy_bonus(policy: Optional[RetrievalPolicy], cand: Dict[str, Any], use_rerank: bool = False) -> float:
    if not policy or policy.label == "DEFAULT":
        return 0.0

    url = (cand.get("chunk_url") or cand.get("source_url") or "").lower()
    source_url = (cand.get("source_url") or "").lower()
    section = (cand.get("section") or "").lower()
    chunk = (cand.get("chunk") or "").lower()

    bonus = 0.0
    if any(fragment in url or fragment in source_url for fragment in policy.preferred_urls):
        bonus += 4.0
    if any(fragment in url or fragment in source_url for fragment in policy.discouraged_urls):
        bonus -= 3.0

    if policy.label == "ARTS_UNDERGRAD_PROGRAM_LIST":
        if _is_answer_bearing_program_list_chunk(cand):
            bonus += 8.0
        if "undergraduate programs" in chunk and _count_numbered_items(chunk) >= 6:
            bonus += 4.0
        if section in {"explore program options", "undergraduate programs"}:
            bonus += 1.5
    elif policy.label == "ARTS_GRAD_PROGRAM_LIST":
        if _count_numbered_items(cand.get("chunk") or "") >= 3:
            bonus += 4.0
        if section in {"graduate programs", "explore graduate programs"}:
            bonus += 1.5
    elif policy.label == "ARTS_DEPARTMENTS_LIST":
        if "department" in section or "department" in chunk:
            bonus += 2.5
        if _count_numbered_items(cand.get("chunk") or "") >= 3:
            bonus += 2.0
    elif policy.label == "MINOR_DECLARATION":
        if any(term in chunk for term in ("myservicehub", "application to graduate", "apply for the minor", "select your minor", "minor")):
            bonus += 3.5
        if any(term in section for term in ("program requirements", "undergraduate program requirements", "curriculum")):
            bonus += 2.0
    elif policy.label == "COURSE_ENROLMENT":
        if any(term in chunk for term in ("course intentions", "priority enrolment", "myservicehub", "new students", "continuing students")):
            bonus += 3.0
    elif policy.label == "COURSE_WAITLIST":
        if any(term in chunk for term in ("wait list", "waitlist", "full class", "full course")):
            bonus += 3.5
    elif policy.label == "COURSE_MANAGEMENT":
        if any(term in chunk for term in ("add", "drop", "swap", "withdraw", "myservicehub")):
            bonus += 2.5
    elif policy.label == "PROGRAM_CHANGE":
        if any(term in chunk for term in ("change programs", "switch programs", "internal transfer", "transfer")):
            bonus += 2.5
    elif policy.label == "ADVISOR_CONTACT":
        if any(term in chunk for term in ("academic advising", "advisor", "department directory", "contact")):
            bonus += 3.0
    elif policy.label in {"COURSE_PLANNING", "GRADUATION_PROGRESS"}:
        if any(term in chunk for term in ("advisement report", "academic advising", "degree progress", "program requirements")):
            bonus += 3.0
    elif policy.label == "ACADEMIC_CONSIDERATION":
        if any(term in chunk for term in ("academic consideration", "missed test", "missed exam", "request")):
            bonus += 3.5
    elif policy.label == "ACADEMIC_STANDING":
        if any(term in chunk for term in ("academic standing", "academic probation", "grades and standings", "failed a course", "fail a course")):
            bonus += 3.0
    elif policy.label == "IMPORTANT_DATES":
        if any(term in chunk for term in ("important dates", "significant dates", "drop dates", "deadline")):
            bonus += 2.5
    elif policy.label == "STUDENT_SUPPORT":
        if any(term in chunk for term in ("student support", "mental health", "counselling", "academic support", "wellbeing")):
            bonus += 2.5

    return bonus * (0.6 if use_rerank else 1.0)


def _apply_query_specific_boosts(query: str, candidates: List[Dict[str, Any]], use_rerank: bool = False, policy: Optional[RetrievalPolicy] = None) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    score_key = "rerank_score" if use_rerank else "hybrid_score"
    for cand in candidates:
        base = float(cand.get(score_key) or 0.0)
        bonus = _program_list_bonus(query, cand, use_rerank=use_rerank) + _policy_bonus(policy, cand, use_rerank=use_rerank)
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



def _pick_with_policy(query: str, candidates: Sequence[Dict[str, Any]], k: int, policy: Optional[RetrievalPolicy]) -> List[Dict[str, Any]]:
    if not policy or policy.label == "DEFAULT":
        if _is_program_list_query(query):
            policy = RetrievalPolicy(label="PROGRAM_LIST", preferred_urls=tuple(filter(None, (_preferred_program_list_url(query),))), same_source_limit=3)
        else:
            return _pick_diverse_chunks(candidates, k)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    per_source: dict[str, int] = {}
    max_from_same_source = max(1, policy.same_source_limit)

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
        if any(fragment in ((cand.get("source_url") or cand.get("chunk_url") or "").lower()) for fragment in policy.preferred_urls)
    ] if policy.preferred_urls else []

    if policy.label in {"ARTS_UNDERGRAD_PROGRAM_LIST", "PROGRAM_LIST"}:
        for cand in preferred_candidates:
            if _is_answer_bearing_program_list_chunk(cand):
                add_candidate(cand)
                break

    for cand in preferred_candidates:
        if len(selected) >= min(k, max_from_same_source):
            break
        add_candidate(cand)

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
    policy: Optional[RetrievalPolicy] = None,
) -> List[Dict[str, Any]]:
    candidates = await retrieve_candidates(pool, query, k=num_candidates)
    candidates = _apply_query_specific_boosts(query, candidates, use_rerank=False, policy=policy)

    if not RERANK_ENABLED:
        return _pick_with_policy(query, candidates, k, policy)

    reranked = rerank(query, candidates, top_k=len(candidates))
    reranked = _apply_query_specific_boosts(query, reranked, use_rerank=True, policy=policy)
    return _pick_with_policy(query, reranked, k, policy)
