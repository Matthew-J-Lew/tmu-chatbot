from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

import asyncpg

from app.rag.embeddings import embed_query
from app.rag.reranker import rerank

from app.api.config import HYBRID_WEIGHT_TEXT, HYBRID_WEIGHT_VECTOR, RERANK_ENABLED
from app.api.program_registry import match_program
from app.api.retrieval_policy import RetrievalPolicy


async def retrieve_candidates(
    pool: asyncpg.Pool,
    query: str,
    k: int,
) -> List[Dict[str, Any]]:
    """Hybrid search in Postgres using vector + keyword search."""
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


_PROGRAM_SLUG_HINTS = {
    "Arts and Contemporary Studies": "arts_contemporary_studies",
    "Criminology": "criminology",
    "Economics and Finance": "economics_finance",
    "English": "english",
    "Environmental and Urban Sustainability": "environment_urban_sustainability",
    "Geographic Analysis": "geographic_analysis",
    "History": "history",
    "Language and Intercultural Relations": "language_intercultural_relations",
    "Philosophy": "philosophy",
    "Politics and Governance": "politics",
    "Psychology": "psychology",
    "Public Administration and Governance": "public_admin",
    "Sociology": "sociology",
    "Undeclared Arts": "undeclared_arts",
}


def _program_slug_hint(query: str) -> Optional[str]:
    program = match_program(query)
    if not program:
        return None
    return _PROGRAM_SLUG_HINTS.get(program)


def _is_arts_undergrad_program_list_query(query: str) -> bool:
    q = _normalize_query(query)
    if "graduate" in q:
        return False
    asks_undergrad_programs = "undergraduate program" in q or "undergraduate programs" in q
    asks_for_list = any(token in q for token in ("list", "every", "all", "which", "what are", "include each program name", "include every program name"))
    return asks_undergrad_programs and asks_for_list


def _is_course_planning_query(query: str) -> bool:
    q = _normalize_query(query)
    markers = (
        "what courses should",
        "what classes should",
        "what should i take",
        "first year",
        "second year",
        "third year",
        "fourth year",
        "curriculum tables",
        "table i",
        "table ii",
        "required group",
        "pick classes",
        "pick courses",
    )
    return any(marker in q for marker in markers)


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

    return bonus * (0.5 if use_rerank else 1.0)


def _course_planning_bonus(query: str, cand: Dict[str, Any], use_rerank: bool = False) -> float:
    if not _is_course_planning_query(query):
        return 0.0

    url = (cand.get("chunk_url") or cand.get("source_url") or "").lower()
    section = (cand.get("section") or "").lower()
    chunk = (cand.get("chunk") or "").lower()
    slug_hint = _program_slug_hint(query)

    bonus = 0.0
    if "/calendar/" in url and "/programs/arts/" in url:
        bonus += 3.0
    if slug_hint and f"/{slug_hint}" in url:
        bonus += 6.0
    if url.rstrip("/").endswith("/table_i") or url.rstrip("/").endswith("/table_ii"):
        bonus += 5.0
    if any(label in section for label in (
        "program overview/curriculum information",
        "full-time, four-year program",
        "full-time, five-year co-op program",
    )):
        bonus += 4.0
    if any(label in section for label in ("table i", "table ii", "required group", "core elective")):
        bonus += 3.0
    if "academic year:" in chunk and "program:" in chunk:
        bonus += 2.0
    if any(noisy in url for noisy in (
        "/arts/undergraduate/academic-support/",
        "/arts/undergraduate/new-students/",
        "/admissions/undergraduate/",
        "/student-financial-assistance/",
    )):
        bonus -= 3.0

    return bonus * (0.5 if use_rerank else 1.0)


def _apply_query_specific_boosts(query: str, candidates: List[Dict[str, Any]], use_rerank: bool = False) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    score_key = "rerank_score" if use_rerank else "hybrid_score"
    for cand in candidates:
        base = float(cand.get(score_key) or 0.0)
        bonus = _program_list_bonus(query, cand, use_rerank=use_rerank)
        bonus += _course_planning_bonus(query, cand, use_rerank=use_rerank)
        cand["query_bonus"] = bonus
        cand["adjusted_score"] = base + bonus

    candidates.sort(key=lambda x: x.get("adjusted_score", 0.0), reverse=True)
    return candidates


def _canonical_url(url: str) -> str:
    return (url or '').strip().lower().rstrip('/')


def _requested_study_year(query: str) -> Optional[str]:
    q = _normalize_query(query)
    for label in ("first year", "second year", "third year", "fourth year"):
        if label in q:
            return label
    return None


_YEAR_SEMESTER_MARKERS = {
    "first year": ("semesters one and two", "semester 1", "semester 2", "year 1"),
    "second year": ("semesters three and four", "semester 3", "semester 4", "year 2"),
    "third year": ("semesters five and six", "semester 5", "semester 6", "year 3"),
    "fourth year": ("semesters seven and eight", "semester 7", "semester 8", "year 4"),
}


def _matches_requested_year(chunk: str, section: str, requested_year: Optional[str]) -> bool:
    if not requested_year:
        return False
    haystack = f"{section}\n{chunk}".lower()
    if requested_year in haystack:
        return True
    return any(marker in haystack for marker in _YEAR_SEMESTER_MARKERS.get(requested_year, ()))


def _is_calendar_support_url(url: str) -> bool:
    return any(part in url for part in (
        '/career-coop-student-success/',
        '/myservicehub-support/',
        '/current-students/course-enrolment/',
        '/arts/undergraduate/academic-support/',
        '/admissions/undergraduate/',
        '/student-financial-assistance/',
    ))


def _policy_slug_aliases(policy: Optional[RetrievalPolicy]) -> tuple[str, ...]:
    if not policy:
        return ()
    aliases = tuple(dict.fromkeys(tuple(policy.program_slug_aliases or ()) + ((policy.program_slug,) if policy.program_slug else ())))
    return tuple(alias for alias in aliases if alias)




def _section_text(cand: Dict[str, Any]) -> str:
    return (cand.get('section') or '').strip().lower()


def _chunk_text(cand: Dict[str, Any]) -> str:
    return (cand.get('chunk') or '').strip().lower()


def _is_exact_program_calendar_url(url: str, program_slug: Optional[str], aliases: tuple[str, ...] = ()) -> bool:
    slug_candidates = tuple(dict.fromkeys(((program_slug,) if program_slug else ()) + tuple(aliases or ())))
    if not slug_candidates:
        return False
    return any(
        re.search(rf"/calendar/\d{{4}}-\d{{4}}/programs/arts/{re.escape(slug)}(?:/|$)", url) is not None
        for slug in slug_candidates
    )


def _is_sibling_program_calendar_url(url: str, program_slug: Optional[str], aliases: tuple[str, ...] = ()) -> bool:
    slug_candidates = tuple(dict.fromkeys(((program_slug,) if program_slug else ()) + tuple(aliases or ())))
    if not slug_candidates or not re.search(r"/calendar/\d{4}-\d{4}/programs/arts/", url):
        return False
    if _is_exact_program_calendar_url(url, program_slug, aliases):
        return False
    return any(
        f"/programs/arts/{slug}_" in url or f"_{slug}" in url
        for slug in slug_candidates
    )


def _extract_arts_calendar_program_slug(url: str) -> Optional[str]:
    m = re.search(r"/calendar/\d{4}-\d{4}/programs/arts/([^/]+)", url)
    if not m:
        return None
    return m.group(1)


def _is_other_arts_calendar_program_url(url: str, program_slug: Optional[str], aliases: tuple[str, ...] = ()) -> bool:
    slug_candidates = set(((program_slug,) if program_slug else ()) + tuple(aliases or ()))
    if not slug_candidates:
        return False
    other_slug = _extract_arts_calendar_program_slug(url)
    if not other_slug:
        return False
    return other_slug not in slug_candidates


def _table_bias(policy: RetrievalPolicy, url: str, section: str, chunk: str, scale: float) -> float:
    if policy.label not in {"COURSE_PLANNING_CALENDAR", "PROGRAM_REQUIREMENTS_CALENDAR"}:
        return 0.0

    bonus = 0.0
    slug_aliases = _policy_slug_aliases(policy)
    requested_year = _requested_study_year(policy.retrieval_query or "")
    exact_program = _is_exact_program_calendar_url(url, policy.program_slug, slug_aliases)
    sibling_program = _is_sibling_program_calendar_url(url, policy.program_slug, slug_aliases)
    other_program = _is_other_arts_calendar_program_url(url, policy.program_slug, slug_aliases)
    is_table_url = url.endswith('/table_i') or url.endswith('/table_ii')
    year_match = _matches_requested_year(chunk, section, requested_year)

    if exact_program:
        bonus += 8.0 * scale
    if sibling_program:
        bonus -= 20.0 * scale
    elif other_program:
        bonus -= 13.0 * scale

    if exact_program and is_table_url:
        bonus += 16.0 * scale
    elif is_table_url and other_program:
        bonus -= 8.0 * scale

    if exact_program and 'full-time, four-year program' in section:
        bonus += 7.0 * scale
        if year_match:
            bonus += 8.0 * scale
    if exact_program and 'full-time, five-year co-op program' in section:
        bonus -= 6.0 * scale if policy.label == 'COURSE_PLANNING_CALENDAR' else 1.5 * scale
    if exact_program and any(term in section for term in ('required group', 'core elective', 'table i', 'table ii')):
        bonus += 8.0 * scale
        if year_match:
            bonus += 6.0 * scale

    if policy.label == 'COURSE_PLANNING_CALENDAR':
        if exact_program and 'program overview/curriculum information' in section:
            bonus -= 8.0 * scale if requested_year else 5.0 * scale
        if 'academic year:' in chunk and 'program:' in chunk and exact_program:
            bonus += 2.0 * scale
        if exact_program and year_match:
            bonus += 10.0 * scale
        if requested_year and exact_program and not year_match and not is_table_url and 'full-time, four-year program' not in section:
            bonus -= 3.0 * scale
    else:
        if exact_program and 'program overview/curriculum information' in section:
            bonus += 1.0 * scale
        if exact_program and year_match:
            bonus += 5.0 * scale

    return bonus


def _apply_policy_preferences(candidates: List[Dict[str, Any]], policy: Optional[RetrievalPolicy], use_rerank: bool = False) -> List[Dict[str, Any]]:
    if not candidates or policy is None:
        return candidates

    scale = 0.5 if use_rerank else 1.0
    for cand in candidates:
        bonus = float(cand.get('query_bonus') or 0.0)
        url = _canonical_url(cand.get('chunk_url') or cand.get('source_url') or '')
        section = _section_text(cand)
        chunk = _chunk_text(cand)

        if policy.preferred_urls and any(pref.lower().rstrip('/') in url for pref in policy.preferred_urls):
            bonus += 4.0 * scale
        if policy.discouraged_urls and any(disc.lower().rstrip('/') in url for disc in policy.discouraged_urls):
            bonus -= 3.0 * scale

        if policy.preferred_section_terms and any(term.lower() in section for term in policy.preferred_section_terms):
            bonus += 2.5 * scale
        if policy.discouraged_section_terms and any(term.lower() in section for term in policy.discouraged_section_terms):
            bonus -= 2.0 * scale

        bonus += _table_bias(policy, url, section, chunk, scale)

        cand['query_bonus'] = bonus
        score_key = 'rerank_score' if use_rerank else 'hybrid_score'
        base = float(cand.get(score_key) or 0.0)
        cand['adjusted_score'] = base + bonus

    candidates.sort(key=lambda x: x.get('adjusted_score', 0.0), reverse=True)
    return candidates


def _suppress_other_program_pages(candidates: List[Dict[str, Any]], policy: Optional[RetrievalPolicy]) -> List[Dict[str, Any]]:
    if not candidates or policy is None:
        return candidates
    if policy.label not in {"COURSE_PLANNING_CALENDAR", "PROGRAM_REQUIREMENTS_CALENDAR"} or not policy.program_slug:
        return candidates

    slug_aliases = _policy_slug_aliases(policy)
    exact_family_count = 0
    exact_table_count = 0
    for cand in candidates:
        url = _canonical_url(cand.get('chunk_url') or cand.get('source_url') or '')
        if _is_exact_program_calendar_url(url, policy.program_slug, slug_aliases):
            exact_family_count += 1
            if url.endswith('/table_i') or url.endswith('/table_ii'):
                exact_table_count += 1

    if exact_family_count < 2 and not (exact_family_count >= 1 and exact_table_count >= 1):
        return candidates

    kept: List[Dict[str, Any]] = []
    for cand in candidates:
        url = _canonical_url(cand.get('chunk_url') or cand.get('source_url') or '')
        if _is_other_arts_calendar_program_url(url, policy.program_slug, slug_aliases):
            continue
        kept.append(cand)
    return kept


def _suppress_curriculum_support_pages(candidates: List[Dict[str, Any]], policy: Optional[RetrievalPolicy]) -> List[Dict[str, Any]]:
    if not candidates or policy is None:
        return candidates
    if policy.label not in {"COURSE_PLANNING_CALENDAR", "PROGRAM_REQUIREMENTS_CALENDAR"} or not policy.program_slug:
        return candidates

    slug_aliases = _policy_slug_aliases(policy)
    exact_program_count = 0
    exact_table_count = 0
    for cand in candidates:
        url = _canonical_url(cand.get('chunk_url') or cand.get('source_url') or '')
        if _is_exact_program_calendar_url(url, policy.program_slug, slug_aliases):
            exact_program_count += 1
            if url.endswith('/table_i') or url.endswith('/table_ii'):
                exact_table_count += 1

    if exact_program_count < 2 and not (exact_program_count >= 1 and exact_table_count >= 1):
        return candidates

    kept: List[Dict[str, Any]] = []
    for cand in candidates:
        url = _canonical_url(cand.get('chunk_url') or cand.get('source_url') or '')
        if _is_calendar_support_url(url):
            continue
        kept.append(cand)
    return kept




def _prefer_targeted_sources_for_support(candidates: List[Dict[str, Any]], policy: Optional[RetrievalPolicy]) -> List[Dict[str, Any]]:
    if not candidates or policy is None or not policy.preferred_urls:
        return candidates

    strict_labels = {
        "ADVISOR_CONTACT",
        "ACADEMIC_CONSIDERATION",
        "MISSED_ASSESSMENT",
        "ACADEMIC_ACCOMMODATIONS",
        "MENTAL_HEALTH_SUPPORT",
        "STUDENT_SUPPORT",
    }
    if policy.label not in strict_labels:
        return candidates

    targeted: List[Dict[str, Any]] = []
    for cand in candidates:
        url = _canonical_url(cand.get('chunk_url') or cand.get('source_url') or '')
        section = _section_text(cand)
        matches_url = any(pref.lower().rstrip('/') in url for pref in policy.preferred_urls)
        matches_section = bool(policy.preferred_section_terms) and any(term.lower() in section for term in policy.preferred_section_terms)
        discouraged = bool(policy.discouraged_urls) and any(disc.lower().rstrip('/') in url for disc in policy.discouraged_urls)
        if (matches_url or matches_section) and not discouraged:
            targeted.append(cand)

    return targeted if len(targeted) >= 2 else candidates

def _enforce_same_source_limit(candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return candidates
    kept: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for cand in candidates:
        key = _canonical_url(cand.get('source_url') or cand.get('chunk_url') or '')
        counts.setdefault(key, 0)
        if counts[key] >= limit:
            continue
        counts[key] += 1
        kept.append(cand)
    return kept


async def retrieve(
    pool: asyncpg.Pool,
    query: str,
    k: int = 4,
    num_candidates: int = 12,
    policy: Optional[RetrievalPolicy] = None,
) -> List[Dict[str, Any]]:
    if _is_course_planning_query(query) or (policy and policy.same_source_limit > 1):
        num_candidates = max(num_candidates, 24)

    candidates = await retrieve_candidates(pool, query, k=num_candidates)
    candidates = _apply_query_specific_boosts(query, candidates, use_rerank=False)
    candidates = _apply_policy_preferences(candidates, policy, use_rerank=False)
    candidates = _suppress_other_program_pages(candidates, policy)
    candidates = _suppress_curriculum_support_pages(candidates, policy)
    candidates = _prefer_targeted_sources_for_support(candidates, policy)

    if not RERANK_ENABLED:
        limited = _enforce_same_source_limit(candidates, policy.same_source_limit if policy else 1)
        return limited[:k]

    reranked = rerank(query, candidates, top_k=len(candidates))
    reranked = _apply_query_specific_boosts(query, reranked, use_rerank=True)
    reranked = _apply_policy_preferences(reranked, policy, use_rerank=True)
    reranked = _suppress_other_program_pages(reranked, policy)
    reranked = _suppress_curriculum_support_pages(reranked, policy)
    reranked = _prefer_targeted_sources_for_support(reranked, policy)
    limited = _enforce_same_source_limit(reranked, policy.same_source_limit if policy else 1)
    return limited[:k]
