import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Minimal stubs so importing app.rag.retrieval does not require runtime deps.
asyncpg_mod = types.ModuleType("asyncpg")
class DummyPool: ...
asyncpg_mod.Pool = DummyPool
sys.modules.setdefault("asyncpg", asyncpg_mod)

st_mod = types.ModuleType("sentence_transformers")
class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, texts, **kwargs):
        import numpy as np
        return np.zeros((len(texts), 384), dtype="float32")

class DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass
    def predict(self, pairs, **kwargs):
        return [0.0 for _ in pairs]

st_mod.SentenceTransformer = DummySentenceTransformer
st_mod.CrossEncoder = DummyCrossEncoder
sys.modules.setdefault("sentence_transformers", st_mod)

from app.api.retrieval_policy import RetrievalPolicy
from app.rag.retrieval import _apply_policy_preferences, _apply_query_specific_boosts


def test_course_planning_query_boosts_calendar_program_and_table_chunks():
    query = "What courses should a first year student in the Criminology program take in TMU Faculty of Arts? Start with the exact Criminology undergraduate calendar page's Full-Time, Four-Year Program and the semester block for 1st & 2nd Semester."
    candidates = [
        {
            "chunk_url": "https://www.torontomu.ca/arts/undergraduate/academic-support/",
            "source_url": "https://www.torontomu.ca/arts/undergraduate/academic-support/",
            "section": "TMU systems and platforms",
            "chunk": "General academic support information.",
            "hybrid_score": 0.82,
        },
        {
            "chunk_url": "https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/",
            "source_url": "https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/",
            "section": "Criminology - Full-Time, Four-Year Program",
            "chunk": "Academic year: 2025-2026\nProgram: Criminology\nSection context: Criminology - Full-Time, Four-Year Program\n\nRequired courses and structure.",
            "hybrid_score": 0.75,
        },
        {
            "chunk_url": "https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/table_i/",
            "source_url": "https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/table_i/",
            "section": "Criminology Required Group 1 - Table I - Humanities",
            "chunk": "Academic year: 2025-2026\nProgram: Criminology\nSection context: Criminology Required Group 1 - Table I - Humanities\n\nChoose one humanities course.",
            "hybrid_score": 0.71,
        },
    ]

    ranked = _apply_query_specific_boosts(query, candidates, use_rerank=False)

    assert ranked[0]["chunk_url"].endswith("/criminology/")
    assert ranked[1]["chunk_url"].endswith("/table_i/")
    assert ranked[-1]["chunk_url"].endswith("/academic-support/")


def test_policy_preferences_penalize_sibling_program_pages_for_specific_program():
    policy = RetrievalPolicy(
        label="COURSE_PLANNING_CALENDAR",
        program_slug="criminology",
        preferred_urls=("/calendar/2025-2026/programs/arts/criminology",),
        preferred_section_terms=("full-time, four-year program", "table i"),
    )
    candidates = [
        {
            "chunk_url": "https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/",
            "source_url": "https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/",
            "section": "Criminology - Full-Time, Four-Year Program",
            "chunk": "Required courses.",
            "hybrid_score": 0.70,
        },
        {
            "chunk_url": "https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology_history/",
            "source_url": "https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology_history/",
            "section": "Criminology and History - Full-Time, Four-Year Program",
            "chunk": "Combined major structure.",
            "hybrid_score": 0.74,
        },
    ]

    ranked = _apply_policy_preferences(candidates, policy, use_rerank=False)
    assert ranked[0]["chunk_url"].endswith("/criminology/")
