import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from app.api.retrieval_policy import choose_retrieval_policy
from app.rag.retrieval import _apply_policy_preferences, _suppress_other_program_pages


def _cand(url: str, section: str, score: float = 1.0):
    return {
        "chunk_url": url,
        "source_url": url,
        "section": section,
        "chunk": "Academic year: 2025-2026\nProgram: Criminology\nUseful curriculum text",
        "hybrid_score": score,
        "rerank_score": score,
    }


def test_exact_program_tables_outrank_sibling_pages_for_course_planning():
    policy = choose_retrieval_policy(
        "What courses should I pick for Criminology first year?",
        "What courses should a first year student in the Criminology program take in TMU Faculty of Arts? Use the undergraduate calendar curriculum tables.",
    )
    candidates = [
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology_politics_governance/", "Program Overview/Curriculum Information", 2.0),
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/", "Program Overview/Curriculum Information", 2.0),
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/table_i/", "Criminology Required Group 1 - Table I", 2.0),
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/table_ii/", "Criminology Core Elective Table II", 2.0),
    ]

    ranked = _apply_policy_preferences(candidates, policy, use_rerank=False)
    urls = [c["chunk_url"] for c in ranked]
    assert urls[0].endswith('/table_i/')
    assert urls[-1].endswith('/criminology_politics_governance/')


def test_program_requirements_still_prefers_exact_program_over_sibling_pages():
    policy = choose_retrieval_policy(
        "What are my required courses for Criminology BA?",
        "What are the required courses, first-year requirements, and degree requirements for the Criminology program in TMU Faculty of Arts?",
    )
    candidates = [
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology_history/", "Program Overview/Curriculum Information", 2.0),
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/", "Full-Time, Four-Year Program", 2.0),
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/criminology/table_i/", "Criminology Required Group 1 - Table I", 2.0),
    ]

    ranked = _apply_policy_preferences(candidates, policy, use_rerank=False)
    urls = [c["chunk_url"] for c in ranked]
    assert urls[0].endswith('/table_i/') or urls[0].endswith('/criminology/')
    assert urls[-1].endswith('/criminology_history/')


def test_other_arts_calendar_program_pages_are_suppressed_when_exact_family_exists():
    policy = choose_retrieval_policy(
        "What courses should I pick for Psychology second year?",
        "What courses should a second year student in the Psychology program take in TMU Faculty of Arts? Prefer exact table rows and avoid combined-program or other-program Arts calendar pages unless absolutely necessary.",
    )
    candidates = [
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/psychology/", "Full-Time, Four-Year Program", 2.0),
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/psychology/table_i/", "Psychology Table I", 2.0),
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/history/", "Full-Time, Four-Year Program", 2.5),
        _cand("https://www.torontomu.ca/calendar/2025-2026/programs/arts/english/", "Full-Time, Four-Year Program", 2.4),
    ]

    ranked = _apply_policy_preferences(candidates, policy, use_rerank=False)
    filtered = _suppress_other_program_pages(ranked, policy)
    urls = [c["chunk_url"] for c in filtered]
    assert any(url.endswith('/psychology/') for url in urls)
    assert any(url.endswith('/psychology/table_i/') for url in urls)
    assert not any(url.endswith('/history/') for url in urls)
    assert not any(url.endswith('/english/') for url in urls)
