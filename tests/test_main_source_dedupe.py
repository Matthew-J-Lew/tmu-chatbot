import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lightweight stubs so importing app.api.main does not require runtime services.
asyncpg_mod = types.ModuleType("asyncpg")
asyncpg_mod.Pool = object
sys.modules.setdefault("asyncpg", asyncpg_mod)

db_mod = types.ModuleType("app.api.db")
async def _noop_async(*args, **kwargs):
    return None
db_mod.init_db_pool = _noop_async
db_mod.close_db_pool = _noop_async
db_mod.get_pool = lambda: None
sys.modules.setdefault("app.api.db", db_mod)

cache_mod = types.ModuleType("app.api.cache")
cache_mod.init_redis = _noop_async
cache_mod.close_redis = _noop_async
cache_mod.make_cache_key = lambda *args, **kwargs: ""
cache_mod.get_redis = lambda: None
async def _none(*args, **kwargs):
    return None
cache_mod.cache_get_json = _none
cache_mod.cache_set_json = _noop_async
sys.modules.setdefault("app.api.cache", cache_mod)

llm_mod = types.ModuleType("app.api.llm_client")
llm_mod.init_llm_client = _noop_async
llm_mod.close_llm_client = _noop_async
llm_mod.generate = _none
async def _gen_stream(*args, **kwargs):
    if False:
        yield ""
llm_mod.generate_stream = _gen_stream
sys.modules.setdefault("app.api.llm_client", llm_mod)

# Retrieval import path also needs these minimal model stubs.
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

from app.api.main import (
    _postprocess_answer,
    _remap_answer_citations,
    build_messages_and_sources,
    build_response_cache_identity,
    build_retrieval_cache_identity,
)
from app.api.retrieval_policy import choose_retrieval_policy


def test_remap_answer_citations_dedupes_sources_by_canonical_page_and_renumbers():
    _, source_lookup = build_messages_and_sources(
        "q",
        [
            {
                "chunk_url": "https://example.com/a",
                "source_url": "https://example.com/a",
                "section": "Program Overview/Curriculum Information",
                "chunk": "Chunk one",
            },
            {
                "chunk_url": "https://example.com/a",
                "source_url": "https://example.com/a",
                "section": "Table I",
                "chunk": "Chunk two",
            },
            {
                "chunk_url": "https://example.com/b",
                "source_url": "https://example.com/b",
                "section": "Table II",
                "chunk": "Chunk three",
            },
        ],
    )

    rewritten, used_sources = _remap_answer_citations(
        "Overview [2]. More detail [1][3].",
        source_lookup,
    )

    assert rewritten == "Overview [1]. More detail [1][2]."
    assert [s.url for s in used_sources] == ["https://example.com/a", "https://example.com/b"]
    assert [s.id for s in used_sources] == [1, 2]


def test_response_cache_identity_uses_concrete_question_for_same_policy_family():
    q1 = "How do I enroll in a class?"
    q2 = "How do I enroll in classes?"
    p1 = choose_retrieval_policy(q1, q1)
    p2 = choose_retrieval_policy(q2, q2)

    assert p1.label == p2.label == "COURSE_ENROLMENT"
    assert build_response_cache_identity(q1, q1, p1) != build_response_cache_identity(q2, q2, p2)


def test_retrieval_cache_identity_uses_concrete_question_for_same_policy_family():
    q1 = "How do I add a class?"
    q2 = "How do I add, drop, or swap classes?"
    p1 = choose_retrieval_policy(q1, q1)
    p2 = choose_retrieval_policy(q2, q2)

    assert p1.label == p2.label == "COURSE_MANAGEMENT"
    assert build_retrieval_cache_identity(q1, p1) != build_retrieval_cache_identity(q2, p2)


def test_remap_answer_citations_shows_only_cited_sources_and_drops_unused_ones():
    _, source_lookup = build_messages_and_sources(
        "q",
        [
            {
                "chunk_url": f"https://example.com/{i}",
                "source_url": f"https://example.com/{i}",
                "section": f"Section {i}",
                "chunk": f"Chunk {i}",
            }
            for i in range(1, 6)
        ],
    )

    rewritten, used_sources = _remap_answer_citations("Use these [4][2].", source_lookup)

    assert rewritten == "Use these [1][2]."
    assert [s.url for s in used_sources] == ["https://example.com/4", "https://example.com/2"]


def test_postprocess_curriculum_answer_removes_summary_notes_and_references_sections():
    answer = """## 1st & 2nd Semester Requirements
- Item A [1]

## 3rd & 4th Semester Requirements
- Item B [2]

## Summary of Degree Requirements
- Repeats everything [1][2]

## Notes on Table I and Table II
- Extra notes [3]

**References:**
- Source A
- Source B
"""

    class Policy:
        label = "PROGRAM_REQUIREMENTS_CALENDAR"

    cleaned = _postprocess_answer(answer, "What are my required courses?", policy=Policy())
    assert "Summary of Degree Requirements" not in cleaned
    assert "Notes on Table I and Table II" not in cleaned
    assert "**References:**" not in cleaned
    assert "1st & 2nd Semester Requirements" in cleaned
    assert "3rd & 4th Semester Requirements" in cleaned
