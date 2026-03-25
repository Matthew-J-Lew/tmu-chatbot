import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import importlib
import sys
import types


redis_mod = types.ModuleType("redis")
redis_asyncio_mod = types.ModuleType("redis.asyncio")


class DummyRedis:
    pass


redis_asyncio_mod.Redis = DummyRedis
redis_mod.asyncio = redis_asyncio_mod
sys.modules.setdefault("redis", redis_mod)
sys.modules.setdefault("redis.asyncio", redis_asyncio_mod)


session_store = importlib.import_module("app.api.session_store")
turn_prep = importlib.import_module("app.api.turn_prep")

SessionState = session_store.SessionState
prepare_turn = turn_prep.prepare_turn


def test_program_slot_fill_flow_asks_then_answers():
    state = SessionState(session_id="abc")

    first = prepare_turn("abc", "What required classes do I need for my major?", state)
    assert first.workflow_reply == turn_prep._PROGRAM_PROMPT
    assert first.state_after.pending_slot == "program"
    assert first.state_after.pending_intent == "PROGRAM_REQUIREMENTS"

    second = prepare_turn("abc", "Criminology BA", first.state_after)
    assert second.workflow_reply is None
    assert "Criminology" in second.effective_question
    assert second.state_after.program == "Criminology"


def test_course_planning_asks_for_program_and_year():
    state = SessionState(session_id="abc")
    result = prepare_turn("abc", "What courses should I pick?", state)

    assert result.workflow_reply == turn_prep._PROGRAM_YEAR_PROMPT
    assert result.state_after.pending_slot == "program_year"
    assert result.state_after.pending_intent == "COURSE_PLANNING"


def test_course_planning_resume_with_program_and_year():
    state = SessionState(session_id="abc", pending_slot="program_year", pending_intent="COURSE_PLANNING")
    result = prepare_turn("abc", "Psychology, second year", state)

    assert result.workflow_reply is None
    assert "Psychology" in result.effective_question
    assert "second year" in result.effective_question
    assert result.state_after.program == "Psychology"
    assert result.state_after.study_year == "second year"


def test_enrolment_question_is_supported_not_rejected():
    state = SessionState(session_id="abc")
    result = prepare_turn("abc", "How do I enroll in classes?", state)

    assert result.workflow_reply is None
    assert "enroll in classes" in result.effective_question.lower() or "enrol" in result.effective_question.lower()
    assert result.state_after.last_intent == "COURSE_ENROLMENT"


def test_departments_question_is_supported_not_rejected():
    state = SessionState(session_id="abc")
    result = prepare_turn("abc", "What departments are in the Faculty of Arts?", state)

    assert result.workflow_reply is None
    assert "departments" in result.effective_question.lower()
    assert result.state_after.last_intent == "DEPARTMENTS_LIST"


def test_academic_consideration_question_is_supported():
    state = SessionState(session_id="abc")
    result = prepare_turn("abc", "What is an academic consideration request?", state)

    assert result.workflow_reply is None
    assert "academic consideration" in result.effective_question.lower()
    assert result.state_after.last_intent == "ACADEMIC_CONSIDERATION"


def test_social_turns_bypass_rag_and_preserve_context():
    state = SessionState(
        session_id="abc",
        program="Psychology",
        turn_count=4,
        metadata={"last_program_turn": 4},
        last_effective_question="What are the required courses for the Psychology program in TMU Faculty of Arts?",
        active_topic="What are the required courses for the Psychology program in TMU Faculty of Arts?",
    )
    result = prepare_turn("abc", "Okay thanks!", state)

    assert result.workflow_reply == "You’re welcome — let me know if you have any other TMU Faculty of Arts questions."
    assert result.state_after.last_effective_question == state.last_effective_question
    assert result.state_after.active_topic == state.active_topic


def test_program_correction_replays_recent_program_task():
    state = SessionState(
        session_id="abc",
        program="Criminology",
        turn_count=5,
        metadata={"last_program_turn": 5},
        last_effective_question="Is there a co-op option for the Criminology program in TMU Faculty of Arts? Include timing or eligibility details if available.",
        last_intent="PROGRAM_COOP",
        active_topic="Is there a co-op option for the Criminology program in TMU Faculty of Arts? Include timing or eligibility details if available.",
    )
    result = prepare_turn("abc", "But im in psychology", state)

    assert result.workflow_reply is None
    assert "Psychology" in result.effective_question
    assert result.state_after.program == "Psychology"
    assert result.state_after.last_intent == "PROGRAM_COOP"


def test_unknown_people_question_falls_back_instead_of_rag():
    state = SessionState(session_id="abc")
    result = prepare_turn("abc", "Who is Dr. Valerie Deacon?", state)

    assert result.workflow_reply == turn_prep._FALLBACK_REPLY
    assert result.state_after.last_effective_question is None


def test_follow_up_with_new_program_can_reuse_last_program_intent():
    state = SessionState(
        session_id="abc",
        program="English",
        turn_count=4,
        metadata={"last_program_turn": 3},
        last_effective_question="What are the required courses, first-year requirements, and degree requirements for the English program in TMU Faculty of Arts? Use the undergraduate calendar when possible.",
        last_intent="PROGRAM_REQUIREMENTS",
    )
    result = prepare_turn("abc", "What about for psychology?", state)

    assert result.workflow_reply is None
    assert "Psychology" in result.effective_question
    assert result.state_after.program == "Psychology"


def test_turn_prep_logger_is_stdout_backed():
    assert turn_prep.logger.level == turn_prep.logging.INFO
    assert turn_prep.logger.propagate is False
    assert any(getattr(h, "_tmu_turn_prep_handler", False) for h in turn_prep.logger.handlers)
