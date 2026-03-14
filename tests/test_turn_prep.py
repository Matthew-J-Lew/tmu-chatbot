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


def test_program_scoped_followup_uses_known_program():
    state = SessionState(
        session_id="abc",
        program="English",
        turn_count=3,
        metadata={"last_program_turn": 3},
    )
    result = prepare_turn("abc", "Is there co-op?", state)

    assert result.workflow_reply is None
    assert "English" in result.effective_question
    assert "co-op" in result.effective_question.lower() or "co op" in result.effective_question.lower()


def test_global_program_list_does_not_force_prior_program_context():
    state = SessionState(
        session_id="abc",
        program="Psychology",
        turn_count=3,
        metadata={"last_program_turn": 3},
    )
    result = prepare_turn("abc", "Can you list all undergraduate programs?", state)

    assert result.workflow_reply is None
    assert result.effective_question == "Can you list all undergraduate programs?"


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

    assert result.workflow_reply == "You’re welcome — let me know if you have any other Faculty of Arts questions."
    assert result.state_after.last_effective_question == state.last_effective_question
    assert result.state_after.active_topic == state.active_topic


def test_capability_question_uses_canned_reply():
    state = SessionState(session_id="abc")
    result = prepare_turn("abc", "what do you do", state)

    assert result.workflow_reply is not None
    assert "undergraduate programs" in result.workflow_reply.lower()
    assert result.state_after.last_effective_question is None


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


def test_stale_program_is_not_reused_for_my_major_question():
    state = SessionState(
        session_id="abc",
        program="Criminology",
        turn_count=20,
        metadata={"last_program_turn": 2},
    )
    result = prepare_turn("abc", "What are the required courses for my major?", state)

    assert result.workflow_reply == turn_prep._PROGRAM_PROMPT
    assert result.state_after.pending_slot == "program"


def test_referential_program_followup_resolves_it_to_program():
    state = SessionState(
        session_id="abc",
        program="Arts and Contemporary Studies",
        turn_count=2,
        metadata={"last_program_turn": 2},
        last_effective_question="Provide an overview of the Arts and Contemporary Studies program in TMU Faculty of Arts, including what it focuses on and key structure details if available.",
        last_intent="PROGRAM_OVERVIEW",
    )
    result = prepare_turn("abc", "But can you list all the courses for it", state)

    assert result.workflow_reply is None
    assert "Arts and Contemporary Studies" in result.effective_question
    assert "for it" not in result.effective_question.lower()


def test_unknown_people_question_falls_back_instead_of_rag():
    state = SessionState(session_id="abc")
    result = prepare_turn("abc", "Who is Dr. Valerie Deacon?", state)

    assert result.workflow_reply == turn_prep._FALLBACK_REPLY
    assert result.state_after.last_effective_question is None


def test_confusion_turn_uses_program_clarification_when_program_missing():
    state = SessionState(session_id="abc", pending_slot="program", pending_intent="PROGRAM_REQUIREMENTS")
    result = prepare_turn("abc", "I dont know", state)

    assert result.workflow_reply == turn_prep._PROGRAM_CLARIFICATION_PROMPT
    assert result.state_after.pending_slot == "program"


def test_student_support_queries_remain_supported():
    state = SessionState(session_id="abc")
    result = prepare_turn("abc", "Im struggling with my mental health", state)

    assert result.workflow_reply is None
    assert "mental health" in result.effective_question.lower()


def test_turn_prep_logger_is_stdout_backed():
    assert turn_prep.logger.level == turn_prep.logging.INFO
    assert turn_prep.logger.propagate is False
    assert any(getattr(h, "_tmu_turn_prep_handler", False) for h in turn_prep.logger.handlers)
