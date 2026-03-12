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


def test_program_slot_fill_flow():
    state = SessionState(session_id="abc")

    first = prepare_turn("abc", "What required classes do I need for my major?", state)
    assert first.workflow_reply is not None
    assert first.state_after.pending_slot == "program"

    second = prepare_turn("abc", "Criminology BA", first.state_after)
    assert second.workflow_reply is None
    assert "Criminology" in second.effective_question
    assert second.state_after.program == "Criminology"


def test_program_scoped_followup_uses_known_program():
    state = SessionState(session_id="abc", program="English")
    result = prepare_turn("abc", "Is there co-op?", state)

    assert result.workflow_reply is None
    assert "English" in result.effective_question


def test_standalone_question_does_not_force_prior_context():
    state = SessionState(
        session_id="abc",
        program="Psychology",
        last_effective_question="What are the required courses for the Psychology program?",
    )
    result = prepare_turn("abc", "How do I apply?", state)

    assert result.effective_question == "How do I apply?"


def test_turn_prep_logger_is_stdout_backed():
    assert turn_prep.logger.level == turn_prep.logging.INFO
    assert turn_prep.logger.propagate is False
    assert any(getattr(h, "_tmu_turn_prep_handler", False) for h in turn_prep.logger.handlers)
