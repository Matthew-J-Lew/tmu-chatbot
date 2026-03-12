from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import re
from typing import Optional

from app.api.program_registry import match_program
from app.api.session_store import SessionState


logger = logging.getLogger("tmu.turn_prep")


_PROGRAM_PROMPT = (
    "Each program has different degree requirements. Which program are you in "
    "(for example: Criminology BA, English BA, Psychology BA)?"
)


@dataclass
class TurnPrepResult:
    session_id: str
    state_before: SessionState
    state_after: SessionState
    effective_question: str
    workflow_reply: Optional[str] = None


def prepare_turn(session_id: str, user_question: str, state_before: SessionState) -> TurnPrepResult:
    state_after = state_before.clone()
    state_after.session_id = session_id
    state_after.turn_count += 1
    state_after.last_user_question = user_question

    detected_program = match_program(user_question)
    if detected_program:
        state_after.program = detected_program

    effective_question = user_question.strip()
    workflow_reply: Optional[str] = None

    if state_after.pending_slot == "program":
        if detected_program:
            state_after.pending_slot = None
            state_after.pending_intent = None
            effective_question = _program_requirements_query(detected_program)
            state_after.last_intent = "PROGRAM_REQUIREMENTS"
            state_after.active_topic = effective_question
        else:
            workflow_reply = _PROGRAM_PROMPT
    elif _asks_for_required_classes(user_question):
        if detected_program or state_after.program:
            program = detected_program or state_after.program
            effective_question = _program_requirements_query(program)
            state_after.program = program
            state_after.pending_slot = None
            state_after.pending_intent = None
            state_after.last_intent = "PROGRAM_REQUIREMENTS"
            state_after.active_topic = effective_question
        else:
            state_after.pending_slot = "program"
            state_after.pending_intent = "PROGRAM_REQUIREMENTS"
            state_after.last_intent = "PROGRAM_REQUIREMENTS"
            workflow_reply = _PROGRAM_PROMPT
    else:
        effective_question = _rewrite_with_context(user_question, state_after)
        state_after.last_intent = _infer_intent_label(user_question, state_after)
        state_after.active_topic = effective_question
        if detected_program:
            state_after.pending_slot = None
            state_after.pending_intent = None

    if workflow_reply:
        # Keep the prior active topic if we're pausing to collect a slot value.
        if not state_after.active_topic:
            state_after.active_topic = state_before.active_topic
        state_after.last_effective_question = state_before.last_effective_question
        effective_question = user_question.strip()
    else:
        state_after.last_effective_question = effective_question

    result = TurnPrepResult(
        session_id=session_id,
        state_before=state_before,
        state_after=state_after,
        effective_question=effective_question,
        workflow_reply=workflow_reply,
    )
    log_turn_prep(result, save_for=session_id)
    return result


def log_turn_prep(result: TurnPrepResult, save_for: Optional[str]) -> None:
    logger.info("SESSION_ID=%s", result.session_id)
    logger.info("STATE_BEFORE=%s", json.dumps(asdict(result.state_before), sort_keys=True))
    logger.info("EFFECTIVE_QUESTION=%s", result.effective_question)
    logger.info("STATE_AFTER=%s", json.dumps(asdict(result.state_after), sort_keys=True))
    logger.info("STATE_SAVED_FOR=%s", save_for)


def _asks_for_required_classes(question: str) -> bool:
    q = _normalize(question)
    patterns = (
        r"\bwhat required classes do i need\b",
        r"\bwhat classes do i need for my major\b",
        r"\brequired classes\b",
        r"\brequired courses\b",
        r"\bdegree requirements\b",
        r"\bcourse requirements\b",
        r"\bwhat courses do i need\b",
    )
    return any(re.search(p, q) for p in patterns)


def _rewrite_with_context(question: str, state: SessionState) -> str:
    q = question.strip()
    if not q:
        return q

    explicit_program = match_program(q)
    if explicit_program:
        return q

    if state.program and _is_program_scoped(q):
        return f"{q} for the {state.program} program in TMU Faculty of Arts"

    if _is_follow_up(q) and state.last_effective_question:
        if state.program:
            return (
                f"{q} for the {state.program} program in TMU Faculty of Arts. "
                f"Prior topic: {state.last_effective_question}"
            )
        return f"{q}. Prior topic: {state.last_effective_question}"

    return q


def _program_requirements_query(program: str) -> str:
    return (
        f"What are the required courses, first-year requirements, and degree requirements "
        f"for the {program} program in TMU Faculty of Arts? Use the undergraduate calendar when possible."
    )


def _is_program_scoped(question: str) -> bool:
    q = _normalize(question)
    return any(
        token in q
        for token in (
            "course",
            "courses",
            "class",
            "classes",
            "elective",
            "electives",
            "required",
            "requirement",
            "requirements",
            "major",
            "minor",
            "co op",
            "coop",
            "internship",
            "career",
            "first year",
            "second year",
            "third year",
            "fourth year",
        )
    )


def _is_follow_up(question: str) -> bool:
    q = _normalize(question)
    standalone_openers = (
        "how do i",
        "how to",
        "what is",
        "what are",
        "when is",
        "where is",
        "who is",
        "show me",
        "list",
        "tell me about",
    )
    markers = (
        "what about",
        "how about",
        "and ",
        "also ",
        "what if",
        "is that",
        "when is that",
    )

    if any(q.startswith(marker) for marker in markers):
        return True

    if re.search(r"\b(it|that|those|them|there|this)\b", q):
        return True

    if any(q.startswith(prefix) for prefix in standalone_openers):
        return False

    return len(q.split()) <= 5


def _infer_intent_label(question: str, state: SessionState) -> str:
    if state.program and _is_program_scoped(question):
        return "PROGRAM_SCOPED"
    if _is_follow_up(question) and state.last_effective_question:
        return "FOLLOW_UP"
    return "RAG_QA"


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
