from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import re
import sys
from typing import Optional

from app.api.program_registry import match_program
from app.api.session_store import SessionState


_SUPPORTED_ACADEMIC_INTENTS = {
    "PROGRAM_REQUIREMENTS",
    "PROGRAM_COOP",
    "PROGRAM_OVERVIEW",
    "PROGRAMS_LIST_UNDERGRAD",
    "PROGRAMS_LIST_GRAD",
    "MINOR_CERTIFICATE",
    "ADMISSIONS",
    "STUDENT_SUPPORT",
    "ACADEMIC_FOLLOW_UP",
}
_PROGRAM_REQUIRED_INTENTS = {"PROGRAM_REQUIREMENTS", "PROGRAM_COOP", "PROGRAM_OVERVIEW"}
_PROGRAM_CONTEXT_MAX_AGE = 8

_PROGRAM_PROMPT = (
    "Each program has different degree requirements. Which program are you in "
    "(for example: Criminology BA, English BA, Psychology BA)?"
)
_PROGRAM_CLARIFICATION_PROMPT = (
    "No problem — tell me your Faculty of Arts program "
    "(for example: Criminology BA, English BA, Psychology BA) and I can help from there."
)
_FALLBACK_REPLY = (
    "Sorry, I didn’t quite understand that. I can help with Faculty of Arts programs, "
    "courses, requirements, co-op, minors, admissions, and related TMU information. "
    "Could you be more specific?"
)


def _build_turn_prep_logger() -> logging.Logger:
    """Create a logger that always writes turn-prep lines to container stdout."""
    logger = logging.getLogger("tmu.turn_prep")
    logger.setLevel(logging.INFO)

    if not any(getattr(h, "_tmu_turn_prep_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handler._tmu_turn_prep_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = _build_turn_prep_logger()


@dataclass
class TurnPrepResult:
    session_id: str
    state_before: SessionState
    state_after: SessionState
    effective_question: str
    workflow_reply: Optional[str] = None


@dataclass(frozen=True)
class RoutedTurn:
    reply: Optional[str] = None
    effective_question: Optional[str] = None
    intent_label: Optional[str] = None
    preserve_context: bool = False
    clear_pending_program: bool = False


@dataclass(frozen=True)
class ConversationAct:
    kind: str
    program: Optional[str] = None
    academic_intent: Optional[str] = None


def prepare_turn(session_id: str, user_question: str, state_before: SessionState) -> TurnPrepResult:
    state_after = state_before.clone()
    state_after.session_id = session_id
    state_after.turn_count += 1
    state_after.last_user_question = user_question
    state_after.metadata = dict(state_after.metadata or {})

    detected_program = match_program(user_question)
    act = _classify_conversation_act(user_question, state_before, detected_program)

    if detected_program and act.kind in {"PROGRAM_DECLARATION", "PROGRAM_CORRECTION"}:
        _remember_program(state_after, detected_program)

    effective_question = user_question.strip()
    workflow_reply: Optional[str] = None

    routed = _route_conversation_act(act, state_before, state_after)
    if routed is not None:
        workflow_reply = routed.reply
        effective_question = routed.effective_question or effective_question
        if routed.intent_label:
            state_after.last_intent = routed.intent_label
        if routed.clear_pending_program:
            state_after.pending_slot = None
            state_after.pending_intent = None
        if not routed.preserve_context and not workflow_reply:
            state_after.active_topic = effective_question
    else:
        academic_intent = act.academic_intent or "RAG_QA"
        state_after.last_intent = academic_intent

        if state_after.pending_slot == "program":
            if detected_program:
                _remember_program(state_after, detected_program)
                resumed_intent = state_after.pending_intent or "PROGRAM_REQUIREMENTS"
                effective_question = _query_for_pending_intent(resumed_intent, detected_program)
                state_after.pending_slot = None
                state_after.pending_intent = None
                state_after.last_intent = resumed_intent
                state_after.active_topic = effective_question
            else:
                workflow_reply = _PROGRAM_PROMPT
        elif academic_intent in _PROGRAM_REQUIRED_INTENTS:
            program = detected_program or _program_for_supported_turn(user_question, state_before)
            if program:
                if detected_program:
                    _remember_program(state_after, program)
                    state_after.pending_slot = None
                    state_after.pending_intent = None
                effective_question = _query_for_pending_intent(academic_intent, program)
                state_after.active_topic = effective_question
            else:
                state_after.pending_slot = "program"
                state_after.pending_intent = academic_intent
                workflow_reply = _PROGRAM_PROMPT
        elif academic_intent == "ACADEMIC_FOLLOW_UP":
            effective_question = _rewrite_follow_up_with_context(user_question, state_before)
            state_after.active_topic = effective_question
        else:
            effective_question = _rewrite_supported_question(user_question, state_before)
            if detected_program:
                _remember_program(state_after, detected_program)
                state_after.pending_slot = None
                state_after.pending_intent = None
            state_after.active_topic = effective_question

    if workflow_reply:
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


def _classify_conversation_act(
    question: str,
    state: SessionState,
    detected_program: Optional[str],
) -> ConversationAct:
    q = _normalize(question)
    if not q:
        return ConversationAct("EMPTY")

    if detected_program and _is_program_declaration(question):
        if state.program and state.program != detected_program:
            return ConversationAct("PROGRAM_CORRECTION", program=detected_program)
        return ConversationAct("PROGRAM_DECLARATION", program=detected_program)

    if _is_greeting(q):
        return ConversationAct("GREETING")
    if _is_acknowledgement(q):
        return ConversationAct("ACKNOWLEDGEMENT")
    if _is_goodbye(q):
        return ConversationAct("GOODBYE")
    if _is_identity_question(q):
        return ConversationAct("BOT_IDENTITY")
    if _is_capability_question(q):
        return ConversationAct("BOT_CAPABILITY")
    if _is_confusion(q):
        return ConversationAct("CONFUSION")
    if _is_emotional_reaction(question, q):
        return ConversationAct("EMOTIONAL_REACTION")

    academic_intent = _classify_supported_academic_intent(question, state, detected_program)
    if academic_intent in _SUPPORTED_ACADEMIC_INTENTS:
        return ConversationAct("SUPPORTED_ACADEMIC", program=detected_program, academic_intent=academic_intent)

    return ConversationAct("UNSUPPORTED")


def _route_conversation_act(
    act: ConversationAct,
    state_before: SessionState,
    state_after: SessionState,
) -> Optional[RoutedTurn]:
    if act.kind == "EMPTY":
        return RoutedTurn(
            reply="I’m here when you’re ready — ask me about Faculty of Arts programs, courses, requirements, or co-op.",
            intent_label="UTILITY",
            preserve_context=True,
        )

    if act.kind == "GREETING":
        return RoutedTurn(
            reply="Hello! What can I help you with today?",
            intent_label="GREETING",
            preserve_context=True,
        )

    if act.kind == "ACKNOWLEDGEMENT":
        return RoutedTurn(
            reply="You’re welcome — let me know if you have any other Faculty of Arts questions.",
            intent_label="UTILITY",
            preserve_context=True,
        )

    if act.kind == "GOODBYE":
        return RoutedTurn(
            reply="Take care! If you need anything else about the Faculty of Arts, I’m here to help.",
            intent_label="UTILITY",
            preserve_context=True,
        )

    if act.kind == "BOT_IDENTITY":
        return RoutedTurn(
            reply="I’m a TMU Faculty of Arts virtual assistant. I help with official Arts program information, requirements, courses, co-op, and related TMU resources.",
            intent_label="BOT_CAPABILITY",
            preserve_context=True,
        )

    if act.kind == "BOT_CAPABILITY":
        return RoutedTurn(
            reply="I can help with Faculty of Arts undergraduate programs, required courses, degree requirements, co-op, minors, admissions, and related TMU information.",
            intent_label="BOT_CAPABILITY",
            preserve_context=True,
        )

    if act.kind == "CONFUSION":
        reply = _PROGRAM_CLARIFICATION_PROMPT if state_before.pending_slot == "program" else _FALLBACK_REPLY
        return RoutedTurn(reply=reply, intent_label="UTILITY", preserve_context=True)

    if act.kind == "EMOTIONAL_REACTION":
        return RoutedTurn(
            reply="I understand. If you want, you can ask another Faculty of Arts question and I’ll do my best to help.",
            intent_label="UTILITY",
            preserve_context=True,
        )

    if act.kind in {"PROGRAM_DECLARATION", "PROGRAM_CORRECTION"} and act.program:
        _remember_program(state_after, act.program)

        pending_intent = state_before.pending_intent
        resumed_intent = pending_intent or _resume_program_intent(state_before)
        if resumed_intent in _PROGRAM_REQUIRED_INTENTS:
            return RoutedTurn(
                effective_question=_query_for_pending_intent(resumed_intent, act.program),
                intent_label=resumed_intent,
                clear_pending_program=True,
            )

        return RoutedTurn(
            reply=f"Got it — I’ve updated your program to {act.program}. What would you like to know about it?",
            intent_label="PROGRAM_DECLARATION",
            preserve_context=True,
            clear_pending_program=True,
        )

    if act.kind == "UNSUPPORTED":
        return RoutedTurn(reply=_FALLBACK_REPLY, intent_label="FALLBACK", preserve_context=True)

    return None


def _classify_supported_academic_intent(
    question: str,
    state: SessionState,
    detected_program: Optional[str],
) -> Optional[str]:
    q = _normalize(question)

    if _asks_for_undergraduate_programs(q):
        return "PROGRAMS_LIST_UNDERGRAD"
    if _asks_for_graduate_programs(q):
        return "PROGRAMS_LIST_GRAD"
    if _asks_for_required_classes(question):
        return "PROGRAM_REQUIREMENTS"
    if _asks_about_coop(q):
        return "PROGRAM_COOP"
    if _asks_about_minor_or_certificate(q):
        return "MINOR_CERTIFICATE"
    if _asks_about_admissions(q):
        return "ADMISSIONS"
    if _asks_about_student_support(q):
        return "STUDENT_SUPPORT"
    if detected_program and _asks_for_program_overview(q):
        return "PROGRAM_OVERVIEW"
    if detected_program and _is_program_scoped(q):
        return "PROGRAM_REQUIREMENTS"
    if state.last_effective_question and _is_supported_follow_up(q):
        return "ACADEMIC_FOLLOW_UP"
    if state.program and _can_apply_program_context(state) and _is_referential_program_followup(question):
        return "PROGRAM_REQUIREMENTS"

    return None


def _query_for_pending_intent(intent: str, program: str) -> str:
    if intent == "PROGRAM_COOP":
        return (
            f"Is there a co-op option for the {program} program in TMU Faculty of Arts? "
            f"Include timing or eligibility details if available."
        )
    if intent == "PROGRAM_OVERVIEW":
        return (
            f"Provide an overview of the {program} program in TMU Faculty of Arts, "
            f"including what it focuses on and key structure details if available."
        )
    return _program_requirements_query(program)


def _program_requirements_query(program: str) -> str:
    return (
        f"What are the required courses, first-year requirements, and degree requirements "
        f"for the {program} program in TMU Faculty of Arts? Use the undergraduate calendar when possible."
    )


def _rewrite_supported_question(question: str, state: SessionState) -> str:
    q = question.strip()
    if not q:
        return q

    explicit_program = match_program(q)
    if explicit_program:
        return q

    if state.program and _can_apply_program_context(state) and _is_referential_program_followup(q):
        return _resolve_program_referent(q, state.program)

    return q


def _rewrite_follow_up_with_context(question: str, state: SessionState) -> str:
    q = question.strip()
    if not q:
        return q

    if state.program and _can_apply_program_context(state) and _is_referential_program_followup(q):
        q = _resolve_program_referent(q, state.program)

    if state.last_effective_question:
        return f"{q}. Prior topic: {state.last_effective_question}"
    return q


def _program_for_supported_turn(question: str, state: SessionState) -> Optional[str]:
    if not state.program:
        return None
    if not _can_apply_program_context(state):
        return None
    if _is_possessive_program_request(question):
        return state.program if _program_context_age(state) <= 3 else None
    return state.program


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
        r"\blist all the required courses\b",
        r"\blist the required courses\b",
        r"\blist all the courses\b",
        r"\bcurriculum\b",
    )
    return any(re.search(p, q) for p in patterns)


def _asks_for_undergraduate_programs(q: str) -> bool:
    return (
        "undergraduate program" in q
        or "undergraduate programs" in q
        or bool(re.search(r"\blist all undergraduate programs\b", q))
    )


def _asks_for_graduate_programs(q: str) -> bool:
    return (
        "graduate program" in q
        or "graduate programs" in q
        or bool(re.search(r"\blist all graduate programs\b", q))
    )


def _asks_about_coop(q: str) -> bool:
    return any(token in q for token in ("co op", "coop", "internship", "work term", "work terms"))


def _asks_about_minor_or_certificate(q: str) -> bool:
    return any(token in q for token in (" minor", "minor ", "certificate", "concentration", "declare a minor"))


def _asks_about_admissions(q: str) -> bool:
    return any(token in q for token in ("apply", "application", "admission", "admissions", "transfer", "requirements to apply"))


def _asks_about_student_support(q: str) -> bool:
    return any(
        token in q
        for token in (
            "mental health",
            "counselling",
            "counseling",
            "probation",
            "advisor",
            "advising",
            "academic support",
            "student support",
            "support services",
        )
    )


def _asks_for_program_overview(q: str) -> bool:
    return any(
        q.startswith(prefix)
        for prefix in (
            "tell me about",
            "what is",
            "what about",
            "how about",
            "give me an overview of",
            "overview of",
        )
    )


def _is_program_scoped(q: str) -> bool:
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
            "co op",
            "coop",
            "internship",
            "first year",
            "second year",
            "third year",
            "fourth year",
            "curriculum",
            "credit",
            "credits",
        )
    )


def _is_supported_follow_up(q: str) -> bool:
    if any(q.startswith(marker) for marker in ("what if", "what about", "how about", "but can you", "and what about", "and can you")):
        return True
    return bool(re.search(r"\b(it|that|those|them|there|this|these)\b", q))


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9:()]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_greeting(q: str) -> bool:
    if q in {"hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening"}:
        return True
    tokens = q.split()
    greeting_tokens = {"hi", "hello", "hey", "hiya"}
    stripped = [token for token in tokens if token in greeting_tokens]
    return bool(stripped) and len(tokens) <= 6 and len(stripped) == len(tokens)


def _is_acknowledgement(q: str) -> bool:
    patterns = (
        r"\bthank(s| you)?\b",
        r"\bok(ay)? thanks\b",
        r"\bgot it\b",
        r"\bperfect\b",
        r"\bawesome\b",
        r"\bsounds good\b",
        r"\bappreciate it\b",
    )
    return any(re.search(p, q) for p in patterns)


def _is_goodbye(q: str) -> bool:
    return q in {"bye", "goodbye", "see you", "see ya", "talk to you later"}


def _is_identity_question(q: str) -> bool:
    patterns = (
        r"\bare you (a )?(real )?bot\b",
        r"\bare you human\b",
        r"\bwho are you\b",
        r"\bwhat are you\b",
    )
    return any(re.search(p, q) for p in patterns)


def _is_capability_question(q: str) -> bool:
    patterns = (
        r"\bwhat do you do\b",
        r"\bwhat can you do\b",
        r"\bhow can you help\b",
        r"\bwhat do you help with\b",
    )
    return any(re.search(p, q) for p in patterns)


def _is_confusion(q: str) -> bool:
    patterns = (
        r"\bi do not know\b",
        r"\bi dont know\b",
        r"\bidk\b",
        r"\buh+\b",
        r"\bum+\b",
        r"\bumm+\b",
        r"\buhh+\b",
        r"\bhuh\b",
    )
    if q in {"?", "what", "huh", "uh", "umm", "uhhh", "uhhhhmmm"}:
        return True
    return len(q.split()) <= 4 and any(re.search(p, q) for p in patterns)


def _is_emotional_reaction(raw_question: str, q: str) -> bool:
    patterns = (
        r"\bohh?\b",
        r"\bthat sucks\b",
        r"\bthat is disappointing\b",
        r"\bokay sad\b",
    )
    return ":(" in raw_question or (len(q.split()) <= 5 and any(re.search(p, q) for p in patterns))


def _is_program_declaration(question: str) -> bool:
    q = _normalize(question)
    patterns = (
        r"\b(i am|i m|im|i am in|i m in|im in)\b",
        r"\bmy major is\b",
        r"\bmy program is\b",
        r"\bbut i am in\b",
        r"\bbut i m in\b",
        r"\bbut im in\b",
        r"\bactually\b",
        r"\bno\b",
    )
    return any(re.search(p, q) for p in patterns) or bool(match_program(question) and len(q.split()) <= 6)


def _can_apply_program_context(state: SessionState) -> bool:
    return bool(state.program) and _program_context_age(state) <= _PROGRAM_CONTEXT_MAX_AGE


def _program_context_age(state: SessionState) -> int:
    last_program_turn = (state.metadata or {}).get("last_program_turn")
    if isinstance(last_program_turn, int):
        return max(0, state.turn_count - last_program_turn)
    return _PROGRAM_CONTEXT_MAX_AGE + 1


def _remember_program(state: SessionState, program: str) -> None:
    state.program = program
    state.metadata = dict(state.metadata or {})
    state.metadata["last_program_turn"] = state.turn_count


def _is_possessive_program_request(question: str) -> bool:
    q = _normalize(question)
    return any(phrase in q for phrase in ("my major", "my program"))


def _resume_program_intent(state: SessionState) -> Optional[str]:
    if state.last_intent in _PROGRAM_REQUIRED_INTENTS and _can_apply_program_context(state):
        return state.last_intent
    return None


def _is_referential_program_followup(question: str) -> bool:
    q = _normalize(question)
    return _is_program_scoped(q) and bool(re.search(r"\b(it|that program|this program)\b", q))


def _resolve_program_referent(question: str, program: str) -> str:
    q = question.strip()
    q = re.sub(r"\bfor\s+it\b", f"for the {program} program", q, flags=re.IGNORECASE)
    q = re.sub(r"\bfor\s+that\s+program\b", f"for the {program} program", q, flags=re.IGNORECASE)
    q = re.sub(r"\bfor\s+this\s+program\b", f"for the {program} program", q, flags=re.IGNORECASE)
    q = re.sub(r"\bit\b", f"the {program} program", q, count=1, flags=re.IGNORECASE)
    return f"{q} in TMU Faculty of Arts"
