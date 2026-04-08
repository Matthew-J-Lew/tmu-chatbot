from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import re
import sys
from typing import Optional

from app.api.program_registry import match_program
from app.api.retrieval_policy import choose_retrieval_policy
from app.api.session_store import SessionState


_SUPPORTED_ACADEMIC_INTENTS = {
    "PROGRAM_REQUIREMENTS",
    "PROGRAM_COOP",
    "PROGRAM_OVERVIEW",
    "PROGRAMS_LIST_UNDERGRAD",
    "PROGRAMS_LIST_GRAD",
    "MINOR_CERTIFICATE",
    "MINOR_DECLARATION",
    "ADMISSIONS",
    "STUDENT_SUPPORT",
    "ACADEMIC_FOLLOW_UP",
    "COURSE_PLANNING",
    "COURSE_ENROLMENT",
    "COURSE_INTENTIONS",
    "COURSE_WAITLIST",
    "COURSE_MANAGEMENT",
    "PROGRAM_CHANGE",
    "GRADUATION_PROGRESS",
    "ACADEMIC_CONSIDERATION",
    "MISSED_ASSESSMENT",
    "ACADEMIC_STANDING",
    "GPA_STANDING",
    "IMPORTANT_DATES",
    "EXAM_DATES",
    "ADVISOR_CONTACT",
    "CHANG_SCHOOL_CREDIT",
    "ACADEMIC_ACCOMMODATIONS",
    "MENTAL_HEALTH_SUPPORT",
    "FACULTY_OF_ARTS_OVERVIEW",
    "CHANG_ENROLMENT",
}
_PROGRAM_REQUIRED_INTENTS = {"PROGRAM_REQUIREMENTS", "PROGRAM_COOP", "PROGRAM_OVERVIEW"}
_PROGRAM_CONTEXT_MAX_AGE = 8
_YEAR_CONTEXT_MAX_AGE = 6

_PROGRAM_PROMPT = (
    "Each program has different degree requirements. Which program are you in "
    "(for example: Criminology BA, English BA, Psychology BA)?"
)
_PROGRAM_CLARIFICATION_PROMPT = (
    "No problem - tell me your Faculty of Arts program "
    "(for example: Criminology BA, English BA, Psychology BA) and I can help from there."
)
_COURSE_PLANNING_PROGRAM_PROMPT = (
    "I can help with that. Which Faculty of Arts program are you in "
    "(for example: Criminology, English, or Psychology)?"
)
_YEAR_PROMPT = (
    "What year are you in "
    "(for example: first year, second year, third year, or fourth year)?"
)
_FALLBACK_REPLY = (
    "Sorry, I didn't quite understand that. I can help with Faculty of Arts programs, "
    "courses, requirements, co-op, minors, admissions, and related TMU information. "
    "Could you be more specific?"
)
_IN_SCOPE_CLARIFICATION_REPLY = (
    "I can help with official TMU and Faculty of Arts information, but that question is a bit broad. "
    "Ask about something specific like admissions, programs, enrolment, requirements, co-op, Chang School courses, or important dates."
)
_OFFICIAL_INFO_LIMITATION_REPLY = (
    "I can help with official TMU and Faculty of Arts information, but I can't reliably answer general public-opinion or review questions. "
    "Ask me about official TMU information instead, like admissions, programs, enrolment, or student services."
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
    study_year: Optional[str] = None


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
    if act.study_year:
        _remember_study_year(state_after, act.study_year)

    effective_question = user_question.strip()
    workflow_reply: Optional[str] = None

    if _should_clear_pending_state_for_new_turn(act, state_before, detected_program):
        _clear_pending_context(state_after)

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
                if resumed_intent == "COURSE_PLANNING":
                    year = act.study_year or _year_for_supported_turn(user_question, state_before)
                    if year:
                        _remember_study_year(state_after, year)
                        effective_question = _course_planning_query(detected_program, year)
                        state_after.pending_slot = None
                        state_after.pending_intent = None
                        state_after.last_intent = resumed_intent
                        state_after.active_topic = effective_question
                    else:
                        state_after.pending_slot = "year"
                        state_after.pending_intent = resumed_intent
                        workflow_reply = _YEAR_PROMPT
                else:
                    effective_question = _query_for_pending_intent(resumed_intent, detected_program)
                    state_after.pending_slot = None
                    state_after.pending_intent = None
                    state_after.last_intent = resumed_intent
                    state_after.active_topic = effective_question
            else:
                workflow_reply = _COURSE_PLANNING_PROGRAM_PROMPT if state_after.pending_intent == "COURSE_PLANNING" else _PROGRAM_PROMPT
        elif state_after.pending_slot == "year":
            year = act.study_year or _extract_study_year(user_question)
            if year and state_after.program:
                _remember_study_year(state_after, year)
                effective_question = _course_planning_query(state_after.program, year)
                state_after.pending_slot = None
                state_after.pending_intent = None
                state_after.last_intent = "COURSE_PLANNING"
                state_after.active_topic = effective_question
            else:
                workflow_reply = _YEAR_PROMPT
        elif academic_intent == "COURSE_PLANNING":
            program = detected_program or _program_for_supported_turn(user_question, state_before)
            year = act.study_year or _year_for_supported_turn(user_question, state_before)
            if program:
                if detected_program:
                    _remember_program(state_after, program)
                    state_after.pending_slot = None
                    state_after.pending_intent = None
                if year:
                    _remember_study_year(state_after, year)
                    effective_question = _course_planning_query(program, year)
                    state_after.active_topic = effective_question
                else:
                    state_after.pending_slot = "year"
                    state_after.pending_intent = academic_intent
                    workflow_reply = _YEAR_PROMPT
            else:
                state_after.pending_slot = "program"
                state_after.pending_intent = academic_intent
                workflow_reply = _COURSE_PLANNING_PROGRAM_PROMPT
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
            if act.study_year:
                _remember_study_year(state_after, act.study_year)
            state_after.active_topic = effective_question

    if workflow_reply:
        if routed and routed.preserve_context:
            if not state_after.active_topic:
                state_after.active_topic = state_before.active_topic
            state_after.last_effective_question = state_before.last_effective_question
        else:
            state_after.active_topic = None
            state_after.last_effective_question = None
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

    detected_year = _extract_study_year(question)

    if state.pending_slot == "year" and detected_year:
        return ConversationAct("YEAR_DECLARATION", study_year=detected_year)

    if detected_program and _is_program_declaration(question):
        if state.program and state.program != detected_program:
            return ConversationAct("PROGRAM_CORRECTION", program=detected_program, study_year=detected_year)
        return ConversationAct("PROGRAM_DECLARATION", program=detected_program, study_year=detected_year)

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
    if academic_intent in _SUPPORTED_ACADEMIC_INTENTS or academic_intent == "RAG_QA":
        return ConversationAct("SUPPORTED_ACADEMIC", program=detected_program, academic_intent=academic_intent, study_year=detected_year)

    if _asks_for_public_opinion(q):
        return ConversationAct("OFFICIAL_INFO_LIMITATION", study_year=detected_year)

    if _is_broad_in_scope_question(question):
        return ConversationAct("IN_SCOPE_CLARIFICATION", study_year=detected_year)

    if _looks_like_in_scope_tmu_question(question):
        return ConversationAct("SUPPORTED_ACADEMIC", program=detected_program, academic_intent="RAG_QA", study_year=detected_year)

    return ConversationAct("UNSUPPORTED", study_year=detected_year)


def _route_conversation_act(
    act: ConversationAct,
    state_before: SessionState,
    state_after: SessionState,
) -> Optional[RoutedTurn]:
    if act.kind == "EMPTY":
        return RoutedTurn(
            reply="I'm here when you're ready - ask me about Faculty of Arts programs, courses, requirements, or co-op.",
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
            reply="You're welcome - let me know if you have any other Faculty of Arts questions.",
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
        if state_before.pending_slot == "program":
            reply = _PROGRAM_CLARIFICATION_PROMPT
        elif state_before.pending_slot == "year":
            reply = _YEAR_PROMPT
        else:
            reply = _FALLBACK_REPLY
        return RoutedTurn(reply=reply, intent_label="UTILITY", preserve_context=True)

    if act.kind == "EMOTIONAL_REACTION":
        return RoutedTurn(
            reply="I understand. If you want, you can ask another Faculty of Arts question and I’ll do my best to help.",
            intent_label="UTILITY",
            preserve_context=True,
        )

    if act.kind in {"PROGRAM_DECLARATION", "PROGRAM_CORRECTION"} and act.program:
        _remember_program(state_after, act.program)
        if act.study_year:
            _remember_study_year(state_after, act.study_year)

        pending_intent = state_before.pending_intent
        resumed_intent = pending_intent or _resume_program_intent(state_before)
        if resumed_intent == "COURSE_PLANNING":
            year = act.study_year or _year_for_supported_turn(state_before.last_user_question or "", state_before)
            if year:
                _remember_study_year(state_after, year)
                return RoutedTurn(
                    effective_question=_course_planning_query(act.program, year),
                    intent_label="COURSE_PLANNING",
                    clear_pending_program=True,
                )
            state_after.pending_slot = "year"
            state_after.pending_intent = "COURSE_PLANNING"
            return RoutedTurn(
                reply=_YEAR_PROMPT,
                intent_label="COURSE_PLANNING",
                preserve_context=True,
            )
        if resumed_intent in _PROGRAM_REQUIRED_INTENTS:
            return RoutedTurn(
                effective_question=_query_for_pending_intent(resumed_intent, act.program),
                intent_label=resumed_intent,
                clear_pending_program=True,
            )

        return RoutedTurn(
            reply=f"Got it - I've updated your program to {act.program}. What would you like to know about it?",
            intent_label="PROGRAM_DECLARATION",
            preserve_context=True,
            clear_pending_program=True,
        )

    if act.kind == "YEAR_DECLARATION" and act.study_year:
        _remember_study_year(state_after, act.study_year)
        if state_before.pending_intent == "COURSE_PLANNING" and state_before.program:
            return RoutedTurn(
                effective_question=_course_planning_query(state_before.program, act.study_year),
                intent_label="COURSE_PLANNING",
                clear_pending_program=True,
            )
        return RoutedTurn(
            reply=f"Got it - you're in {act.study_year}. What would you like to know?",
            intent_label="ACADEMIC_CONTEXT",
            preserve_context=True,
            clear_pending_program=True,
        )

    if act.kind == "IN_SCOPE_CLARIFICATION":
        return RoutedTurn(
            reply=_IN_SCOPE_CLARIFICATION_REPLY,
            intent_label="UTILITY",
            preserve_context=False,
            clear_pending_program=True,
        )

    if act.kind == "OFFICIAL_INFO_LIMITATION":
        return RoutedTurn(
            reply=_OFFICIAL_INFO_LIMITATION_REPLY,
            intent_label="UTILITY",
            preserve_context=False,
            clear_pending_program=True,
        )

    if act.kind == "UNSUPPORTED":
        return RoutedTurn(
            reply=_FALLBACK_REPLY,
            intent_label="FALLBACK",
            preserve_context=False,
            clear_pending_program=True,
        )

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
    if _asks_for_course_planning(question):
        return "COURSE_PLANNING"
    if _asks_for_required_classes(question):
        return "PROGRAM_REQUIREMENTS"
    if _asks_about_coop(q):
        return "PROGRAM_COOP"

    direct_policy = choose_retrieval_policy(question, question)
    if direct_policy.label != "DEFAULT":
        return direct_policy.label

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
    if state.program and _can_apply_program_context(state) and _is_referential_program_followup(question):
        return "PROGRAM_REQUIREMENTS"
    if state.last_effective_question and _is_supported_follow_up(question):
        return "ACADEMIC_FOLLOW_UP"

    policy = choose_retrieval_policy(question, _rewrite_supported_question(question, state))
    if policy.label != "DEFAULT":
        return policy.label

    return None


def _query_for_pending_intent(intent: str, program: str) -> str:
    if intent == "COURSE_PLANNING":
        return _course_planning_query(program, "first year")
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
        f"What are the required courses, semester-by-semester requirements, and degree requirements "
        f"for the {program} program in TMU Faculty of Arts? Start with the exact single-program undergraduate calendar page's Full-Time, Four-Year Program structure, semester blocks, and required-course groupings. "
        f"Use Table I/II or Required Group pages only to support or clarify those requirement groups when relevant, and avoid combined-program or other-program Arts calendar pages unless absolutely necessary."
    )


def _course_planning_query(program: str, study_year: str) -> str:
    year_phrase = study_year.replace("-", " ")
    semester_hint = {
        "first year": "1st & 2nd Semester",
        "second year": "3rd & 4th Semester",
        "third year": "5th & 6th Semester",
        "fourth year": "7th & 8th Semester",
    }.get(year_phrase, year_phrase)
    return (
        f"What courses should a {year_phrase} student in the {program} program take in TMU Faculty of Arts? "
        f"Start with the exact {program} undergraduate calendar page's Full-Time, Four-Year Program and the semester block for {semester_hint}. "
        f"Use Table I/II pages or Required Group pages only to support or clarify requirement groups that the main program page points to. "
        f"Prefer exact semester rows and year-specific required-course lists over general overview prose, and avoid combined-program or other-program Arts calendar pages unless absolutely necessary. "
        f"Only include courses or requirement groups that are explicitly supported by the exact year-specific curriculum evidence. "
        f"Unless the student explicitly asks about co-op, default to the standard full-time four-year path and mention any co-op variation only briefly if relevant."
    )


def _rewrite_supported_question(question: str, state: SessionState) -> str:
    q = question.strip()
    if not q:
        return q

    explicit_program = match_program(q)
    if explicit_program:
        return q

    normalized = _normalize(q)
    if _asks_for_undergraduate_programs(normalized):
        if _asks_for_program_count(normalized):
            return "How many undergraduate programs are offered by the TMU Faculty of Arts?"
        return (
            "List every undergraduate program offered by the TMU Faculty of Arts. "
            "Include each program name exactly as shown on the Faculty of Arts undergraduate programs page."
        )
    if _asks_for_graduate_programs(normalized):
        if _asks_for_program_count(normalized):
            return "How many graduate programs are offered by the TMU Faculty of Arts?"
        return (
            "List every graduate program offered by the TMU Faculty of Arts. "
            "Include each program name exactly if available."
        )

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


def _clear_pending_context(state: SessionState) -> None:
    state.pending_slot = None
    state.pending_intent = None
    state.active_topic = None


def _should_clear_pending_state_for_new_turn(
    act: ConversationAct,
    state: SessionState,
    detected_program: Optional[str],
) -> bool:
    if not state.pending_slot and not state.pending_intent:
        return False

    if act.kind in {"PROGRAM_DECLARATION", "PROGRAM_CORRECTION", "YEAR_DECLARATION"}:
        return False

    if state.pending_slot == "program" and detected_program:
        return False
    if state.pending_slot == "year" and act.study_year:
        return False

    if act.kind in {"UNSUPPORTED", "IN_SCOPE_CLARIFICATION", "OFFICIAL_INFO_LIMITATION"}:
        return True

    if act.kind != "SUPPORTED_ACADEMIC":
        return False

    pending_intent = state.pending_intent
    if state.pending_slot == "program":
        return pending_intent != act.academic_intent

    if state.pending_slot == "year":
        return pending_intent != "COURSE_PLANNING" or act.academic_intent != "COURSE_PLANNING"

    return False


def _asks_for_course_planning(question: str) -> bool:
    q = _normalize(question)
    patterns = (
        r"\bwhat courses should i pick\b",
        r"\bwhat classes should i pick\b",
        r"\bwhat courses should i take\b",
        r"\bwhat classes should i take\b",
        r"\bwhat should i take\b",
        r"\bpick classes\b",
        r"\bpick courses\b",
        r"\bfirst year courses\b",
        r"\bsecond year courses\b",
        r"\bthird year courses\b",
        r"\bfourth year courses\b",
        r"\bfirst year classes\b",
        r"\bsecond year classes\b",
        r"\bthird year classes\b",
        r"\bfourth year classes\b",
    )
    return any(re.search(p, q) for p in patterns)


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


def _asks_for_program_count(q: str) -> bool:
    return bool(re.search(r"\bhow many\b", q)) or "number of" in q or "count of" in q


def _asks_about_coop(q: str) -> bool:
    return any(token in q for token in ("co op", "coop", "internship", "work term", "work terms"))


def _asks_about_minor_or_certificate(q: str) -> bool:
    return any(token in q for token in (" minor", "minor ", "certificate", "concentration", "declare a minor", "pick a minor", "choose a minor", "select a minor"))


def _asks_about_admissions(q: str) -> bool:
    return any(
        token in q
        for token in (
            "apply", "application", "admission", "admissions", "transfer", "requirements to apply",
            "join tmu", "join the university", "get into tmu", "get into the university",
        )
    )


def _asks_about_student_support(q: str) -> bool:
    return any(
        token in q
        for token in (
            "mental health",
            "counselling",
            "counseling",
            "academic support",
            "student support",
            "support services",
            "wellbeing",
            "well being",
            "accommodation",
            "accommodations",
            "probation",
        )
    )


def _asks_for_program_overview(q: str) -> bool:
    return any(
        q.startswith(prefix)
        for prefix in (
            "tell me about",
            "can you tell me about",
            "what can you tell me about",
            "tell me more about",
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


def _asks_for_public_opinion(q: str) -> bool:
    return any(
        phrase in q
        for phrase in (
            "what do people say about",
            "what do students say about",
            "reviews of",
            "reviews about",
            "is tmu good",
            "is the university good",
            "reputation of",
        )
    )


def _looks_like_in_scope_tmu_question(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False

    if _asks_for_public_opinion(q):
        return False

    domain_tokens = (
        "tmu", "toronto metropolitan", "faculty of arts", "chang school", "myservicehub",
        "course", "courses", "class", "classes", "program", "programs", "major", "minor",
        "certificate", "co op", "coop", "admission", "admissions", "apply", "application",
        "enroll", "enrol", "register", "waitlist", "wait list", "course intentions",
        "academic consideration", "exam", "exams", "deadline", "advisor", "advising",
        "gpa", "probation", "accommodation", "accommodations", "counselling", "counseling",
    )
    asks = (
        q.startswith((
            "how", "what", "when", "where", "who", "can", "do", "does", "is", "are",
            "tell me", "give me", "i want", "i need", "help me",
        ))
        or "?" in question
    )
    return asks and any(token in q for token in domain_tokens)


def _is_broad_in_scope_question(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False

    if any(
        phrase in q
        for phrase in (
            "tell me about tmu",
            "can you tell me about tmu",
            "what can you tell me about tmu",
            "tell me about toronto metropolitan",
            "can you tell me about toronto metropolitan",
            "what can you tell me about toronto metropolitan",
        )
    ):
        return True

    broad_heads = ("tell me about", "can you tell me about", "what can you tell me about", "what is")
    broad_targets = ("tmu", "toronto metropolitan", "the university")
    return q.startswith(broad_heads) and any(target in q for target in broad_targets)


def _is_supported_follow_up(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False

    explicit_followup_markers = (
        "what if",
        "what about",
        "how about",
        "and what about",
        "and how about",
        "but what about",
        "but how about",
        "but can you",
        "and can you",
        "does that",
        "does it",
        "will that",
        "will it",
        "can i do that",
        "can i do it",
        "how does that",
        "how does it",
        "when is that",
        "when is it",
    )
    if any(q.startswith(marker) for marker in explicit_followup_markers):
        return True

    if _has_standalone_topic_anchor(q):
        return False

    referential_pattern = re.compile(r"\b(it|that|those|them|there|this|these)\b")
    if len(q.split()) <= 8 and referential_pattern.search(q):
        return True
    return False


def _has_standalone_topic_anchor(q: str) -> bool:
    if match_program(q):
        return True
    anchors = (
        "chang school",
        "exam",
        "exams",
        "final exam",
        "important dates",
        "deadline",
        "gpa",
        "probation",
        "minor",
        "accommodation",
        "accommodations",
        "mental health",
        "counselling",
        "counseling",
        "major",
        "program",
        "class",
        "classes",
        "course",
        "courses",
        "advisor",
        "advising",
        "waitlist",
        "wait list",
        "enroll",
        "enrol",
        "transfer",
        "degree",
        "academic consideration",
        "test",
        "quiz",
        "midterm",
    )
    return any(token in q for token in anchors)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9:()]+", " ", text)
    text = re.sub(r"\bjoin (a|the)? ?class\b", "enroll in a class", text)
    text = re.sub(r"\bjoin (a|the)? ?course\b", "enroll in a course", text)
    text = re.sub(r"\bjoin classes\b", "enroll in classes", text)
    text = re.sub(r"\bjoin courses\b", "enroll in courses", text)
    text = re.sub(r"\bjoin tmu\b", "apply to tmu", text)
    text = re.sub(r"\bjoin the university\b", "apply to tmu", text)
    text = re.sub(r"\bregister for (a )?class\b", "enroll in a class", text)
    text = re.sub(r"\bregister for (a )?course\b", "enroll in a course", text)
    text = re.sub(r"\bregister for classes\b", "enroll in classes", text)
    text = re.sub(r"\bregister for courses\b", "enroll in courses", text)
    text = re.sub(r"\bchang\b", "chang school", text)
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


def _remember_study_year(state: SessionState, study_year: str) -> None:
    state.study_year = study_year
    state.metadata = dict(state.metadata or {})
    state.metadata["last_year_turn"] = state.turn_count


def _year_context_age(state: SessionState) -> int:
    last_year_turn = (state.metadata or {}).get("last_year_turn")
    if isinstance(last_year_turn, int):
        return max(0, state.turn_count - last_year_turn)
    return _YEAR_CONTEXT_MAX_AGE + 1


def _can_apply_year_context(state: SessionState) -> bool:
    return bool(state.study_year) and _year_context_age(state) <= _YEAR_CONTEXT_MAX_AGE


def _year_for_supported_turn(question: str, state: SessionState) -> Optional[str]:
    explicit_year = _extract_study_year(question)
    if explicit_year:
        return explicit_year
    if _can_apply_year_context(state):
        return state.study_year
    return None


def _extract_study_year(question: str) -> Optional[str]:
    q = _normalize(question)
    mapping = {
        "first year": (r"\b(first year|1st year|year 1)\b",),
        "second year": (r"\b(second year|2nd year|year 2)\b",),
        "third year": (r"\b(third year|3rd year|year 3)\b",),
        "fourth year": (r"\b(fourth year|4th year|year 4)\b",),
    }
    for label, patterns in mapping.items():
        if any(re.search(p, q) for p in patterns):
            return label
    return None


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
