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
    "DEPARTMENTS_LIST",
    "MINOR_CERTIFICATE",
    "ADMISSIONS",
    "STUDENT_SUPPORT",
    "ACADEMIC_FOLLOW_UP",
    "COURSE_ENROLMENT",
    "COURSE_WAITLIST",
    "COURSE_MANAGEMENT",
    "COURSE_PLANNING",
    "PROGRAM_CHANGE",
    "ADVISOR_CONTACT",
    "IMPORTANT_DATES",
    "ACADEMIC_CONSIDERATION",
    "ACADEMIC_STANDING",
    "GRADUATION_PROGRESS",
    "GENERAL_STUDENT_INFO",
}
_PROGRAM_REQUIRED_INTENTS = {"PROGRAM_REQUIREMENTS", "PROGRAM_COOP", "PROGRAM_OVERVIEW"}
_PROGRAM_AND_YEAR_HELPFUL_INTENTS = {"COURSE_PLANNING", "GRADUATION_PROGRESS"}
_PROGRAM_CONTEXT_MAX_AGE = 8
_YEAR_CONTEXT_MAX_AGE = 8

_PROGRAM_PROMPT = (
    "I can help with that. Which Faculty of Arts program are you in "
    "(for example: Criminology BA, English BA, or Psychology BA)?"
)
_PROGRAM_YEAR_PROMPT = (
    "I can help with that. What program are you in, and what year are you in right now "
    "(for example: first year or second year)?"
)
_YEAR_PROMPT = (
    "What year are you in right now (for example: first year or second year)?"
)
_PROGRAM_CLARIFICATION_PROMPT = (
    "No problem — tell me your Faculty of Arts program "
    "(for example: Criminology BA, English BA, or Psychology BA) and I can help from there."
)
_FALLBACK_REPLY = (
    "I’m best at TMU Faculty of Arts questions about programs, courses, enrolment, requirements, co-op, minors, "
    "advising, admissions, and related student support. Could you tell me a bit more about what you need?"
)


def _build_turn_prep_logger() -> logging.Logger:
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
    study_year: Optional[str] = None
    academic_intent: Optional[str] = None


@dataclass(frozen=True)
class PendingResolution:
    effective_question: Optional[str] = None
    workflow_reply: Optional[str] = None
    intent_label: Optional[str] = None
    clear_pending: bool = False


def prepare_turn(session_id: str, user_question: str, state_before: SessionState) -> TurnPrepResult:
    state_after = state_before.clone()
    state_after.session_id = session_id
    state_after.turn_count += 1
    state_after.last_user_question = user_question
    state_after.metadata = dict(state_after.metadata or {})

    detected_program = match_program(user_question)
    detected_year = _match_study_year(user_question)
    act = _classify_conversation_act(user_question, state_before, detected_program, detected_year)

    if detected_program and act.kind in {"PROGRAM_DECLARATION", "PROGRAM_CORRECTION"}:
        _remember_program(state_after, detected_program)
    if detected_year:
        _remember_study_year(state_after, detected_year)

    effective_question = user_question.strip()
    workflow_reply: Optional[str] = None

    routed = _route_conversation_act(act, state_before, state_after)
    if routed is not None:
        workflow_reply = routed.reply
        effective_question = routed.effective_question or effective_question
        if routed.intent_label:
            state_after.last_intent = routed.intent_label
            state_after.question_mode = routed.intent_label
        if routed.clear_pending_program:
            state_after.pending_slot = None
            state_after.pending_intent = None
        if not routed.preserve_context and not workflow_reply:
            state_after.active_topic = effective_question
    else:
        academic_intent = act.academic_intent or "GENERAL_STUDENT_INFO"
        state_after.last_intent = academic_intent
        state_after.question_mode = academic_intent

        pending = _handle_pending_context(user_question, state_before, state_after, academic_intent, detected_program, detected_year)
        if pending is not None:
            workflow_reply = pending.workflow_reply
            if pending.effective_question:
                effective_question = pending.effective_question
            if pending.intent_label:
                state_after.last_intent = pending.intent_label
                state_after.question_mode = pending.intent_label
            if pending.clear_pending:
                state_after.pending_slot = None
                state_after.pending_intent = None
                if not workflow_reply:
                    state_after.active_topic = effective_question
        else:
            effective_question, workflow_reply = _handle_supported_academic_turn(
                user_question=user_question,
                state_before=state_before,
                state_after=state_after,
                academic_intent=academic_intent,
                detected_program=detected_program,
                detected_year=detected_year,
            )

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
    detected_year: Optional[str],
) -> ConversationAct:
    q = _normalize(question)
    if not q:
        return ConversationAct("EMPTY")

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
    if academic_intent in _SUPPORTED_ACADEMIC_INTENTS:
        return ConversationAct("SUPPORTED_ACADEMIC", program=detected_program, study_year=detected_year, academic_intent=academic_intent)

    if _is_probably_on_domain_question(question, state):
        return ConversationAct("SUPPORTED_ACADEMIC", program=detected_program, study_year=detected_year, academic_intent="GENERAL_STUDENT_INFO")

    return ConversationAct("OFF_DOMAIN")


def _route_conversation_act(
    act: ConversationAct,
    state_before: SessionState,
    state_after: SessionState,
) -> Optional[RoutedTurn]:
    if act.kind == "EMPTY":
        return RoutedTurn(
            reply="I’m here when you’re ready — ask me about Faculty of Arts programs, courses, enrolment, requirements, or co-op.",
            intent_label="UTILITY",
            preserve_context=True,
        )
    if act.kind == "GREETING":
        return RoutedTurn(reply="Hello! What can I help you with today?", intent_label="GREETING", preserve_context=True)
    if act.kind == "ACKNOWLEDGEMENT":
        return RoutedTurn(
            reply="You’re welcome — let me know if you have any other TMU Faculty of Arts questions.",
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
            reply="I’m a TMU Faculty of Arts virtual assistant. I help with official Arts program information, requirements, enrolment, co-op, and related student resources.",
            intent_label="BOT_CAPABILITY",
            preserve_context=True,
        )
    if act.kind == "BOT_CAPABILITY":
        return RoutedTurn(
            reply="I can help with Faculty of Arts undergraduate and graduate programs, required courses, enrolment, waitlists, co-op, minors, advising, admissions, and related TMU information.",
            intent_label="BOT_CAPABILITY",
            preserve_context=True,
        )
    if act.kind == "CONFUSION":
        reply = _PROGRAM_CLARIFICATION_PROMPT if state_before.pending_slot in {"program", "program_year"} else _FALLBACK_REPLY
        return RoutedTurn(reply=reply, intent_label="UTILITY", preserve_context=True)
    if act.kind == "EMOTIONAL_REACTION":
        return RoutedTurn(
            reply="I understand. If you want, ask me another TMU Faculty of Arts question and I’ll do my best to help.",
            intent_label="UTILITY",
            preserve_context=True,
        )
    if act.kind in {"PROGRAM_DECLARATION", "PROGRAM_CORRECTION"} and act.program:
        _remember_program(state_after, act.program)
        if act.study_year:
            _remember_study_year(state_after, act.study_year)
        resumed_intent = state_before.pending_intent or _resume_program_intent(state_before)
        if resumed_intent in _PROGRAM_REQUIRED_INTENTS:
            return RoutedTurn(
                effective_question=_query_for_pending_intent(resumed_intent, program=act.program, study_year=act.study_year or state_before.study_year),
                intent_label=resumed_intent,
                clear_pending_program=True,
            )
        if resumed_intent in _PROGRAM_AND_YEAR_HELPFUL_INTENTS:
            year = act.study_year or state_before.study_year if _can_apply_year_context(state_before) else act.study_year
            if not year:
                state_after.pending_slot = "year"
                state_after.pending_intent = resumed_intent
                return RoutedTurn(reply=_YEAR_PROMPT, intent_label=resumed_intent, preserve_context=True)
            return RoutedTurn(
                effective_question=_query_for_pending_intent(resumed_intent, program=act.program, study_year=year),
                intent_label=resumed_intent,
                clear_pending_program=True,
            )
        return RoutedTurn(
            reply=f"Got it — I’ve updated your program to {act.program}. What would you like to know about it?",
            intent_label="PROGRAM_DECLARATION",
            preserve_context=True,
            clear_pending_program=True,
        )
    if act.kind == "OFF_DOMAIN":
        return RoutedTurn(reply=_FALLBACK_REPLY, intent_label="FALLBACK", preserve_context=True)
    return None


def _handle_pending_context(
    user_question: str,
    state_before: SessionState,
    state_after: SessionState,
    academic_intent: str,
    detected_program: Optional[str],
    detected_year: Optional[str],
) -> Optional[PendingResolution]:
    slot = state_before.pending_slot
    pending_intent = state_before.pending_intent or academic_intent
    if not slot:
        return None

    program = detected_program or state_before.program if _can_apply_program_context(state_before) else detected_program
    study_year = detected_year or state_before.study_year if _can_apply_year_context(state_before) else detected_year

    if slot == "program":
        if program:
            return PendingResolution(
                effective_question=_query_for_pending_intent(pending_intent, program=program, study_year=study_year),
                intent_label=pending_intent,
                clear_pending=True,
            )
        return PendingResolution(workflow_reply=_PROGRAM_PROMPT, intent_label=pending_intent)

    if slot == "program_year":
        if program and study_year:
            return PendingResolution(
                effective_question=_query_for_pending_intent(pending_intent, program=program, study_year=study_year),
                intent_label=pending_intent,
                clear_pending=True,
            )
        if program and not study_year:
            state_after.pending_slot = "year"
            state_after.pending_intent = pending_intent
            return PendingResolution(workflow_reply=_YEAR_PROMPT, intent_label=pending_intent)
        return PendingResolution(workflow_reply=_PROGRAM_YEAR_PROMPT, intent_label=pending_intent)

    if slot == "year":
        if study_year:
            return PendingResolution(
                effective_question=_query_for_pending_intent(pending_intent, program=state_before.program or detected_program, study_year=study_year),
                intent_label=pending_intent,
                clear_pending=True,
            )
        return PendingResolution(workflow_reply=_YEAR_PROMPT, intent_label=pending_intent)

    return None


def _handle_supported_academic_turn(
    user_question: str,
    state_before: SessionState,
    state_after: SessionState,
    academic_intent: str,
    detected_program: Optional[str],
    detected_year: Optional[str],
) -> tuple[str, Optional[str]]:
    if state_after.pending_slot == "program":
        return user_question.strip(), _PROGRAM_PROMPT

    if academic_intent in _PROGRAM_REQUIRED_INTENTS:
        program = detected_program or _program_for_supported_turn(user_question, state_before)
        if program:
            return _query_for_pending_intent(academic_intent, program=program, study_year=detected_year or state_before.study_year), None
        state_after.pending_slot = "program"
        state_after.pending_intent = academic_intent
        return user_question.strip(), _PROGRAM_PROMPT

    if academic_intent in _PROGRAM_AND_YEAR_HELPFUL_INTENTS:
        program = detected_program or _program_for_supported_turn(user_question, state_before)
        study_year = detected_year or (state_before.study_year if _can_apply_year_context(state_before) else None)
        if program and study_year:
            return _query_for_pending_intent(academic_intent, program=program, study_year=study_year), None
        state_after.pending_slot = "program_year"
        state_after.pending_intent = academic_intent
        return user_question.strip(), _PROGRAM_YEAR_PROMPT

    if academic_intent == "ACADEMIC_FOLLOW_UP":
        if detected_program and state_before.last_intent in (_PROGRAM_REQUIRED_INTENTS | _PROGRAM_AND_YEAR_HELPFUL_INTENTS):
            if detected_program:
                _remember_program(state_after, detected_program)
            return _query_for_pending_intent(
                state_before.last_intent,
                program=detected_program or state_before.program,
                study_year=detected_year or state_before.study_year,
            ), None
        return _rewrite_follow_up_with_context(user_question, state_before), None

    effective = _rewrite_supported_question(user_question, state_before, academic_intent, detected_program, detected_year)
    if detected_program:
        _remember_program(state_after, detected_program)
    if detected_year:
        _remember_study_year(state_after, detected_year)
    state_after.active_topic = effective
    return effective, None


def _classify_supported_academic_intent(question: str, state: SessionState, detected_program: Optional[str]) -> Optional[str]:
    q = _normalize(question)
    if _asks_for_undergraduate_programs(q):
        return "PROGRAMS_LIST_UNDERGRAD"
    if _asks_for_graduate_programs(q):
        return "PROGRAMS_LIST_GRAD"
    if _asks_about_departments(q):
        return "DEPARTMENTS_LIST"
    if _asks_for_required_classes(question):
        return "PROGRAM_REQUIREMENTS"
    if _asks_about_coop(q):
        return "PROGRAM_COOP"
    if _asks_about_course_enrolment(q):
        return "COURSE_ENROLMENT"
    if _asks_about_waitlist_or_full_class(q):
        return "COURSE_WAITLIST"
    if _asks_about_course_management(q):
        return "COURSE_MANAGEMENT"
    if _asks_about_program_change(q):
        return "PROGRAM_CHANGE"
    if _asks_about_advisor(q):
        return "ADVISOR_CONTACT"
    if _asks_about_course_planning(q):
        return "COURSE_PLANNING"
    if _asks_about_graduation_progress(q):
        return "GRADUATION_PROGRESS"
    if _asks_about_minor_or_certificate(q):
        return "MINOR_CERTIFICATE"
    if _asks_about_admissions(q):
        return "ADMISSIONS"
    if _asks_about_important_dates(q):
        return "IMPORTANT_DATES"
    if _asks_about_academic_consideration(q):
        return "ACADEMIC_CONSIDERATION"
    if _asks_about_academic_standing(q):
        return "ACADEMIC_STANDING"
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


def _query_for_pending_intent(intent: str, program: Optional[str] = None, study_year: Optional[str] = None) -> str:
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
    if intent == "PROGRAM_REQUIREMENTS":
        return _program_requirements_query(program or "the selected")
    if intent == "COURSE_PLANNING":
        year_prefix = f"for a {study_year} student " if study_year else ""
        return (
            f"How should {year_prefix}in the {program} program at TMU Faculty of Arts plan courses? "
            f"Use Advisement Report, academic advising, and program requirements guidance if available."
        )
    if intent == "GRADUATION_PROGRESS":
        year_prefix = f"for a {study_year} student " if study_year else ""
        return (
            f"How can {year_prefix}in the {program} program at TMU Faculty of Arts check if they are on track to graduate? "
            f"Use Advisement Report and advising resources if available."
        )
    return program or ""


def _program_requirements_query(program: str) -> str:
    return (
        f"What are the required courses, first-year requirements, and degree requirements "
        f"for the {program} program in TMU Faculty of Arts? Use the undergraduate calendar when possible."
    )


def _rewrite_supported_question(
    question: str,
    state: SessionState,
    academic_intent: str,
    detected_program: Optional[str],
    detected_year: Optional[str],
) -> str:
    q = question.strip()
    if not q:
        return q

    normalized = _normalize(q)
    program = detected_program or (state.program if _can_apply_program_context(state) else None)
    study_year = detected_year or (state.study_year if _can_apply_year_context(state) else None)

    if academic_intent == "PROGRAMS_LIST_UNDERGRAD":
        return "List every undergraduate program offered by the TMU Faculty of Arts. Include each program name exactly as shown on the Faculty of Arts undergraduate programs page."
    if academic_intent == "PROGRAMS_LIST_GRAD":
        return "List every graduate program offered by the TMU Faculty of Arts. Include each program name exactly if available."
    if academic_intent == "DEPARTMENTS_LIST":
        return "List the departments in the TMU Faculty of Arts."
    if academic_intent == "MINOR_CERTIFICATE":
        return "How do TMU students declare a minor? Include Application to Graduate and MyServiceHub details if available."
    if academic_intent == "COURSE_ENROLMENT":
        return (
            "How do TMU students enrol in classes? Include new vs continuing student enrolment, course intentions, "
            "priority enrolment, and MyServiceHub if available."
        )
    if academic_intent == "COURSE_WAITLIST":
        return "What should TMU students do if a class is full? Include wait list details if available."
    if academic_intent == "COURSE_MANAGEMENT":
        return "How do TMU students add, drop, swap, or withdraw from courses in MyServiceHub?"
    if academic_intent == "PROGRAM_CHANGE":
        return "How can a TMU student switch programs or change majors/plans? Include internal change or transfer guidance if available."
    if academic_intent == "ADVISOR_CONTACT":
        if program:
            return (
                f"Who should a student in the {program} program at TMU Faculty of Arts contact for academic advising? "
                f"Include Faculty academic advising and program/department contacts if available."
            )
        return "Who should a TMU Faculty of Arts student contact for academic advising? Include Faculty academic advising and department contacts if available."
    if academic_intent == "IMPORTANT_DATES":
        return "What important TMU academic dates or course drop dates should students know this semester?"
    if academic_intent == "ACADEMIC_CONSIDERATION":
        return "What is a TMU academic consideration request and how do students submit one?"
    if academic_intent == "ACADEMIC_STANDING":
        return "What happens if a TMU student fails a course or is on academic probation or academic standing?"
    if academic_intent == "STUDENT_SUPPORT":
        if any(term in normalized for term in ("mental health", "counselling", "counseling")):
            return "What official mental health or counselling supports are available to TMU students?"
        return "What official TMU student support resources are available for students?"
    if academic_intent == "ADMISSIONS":
        return q
    if academic_intent == "GENERAL_STUDENT_INFO":
        if program and _is_referential_program_followup(q):
            return _resolve_program_referent(q, program)
        return q

    if program and _is_referential_program_followup(q):
        return _resolve_program_referent(q, program)
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
    if not state.program or not _can_apply_program_context(state):
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
        r"\bcurriculum\b",
    )
    return any(re.search(p, q) for p in patterns)


def _asks_for_undergraduate_programs(q: str) -> bool:
    return "undergraduate program" in q or "undergraduate programs" in q or bool(re.search(r"\blist all undergraduate programs\b", q))


def _asks_for_graduate_programs(q: str) -> bool:
    return "graduate program" in q or "graduate programs" in q or bool(re.search(r"\blist all graduate programs\b", q))


def _asks_about_departments(q: str) -> bool:
    return "department" in q or "departments" in q


def _asks_about_coop(q: str) -> bool:
    return any(token in q for token in ("co op", "coop", "internship", "work term", "work terms"))


def _asks_about_minor_or_certificate(q: str) -> bool:
    return any(token in q for token in (" minor", "minor ", "certificate", "concentration", "declare a minor"))


def _asks_about_admissions(q: str) -> bool:
    return any(token in q for token in ("apply", "application", "admission", "admissions", "requirements to apply"))


def _asks_about_student_support(q: str) -> bool:
    return any(token in q for token in ("mental health", "counselling", "counseling", "student support", "support services", "academic support", "wellbeing", "well being"))


def _asks_about_course_enrolment(q: str) -> bool:
    return any(token in q for token in ("enroll in classes", "enrol in classes", "enroll in courses", "enrol in courses", "course intentions", "priority enrollment", "priority enrolment"))


def _asks_about_waitlist_or_full_class(q: str) -> bool:
    return "wait list" in q or "waitlist" in q or "class is full" in q or "course is full" in q


def _asks_about_course_management(q: str) -> bool:
    return any(token in q for token in ("add, drop", "add drop", "swap courses", "swap a course", "drop a course", "withdraw from a course"))


def _asks_about_program_change(q: str) -> bool:
    return any(token in q for token in ("switch my program", "switch programs", "change my program", "change programs", "switch my major", "change my major"))


def _asks_about_advisor(q: str) -> bool:
    return any(token in q for token in ("who is my advisor", "who's my advisor", "who is my adviser", "advisor", "adviser", "advising"))


def _asks_about_course_planning(q: str) -> bool:
    return any(token in q for token in ("what courses should i pick", "what classes should i pick", "what courses should i take", "what classes should i take", "choose my electives", "pick my electives"))


def _asks_about_graduation_progress(q: str) -> bool:
    return any(token in q for token in ("on track to graduate", "graduate in 4 years", "track to graduate", "advisement report", "degree progress"))


def _asks_about_important_dates(q: str) -> bool:
    return "important dates" in q or "significant dates" in q or "deadline" in q or "drop dates" in q


def _asks_about_academic_consideration(q: str) -> bool:
    return "academic consideration" in q or "missed a test" in q or "missed an exam" in q


def _asks_about_academic_standing(q: str) -> bool:
    return any(token in q for token in ("fail a course", "failed a course", "what happens if i fail", "if i fail a class", "academic probation", "academic standing", "grades and standings"))


def _asks_for_program_overview(q: str) -> bool:
    return any(q.startswith(prefix) for prefix in ("tell me about", "what is", "what about", "how about", "give me an overview of", "overview of"))


def _is_program_scoped(q: str) -> bool:
    return any(token in q for token in ("course", "courses", "class", "classes", "elective", "electives", "required", "requirement", "requirements", "co op", "coop", "internship", "first year", "second year", "third year", "fourth year", "curriculum", "credit", "credits"))


def _is_supported_follow_up(q: str) -> bool:
    if any(q.startswith(marker) for marker in ("what if", "what about", "how about", "but can you", "and what about", "and can you", "what about for", "and for")):
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
    patterns = (r"\bthank(s| you)?\b", r"\bok(ay)? thanks\b", r"\bgot it\b", r"\bperfect\b", r"\bawesome\b", r"\bsounds good\b", r"\bappreciate it\b")
    return any(re.search(p, q) for p in patterns)


def _is_goodbye(q: str) -> bool:
    return q in {"bye", "goodbye", "see you", "see ya", "talk to you later"}


def _is_identity_question(q: str) -> bool:
    patterns = (r"\bare you (a )?(real )?bot\b", r"\bare you human\b", r"\bwho are you\b", r"\bwhat are you\b")
    return any(re.search(p, q) for p in patterns)


def _is_capability_question(q: str) -> bool:
    patterns = (r"\bwhat do you do\b", r"\bwhat can you do\b", r"\bhow can you help\b", r"\bwhat do you help with\b")
    return any(re.search(p, q) for p in patterns)


def _is_confusion(q: str) -> bool:
    patterns = (r"\bi do not know\b", r"\bi dont know\b", r"\bidk\b", r"\buh+\b", r"\bum+\b", r"\bumm+\b", r"\buhh+\b", r"\bhuh\b")
    if q in {"?", "what", "huh", "uh", "umm", "uhhh", "uhhhhmmm"}:
        return True
    return len(q.split()) <= 4 and any(re.search(p, q) for p in patterns)


def _is_emotional_reaction(raw_question: str, q: str) -> bool:
    patterns = (r"\bohh?\b", r"\bthat sucks\b", r"\bthat is disappointing\b", r"\bokay sad\b")
    return ":(" in raw_question or (len(q.split()) <= 5 and any(re.search(p, q) for p in patterns))


def _is_program_declaration(question: str) -> bool:
    q = _normalize(question)
    patterns = (r"\b(i am|i m|im|i am in|i m in|im in)\b", r"\bmy major is\b", r"\bmy program is\b", r"\bbut i am in\b", r"\bbut i m in\b", r"\bbut im in\b", r"\bactually\b", r"\bno\b")
    return any(re.search(p, q) for p in patterns) or bool(match_program(question) and len(q.split()) <= 6)


def _is_probably_on_domain_question(question: str, state: SessionState) -> bool:
    q = _normalize(question)
    if state.program and _is_supported_follow_up(q):
        return True
    domain_terms = (
        "tmu", "toronto metropolitan", "faculty of arts", "program", "major", "minor", "course", "courses", "class",
        "classes", "semester", "elective", "enroll", "enrol", "waitlist", "wait list", "advisor", "advising", "myservicehub",
        "curriculum", "requirements", "credits", "co op", "coop", "department", "departments", "graduate", "undergraduate",
        "academic", "probation", "consideration", "graduat", "support", "application", "admission", "admissions",
    )
    return any(term in q for term in domain_terms)


def _match_study_year(question: str) -> Optional[str]:
    q = _normalize(question)
    patterns = {
        "first year": r"\b(first year|1st year|year 1|year one)\b",
        "second year": r"\b(second year|2nd year|year 2|year two)\b",
        "third year": r"\b(third year|3rd year|year 3|year three)\b",
        "fourth year": r"\b(fourth year|4th year|year 4|year four)\b",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, q):
            return label
    return None


def _can_apply_program_context(state: SessionState) -> bool:
    return bool(state.program) and _program_context_age(state) <= _PROGRAM_CONTEXT_MAX_AGE


def _can_apply_year_context(state: SessionState) -> bool:
    return bool(state.study_year) and _year_context_age(state) <= _YEAR_CONTEXT_MAX_AGE


def _program_context_age(state: SessionState) -> int:
    last_program_turn = (state.metadata or {}).get("last_program_turn")
    if isinstance(last_program_turn, int):
        return max(0, state.turn_count - last_program_turn)
    return _PROGRAM_CONTEXT_MAX_AGE + 1


def _year_context_age(state: SessionState) -> int:
    last_year_turn = (state.metadata or {}).get("last_year_turn")
    if isinstance(last_year_turn, int):
        return max(0, state.turn_count - last_year_turn)
    return _YEAR_CONTEXT_MAX_AGE + 1


def _remember_program(state: SessionState, program: str) -> None:
    state.program = program
    state.metadata = dict(state.metadata or {})
    state.metadata["last_program_turn"] = state.turn_count


def _remember_study_year(state: SessionState, study_year: str) -> None:
    state.study_year = study_year
    state.metadata = dict(state.metadata or {})
    state.metadata["last_year_turn"] = state.turn_count


def _is_possessive_program_request(question: str) -> bool:
    q = _normalize(question)
    return any(phrase in q for phrase in ("my major", "my program"))


def _resume_program_intent(state: SessionState) -> Optional[str]:
    if state.last_intent in (_PROGRAM_REQUIRED_INTENTS | _PROGRAM_AND_YEAR_HELPFUL_INTENTS) and _can_apply_program_context(state):
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
