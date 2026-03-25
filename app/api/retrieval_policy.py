from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Tuple


@dataclass(frozen=True)
class RetrievalPolicy:
    label: str = "DEFAULT"
    retrieval_query: Optional[str] = None
    preferred_urls: Tuple[str, ...] = ()
    discouraged_urls: Tuple[str, ...] = ()
    same_source_limit: int = 1
    canonical_fallback: bool = False

    def cache_token(self) -> str:
        return self.label


_DEFAULT = RetrievalPolicy()


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_arts_context(q: str) -> bool:
    return any(token in q for token in ("faculty of arts", "arts", "tmu", "toronto metropolitan"))


def _asks_for_list(q: str) -> bool:
    return any(token in q for token in ("list", "show", "every", "all", "what are", "which are"))


def _asks_for_count(q: str) -> bool:
    return "how many" in q or "number of" in q or "count of" in q


def _is_arts_undergrad_program_list(q: str) -> bool:
    if re.search(r"\bgraduate\b", q):
        return False
    asks_programs = "undergraduate program" in q or "undergraduate programs" in q
    return asks_programs and (_asks_for_list(q) or _asks_for_count(q)) and _contains_arts_context(q)


def _is_arts_grad_program_list(q: str) -> bool:
    asks_programs = "graduate program" in q or "graduate programs" in q
    return asks_programs and (_asks_for_list(q) or _asks_for_count(q)) and _contains_arts_context(q)


def _is_department_list(q: str) -> bool:
    asks_departments = "department" in q or "departments" in q
    return asks_departments and (_asks_for_list(q) or _asks_for_count(q) or q.startswith("what departments") or q.startswith("which departments")) and _contains_arts_context(q)


def _is_minor_declaration_question(q: str) -> bool:
    return (("declare a minor" in q or "declaring a minor" in q or "apply for a minor" in q)
            or ("minor" in q and any(token in q for token in ("declare", "declaring", "apply", "select", "myservicehub"))))


def _is_course_enrolment_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "enroll in classes", "enrol in classes", "enroll in courses", "enrol in courses",
        "class enrollment", "course enrollment", "course enrolment", "course intentions",
        "priority enrollment", "priority enrolment",
    ))


def _is_waitlist_question(q: str) -> bool:
    return "wait list" in q or "waitlist" in q or "class is full" in q or "course is full" in q


def _is_course_management_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "add drop or swap", "add drop swap", "add drop or withdraw", "drop a course",
        "drop or swap", "swap courses", "swap a course", "withdraw from a course",
    ))


def _is_program_change_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "switch my program", "switch programs", "change my program", "change programs",
        "switch my major", "change my major", "internal transfer", "transfer programs",
    ))


def _is_advisor_contact_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "who is my advisor", "who s my advisor", "who is my adviser", "who should i talk to",
        "who can i talk to", "academic advisor", "academic advising", "department advisor",
    ))


def _is_course_planning_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "what courses should i pick", "what classes should i pick", "what courses should i take",
        "what classes should i take", "plan my courses", "choose my electives", "pick my electives",
    ))


def _is_graduation_progress_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "on track to graduate", "track to graduate", "graduate in 4 years", "graduate on time",
        "advisement report", "degree progress", "graduation progress",
    ))


def _is_academic_consideration_question(q: str) -> bool:
    return "academic consideration" in q or "missed a test" in q or "missed an exam" in q


def _is_academic_standing_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "fail a course", "failed a course", "what happens if i fail", "if i fail a class",
        "academic probation", "academic standing", "grades and standings",
    ))


def _is_important_dates_question(q: str) -> bool:
    return "important dates" in q or "significant dates" in q or "course drop dates" in q or "deadline" in q


def _is_student_support_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "mental health", "counselling", "counseling", "student support", "support services",
        "academic support", "wellbeing", "well being",
    ))


def choose_retrieval_policy(raw_question: str, effective_question: str) -> RetrievalPolicy:
    raw = _normalize(raw_question)
    eff = _normalize(effective_question)
    combined = f"{raw} || {eff}"

    if _is_arts_undergrad_program_list(combined):
        return RetrievalPolicy(
            label="ARTS_UNDERGRAD_PROGRAM_LIST",
            retrieval_query="TMU Faculty of Arts undergraduate programs. List every undergraduate program name.",
            preferred_urls=("/arts/undergraduate/programs", "/arts/undergraduate"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/", "/curriculum-advising/", "/career-coop-student-success/"),
            same_source_limit=3,
            canonical_fallback=True,
        )

    if _is_arts_grad_program_list(combined):
        return RetrievalPolicy(
            label="ARTS_GRAD_PROGRAM_LIST",
            retrieval_query="TMU Faculty of Arts graduate programs. List every graduate program name.",
            preferred_urls=("/arts/graduate/graduate-programs", "/arts/graduate"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/", "/curriculum-advising/"),
            same_source_limit=3,
            canonical_fallback=True,
        )

    if _is_department_list(combined):
        return RetrievalPolicy(
            label="ARTS_DEPARTMENTS_LIST",
            retrieval_query="List the departments in TMU Faculty of Arts.",
            preferred_urls=("/arts/about/departments", "/arts/undergraduate/programs"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/"),
            same_source_limit=3,
            canonical_fallback=True,
        )

    if _is_minor_declaration_question(combined):
        return RetrievalPolicy(
            label="MINOR_DECLARATION",
            retrieval_query="How do TMU students declare a minor? Include Application to Graduate and MyServiceHub details if available.",
            preferred_urls=("/curriculum-advising/curriculum-requirements/program-requirements", "/myservicehub-support/students/academics", "/curriculum-advising/curriculum-requirements/"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/", "/arts/graduate/"),
            same_source_limit=2,
        )

    if _is_course_enrolment_question(combined):
        return RetrievalPolicy(
            label="COURSE_ENROLMENT",
            retrieval_query="How do TMU students enroll in classes? Include new vs continuing student enrolment, course intentions, priority enrolment, and MyServiceHub if available.",
            preferred_urls=("/current-students/course-enrolment", "/myservicehub-support/students/academics"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/apply/"),
            same_source_limit=2,
        )

    if _is_waitlist_question(combined):
        return RetrievalPolicy(
            label="COURSE_WAITLIST",
            retrieval_query="What should TMU students do if a class is full? Include wait list guidance if available.",
            preferred_urls=("/current-students/course-enrolment/wait-list", "/current-students/course-enrolment"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_course_management_question(combined):
        return RetrievalPolicy(
            label="COURSE_MANAGEMENT",
            retrieval_query="How do TMU students add, drop, swap, or withdraw from courses in MyServiceHub?",
            preferred_urls=("/myservicehub-support/students/academics", "/current-students/course-enrolment/drops-withdrawals", "/current-students/course-enrolment"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_program_change_question(combined):
        return RetrievalPolicy(
            label="PROGRAM_CHANGE",
            retrieval_query="How can a TMU student switch programs or change majors or plans? Include internal change or transfer guidance if available.",
            preferred_urls=("/admissions/undergraduate/requirements/transfer-student", "/myservicehub-support/students/academics", "/curriculum-advising/"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_advisor_contact_question(combined):
        return RetrievalPolicy(
            label="ADVISOR_CONTACT",
            retrieval_query="Who should a TMU Faculty of Arts student contact for academic advising? Include Faculty academic advising and department contacts if available.",
            preferred_urls=("/arts/undergraduate/academic-support/academic-advising", "/arts/about/departments", "/arts/undergraduate/academic-support"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_course_planning_question(combined):
        return RetrievalPolicy(
            label="COURSE_PLANNING",
            retrieval_query="How should a TMU Faculty of Arts student plan courses? Include Advisement Report, academic advising, and program requirements guidance if available.",
            preferred_urls=("/myservicehub-support/students/academics/advisement-report", "/arts/undergraduate/academic-support/academic-advising", "/curriculum-advising/curriculum-requirements/program-requirements"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_graduation_progress_question(combined):
        return RetrievalPolicy(
            label="GRADUATION_PROGRESS",
            retrieval_query="How can a TMU student check if they are on track to graduate? Include Advisement Report and advising resources.",
            preferred_urls=("/myservicehub-support/students/academics/advisement-report", "/arts/undergraduate/academic-support/academic-advising"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_academic_consideration_question(combined):
        return RetrievalPolicy(
            label="ACADEMIC_CONSIDERATION",
            retrieval_query="What is a TMU academic consideration request and how do students submit one?",
            preferred_urls=("/current-students/academic-consideration", "/arts/undergraduate/academic-support"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_academic_standing_question(combined):
        return RetrievalPolicy(
            label="ACADEMIC_STANDING",
            retrieval_query="What happens if a TMU student fails a course or is on academic probation or academic standing?",
            preferred_urls=("/arts/undergraduate/academic-support/academic-grades-and-standings", "/curriculum-advising/"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_important_dates_question(combined):
        return RetrievalPolicy(
            label="IMPORTANT_DATES",
            retrieval_query="What important TMU academic dates or course drop dates should students know?",
            preferred_urls=("/calendar/2025-2026/dates", "/calendar/2026-2027/dates", "/current-students/course-enrolment/drops-withdrawals/drop-course", "/admissions/undergraduate/apply/application-dates"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_student_support_question(combined):
        return RetrievalPolicy(
            label="STUDENT_SUPPORT",
            retrieval_query="What official TMU student support resources are available? Prioritize academic advising, counselling, and support services.",
            preferred_urls=("/arts/undergraduate/academic-support", "/current-students/academic-consideration"),
            discouraged_urls=("/admissions/undergraduate/apply/",),
            same_source_limit=2,
        )

    return _DEFAULT
