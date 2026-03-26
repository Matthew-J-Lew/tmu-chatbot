from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Optional, Tuple

from app.api.program_registry import match_program


_PROGRAM_CALENDAR_SLUG_ALIASES = {
    "Arts and Contemporary Studies": ("arts_contemporary_studies", "arts-contemporary-studies"),
    "Criminology": ("criminology",),
    "Economics and Finance": ("economics_finance", "economics-and-finance"),
    "English": ("english",),
    "Environmental and Urban Sustainability": ("environment_urban_sustainability", "environmental_and_urban_sustainability"),
    "Geographic Analysis": ("geographic_analysis",),
    "History": ("history",),
    "Language and Intercultural Relations": ("language_intercultural_relations",),
    "Philosophy": ("philosophy",),
    "Politics and Governance": ("politics", "politics_governance"),
    "Psychology": ("psychology",),
    "Public Administration and Governance": ("public_admin", "public_administration", "public_administration_governance"),
    "Sociology": ("sociology",),
    "Undeclared Arts": ("undeclared_arts",),
}


@dataclass(frozen=True)
class RetrievalPolicy:
    label: str = "DEFAULT"
    retrieval_query: Optional[str] = None
    preferred_urls: Tuple[str, ...] = ()
    discouraged_urls: Tuple[str, ...] = ()
    preferred_section_terms: Tuple[str, ...] = ()
    discouraged_section_terms: Tuple[str, ...] = ()
    same_source_limit: int = 1
    canonical_fallback: bool = False
    program_slug: Optional[str] = None
    program_slug_aliases: Tuple[str, ...] = ()

    def cache_token(self) -> str:
        slug_token = f":{self.program_slug}" if self.program_slug else ""
        return f"{self.label}{slug_token}"


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


def _program_slug_aliases(raw_question: str, effective_question: str) -> Tuple[str, ...]:
    for candidate in (effective_question, raw_question):
        program = match_program(candidate)
        if program:
            return _PROGRAM_CALENDAR_SLUG_ALIASES.get(program, ())
    return ()


def _extract_program_slug(raw_question: str, effective_question: str) -> Optional[str]:
    aliases = _program_slug_aliases(raw_question, effective_question)
    return aliases[0] if aliases else None


def _extract_study_year(raw_question: str, effective_question: str) -> Optional[str]:
    combined = _normalize(f"{raw_question} {effective_question}")
    mapping = {
        "first year": (r"\b(first year|1st year|year 1)\b",),
        "second year": (r"\b(second year|2nd year|year 2)\b",),
        "third year": (r"\b(third year|3rd year|year 3)\b",),
        "fourth year": (r"\b(fourth year|4th year|year 4)\b",),
    }
    for label, patterns in mapping.items():
        if any(re.search(p, combined) for p in patterns):
            return label
    return None


def _mentions_coop(raw_question: str, effective_question: str) -> bool:
    combined = _normalize(f"{raw_question} {effective_question}")
    return any(token in combined for token in ("co op", "coop", "co operative", "five year co op", "five year coop"))


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
            or ("minor" in q and any(token in q for token in ("declare", "declaring", "apply", "select", "myservicehub", "pick", "choose"))))


def _is_accommodations_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "academic accommodations", "academic accommodation", "accommodations", "accommodation support",
        "how do i get accommodations", "how do i get accommodation", "register for accommodations",
        "accommodation request", "acr",
    ))


def _is_course_intentions_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "course intentions", "course intentions period", "miss the course intentions period",
        "missed the course intentions period", "miss the course intention period",
        "priority enrollment", "priority enrolment", "priority enrolment period",
        "priority enrollment period",
    ))


def _is_course_enrolment_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "enroll in a class", "enrol in a class", "enroll in class", "enrol in class",
        "enroll in classes", "enrol in classes", "enroll in a course", "enrol in a course",
        "enroll in courses", "enrol in courses", "sign up for classes", "sign up for courses",
        "class enrollment", "class enrolment", "course enrollment", "course enrolment",
    ))


def _is_waitlist_question(q: str) -> bool:
    return "wait list" in q or "waitlist" in q or "class is full" in q or "course is full" in q


def _is_course_management_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "add drop or swap", "add drop swap", "add drop or withdraw", "add a class",
        "drop a class", "drop a course", "drop or swap", "swap classes", "swap courses",
        "swap a class", "swap a course", "withdraw from a class", "withdraw from a course",
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
        "what classes should i take", "what should i take", "plan my courses", "choose my electives",
        "pick my electives", "pick classes", "pick courses", "first year courses", "second year courses",
        "third year courses", "fourth year courses", "first year classes", "second year classes",
        "third year classes", "fourth year classes",
    ))


def _is_program_requirements_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "required classes", "required courses", "degree requirements", "course requirements",
        "what courses do i need", "what classes do i need", "curriculum", "list the required courses",
        "list all the required courses", "what are my required courses", "what are the required courses",
    ))


def _is_graduation_progress_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "on track to graduate", "track to graduate", "graduate in 4 years", "graduate on time",
        "advisement report", "degree progress", "graduation progress",
    ))


def _is_missed_assessment_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "missed a test", "miss a test", "missed test", "miss a quiz", "missed a quiz",
        "missed quiz", "missed an exam", "miss an exam", "missed exam",
        "miss a midterm", "missed a midterm", "missed midterm",
    ))


def _is_academic_consideration_question(q: str) -> bool:
    return "academic consideration" in q or "consideration request" in q


def _is_gpa_standing_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "my gpa falls", "gpa falls", "gpa drops", "gpa drop", "my gpa drops",
        "low gpa", "gpa is too low", "gpa gets too low",
    ))


def _is_academic_standing_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "fail a course", "failed a course", "what happens if i fail", "if i fail a class",
        "academic probation", "academic standing", "grades and standings", "on probation",
        "i am on probation", "im on probation", "i'm on probation",
    ))


def _resolve_current_term_context(now: Optional[datetime] = None) -> Tuple[str, str, str, str]:
    now = now or datetime.now()
    year = now.year
    month = now.month
    if month <= 4:
        term_label = f"Winter {year}"
        term_token = "winter"
        calendar_year = f"{year - 1}-{year}"
    elif month <= 8:
        term_label = f"Spring/Summer {year}"
        term_token = "spring/summer"
        calendar_year = f"{year - 1}-{year}"
    else:
        term_label = f"Fall {year}"
        term_token = "fall"
        calendar_year = f"{year}-{year + 1}"
    calendar_path = f"/calendar/{calendar_year}/dates"
    return term_label, term_token, calendar_year, calendar_path


def _is_exam_dates_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "exam period", "final exam period", "exam dates", "final exam dates",
        "when are exams", "when is exams", "when is the exam period", "when are the exams",
    ))


def _is_important_dates_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "important dates", "significant dates", "course drop dates", "deadline",
    ))


def _is_mental_health_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "mental health", "counselling", "counseling", "counsellor", "counselor",
        "in crisis", "counselling appointment", "booking an appointment",
    ))


def _is_student_support_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "student support", "support services", "academic support", "wellbeing", "well being",
        "student wellbeing", "support resources",
    )) or _is_mental_health_question(q)


def _is_chang_school_credit_question(q: str) -> bool:
    return "chang school" in q and any(
        phrase in q for phrase in (
            "count towards my degree", "count toward my degree", "count for my degree",
            "count towards the degree", "count toward the degree", "degree credit",
            "open elective", "program requirement", "transfer credit",
        )
    )


def _is_program_overview_question(q: str) -> bool:
    return any(phrase in q for phrase in (
        "what can you tell me about", "can you tell me about", "tell me about",
        "tell me more about", "overview of", "give me an overview of",
        "what is the", "what is ", "what about", "how about",
    )) and "program" in q


def _calendar_policy(label: str, effective_question: str, slug: str, *, slug_aliases: Tuple[str, ...] = (), prefer_coop: bool = False) -> RetrievalPolicy:
    slug_aliases = tuple(dict.fromkeys((slug,) + tuple(slug_aliases or ())))
    preferred_urls = []
    for candidate_slug in slug_aliases:
        base_path_2025 = f"/calendar/2025-2026/programs/arts/{candidate_slug}"
        base_path_2026 = f"/calendar/2026-2027/programs/arts/{candidate_slug}"
        preferred_urls.extend([
            base_path_2025,
            base_path_2026,
            f"{base_path_2025}/table_i",
            f"{base_path_2025}/table_ii",
            f"{base_path_2026}/table_i",
            f"{base_path_2026}/table_ii",
        ])

    if label == "COURSE_PLANNING_CALENDAR":
        preferred_sections = [
            "table i",
            "table ii",
            "required group",
            "core elective",
            "full-time, four-year program",
            "liberal studies",
            "open electives",
        ]
        discouraged_sections = ["program overview/curriculum information"]
        if prefer_coop:
            preferred_sections.insert(5, "full-time, five-year co-op program")
        else:
            discouraged_sections.append("full-time, five-year co-op program")
        same_source_limit = 3
    else:
        preferred_sections = [
            "full-time, four-year program",
            "program overview/curriculum information",
            "table i",
            "table ii",
            "required group",
            "core elective",
            "liberal studies",
            "open electives",
        ]
        if prefer_coop:
            preferred_sections.insert(2, "full-time, five-year co-op program")
            discouraged_sections = []
        else:
            discouraged_sections = ["full-time, five-year co-op program"]
        same_source_limit = 4

    return RetrievalPolicy(
        label=label,
        retrieval_query=effective_question,
        preferred_urls=tuple(preferred_urls),
        discouraged_urls=(
            "/myservicehub-support/students/academics/advisement-report",
            "/arts/undergraduate/academic-support/academic-advising",
            "/arts/undergraduate/academic-support/",
            "/admissions/undergraduate/",
            "/student-financial-assistance/",
            "/career-coop-student-success/",
        ),
        preferred_section_terms=tuple(preferred_sections),
        discouraged_section_terms=tuple(discouraged_sections),
        same_source_limit=same_source_limit,
        program_slug=slug,
        program_slug_aliases=slug_aliases,
    )


def choose_retrieval_policy(raw_question: str, effective_question: str) -> RetrievalPolicy:
    raw = _normalize(raw_question)
    eff = _normalize(effective_question)
    combined = f"{raw} || {eff}"
    program_slug = _extract_program_slug(raw_question, effective_question)
    program_slug_aliases = _program_slug_aliases(raw_question, effective_question)
    study_year = _extract_study_year(raw_question, effective_question)

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

    if _is_course_intentions_question(combined):
        return RetrievalPolicy(
            label="COURSE_INTENTIONS",
            retrieval_query="What should a TMU student do if they miss the course intentions period? Include course intentions timing, later enrolment windows, and MyServiceHub if available.",
            preferred_urls=("/current-students/course-enrolment/course-intentions", "/current-students/course-enrolment", "/myservicehub-support/students/academics"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/apply/"),
            same_source_limit=2,
        )

    if _is_course_enrolment_question(combined):
        return RetrievalPolicy(
            label="COURSE_ENROLMENT",
            retrieval_query="How do TMU students enroll in classes? Include new vs continuing student enrolment and MyServiceHub if available.",
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
            retrieval_query="Who should a TMU Faculty of Arts student contact for academic advising? Prioritize the Faculty of Arts academic advising page and official undergraduate Arts advising contacts only.",
            preferred_urls=("/arts/undergraduate/academic-support/academic-advising", "/arts/undergraduate/academic-support"),
            discouraged_urls=("/student-financial-assistance/", "/programs/undergraduate/", "/engineering/", "/science/", "/the-chang-school/", "/arts/about/departments"),
            preferred_section_terms=("academic advising", "contact", "undergraduate", "arts"),
            discouraged_section_terms=("engineering", "chemical engineering", "graduate", "department chair"),
            same_source_limit=1,
        )

    if _is_chang_school_credit_question(combined):
        return RetrievalPolicy(
            label="CHANG_SCHOOL_CREDIT",
            retrieval_query="Can TMU students take a Chang School class and have it count toward their degree? Include open elective or program requirement caveats if available.",
            preferred_urls=("/the-chang-school/", "/curriculum-advising/curriculum-requirements/program-requirements", "/myservicehub-support/students/academics"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_course_planning_question(combined) and program_slug and study_year:
        return _calendar_policy(
            "COURSE_PLANNING_CALENDAR",
            effective_question,
            program_slug,
            slug_aliases=program_slug_aliases,
            prefer_coop=_mentions_coop(raw_question, effective_question),
        )

    if _is_program_overview_question(combined) and program_slug:
        return RetrievalPolicy(
            label="PROGRAM_OVERVIEW",
            retrieval_query=effective_question,
            preferred_urls=(
                f"/calendar/2025-2026/programs/arts/{program_slug}",
                f"/calendar/2026-2027/programs/arts/{program_slug}",
                "/arts/undergraduate/programs",
            ),
            preferred_section_terms=("program overview", "curriculum information", "overview"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
            program_slug=program_slug,
            program_slug_aliases=program_slug_aliases,
        )

    if _is_program_requirements_question(combined) and program_slug:
        return _calendar_policy(
            "PROGRAM_REQUIREMENTS_CALENDAR",
            effective_question,
            program_slug,
            slug_aliases=program_slug_aliases,
            prefer_coop=_mentions_coop(raw_question, effective_question),
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

    if _is_missed_assessment_question(combined):
        return RetrievalPolicy(
            label="MISSED_ASSESSMENT",
            retrieval_query="What should a TMU student do if they miss a test, quiz, midterm, or exam? Include academic consideration guidance if available.",
            preferred_urls=("/current-students/academic-consideration", "/arts/undergraduate/academic-support"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/"),
            preferred_section_terms=("academic consideration", "missed test", "missed exam", "support"),
            same_source_limit=2,
        )

    if _is_academic_consideration_question(combined):
        return RetrievalPolicy(
            label="ACADEMIC_CONSIDERATION",
            retrieval_query="What is a TMU academic consideration request and how do students submit one?",
            preferred_urls=("/current-students/academic-consideration", "/arts/undergraduate/academic-support"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/"),
            preferred_section_terms=("academic consideration", "request", "submit"),
            same_source_limit=2,
        )

    if _is_accommodations_question(combined):
        return RetrievalPolicy(
            label="ACADEMIC_ACCOMMODATIONS",
            retrieval_query="What academic accommodations are available at TMU and how do students register or get them? Prioritize official TMU accommodations pages and registration guidance.",
            preferred_urls=("/accommodations/", "/student-wellbeing/"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/", "/current-students/academic-consideration"),
            preferred_section_terms=("academic accommodation", "register", "how to get accommodations", "accommodation support"),
            same_source_limit=2,
        )

    if _is_gpa_standing_question(combined):
        return RetrievalPolicy(
            label="GPA_STANDING",
            retrieval_query="What happens if a TMU student's GPA falls? Include academic standing and probation guidance if available.",
            preferred_urls=("/arts/undergraduate/academic-support/academic-grades-and-standings", "/curriculum-advising/"),
            discouraged_urls=("/student-financial-assistance/",),
            same_source_limit=2,
        )

    if _is_academic_standing_question(combined):
        return RetrievalPolicy(
            label="ACADEMIC_STANDING",
            retrieval_query="What happens if a TMU student fails a course or is on academic probation or academic standing? Include next-step support resources such as academic coaches if available.",
            preferred_urls=("/arts/undergraduate/academic-support/academic-grades-and-standings", "/arts/undergraduate/student-experience/academic-coaches", "/curriculum-advising/"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/"),
            preferred_section_terms=("probation", "academic standing", "academic coaches"),
            same_source_limit=2,
        )

    if _is_exam_dates_question(combined):
        term_label, term_token, _calendar_year, calendar_path = _resolve_current_term_context()
        retrieval_query = "When is the TMU exam period and where are final exam dates listed?"
        preferred_section_terms = ("exam period", "final exams", "significant dates")
        if "this semester" in combined or "this term" in combined:
            retrieval_query = f"When is the TMU exam period for {term_label}? Prioritize the significant dates page for that semester and the official final exam period listing."
            preferred_section_terms = (term_token, term_label.lower(), "exam period", "final exams", "significant dates")
        return RetrievalPolicy(
            label="EXAM_DATES",
            retrieval_query=retrieval_query,
            preferred_urls=(calendar_path, "/calendar/2025-2026/dates", "/calendar/2026-2027/dates"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/apply/application-dates"),
            preferred_section_terms=preferred_section_terms,
            same_source_limit=2,
        )

    if _is_important_dates_question(combined):
        term_label, term_token, _calendar_year, calendar_path = _resolve_current_term_context()
        retrieval_query = "What important TMU academic dates or course drop dates should students know?"
        preferred_section_terms = ("significant dates", "drop course", "drop deadline")
        if "this semester" in combined or "this term" in combined:
            retrieval_query = f"What are the important TMU academic dates for {term_label}? Prioritize the official significant dates calendar for that semester and the course drop deadlines for that term."
            preferred_section_terms = (term_token, term_label.lower(), "significant dates", "drop course", "drop deadline")
        return RetrievalPolicy(
            label="IMPORTANT_DATES",
            retrieval_query=retrieval_query,
            preferred_urls=(calendar_path, "/current-students/course-enrolment/drops-withdrawals/drop-course", "/calendar/2025-2026/dates", "/calendar/2026-2027/dates"),
            discouraged_urls=("/student-financial-assistance/", "/admissions/undergraduate/apply/application-dates"),
            preferred_section_terms=preferred_section_terms,
            same_source_limit=2,
        )

    if _is_mental_health_question(combined):
        return RetrievalPolicy(
            label="MENTAL_HEALTH_SUPPORT",
            retrieval_query="What TMU mental health and counselling resources are available, including crisis help and how to book counselling if relevant?",
            preferred_urls=("/student-wellbeing/counselling/", "/student-wellbeing/"),
            discouraged_urls=("/admissions/undergraduate/", "/student-financial-assistance/", "/current-students/academic-consideration"),
            preferred_section_terms=("counselling", "crisis", "booking an appointment", "mental health"),
            same_source_limit=2,
        )

    if _is_student_support_question(combined):
        return RetrievalPolicy(
            label="STUDENT_SUPPORT",
            retrieval_query="What official TMU student support resources are available? Prioritize academic advising, counselling, academic support, and official support services.",
            preferred_urls=("/arts/undergraduate/academic-support", "/student-wellbeing/", "/current-students/academic-consideration"),
            discouraged_urls=("/admissions/undergraduate/apply/", "/student-financial-assistance/"),
            preferred_section_terms=("support", "academic advising", "counselling", "student wellbeing"),
            same_source_limit=2,
        )

    return _DEFAULT
