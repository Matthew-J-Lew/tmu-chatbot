from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional


ARTS_UNDERGRAD_PROGRAMS_URL = "https://www.torontomu.ca/arts/undergraduate/programs/"
ARTS_UNDERGRAD_PROGRAMS_SECTION = "Explore program options"
ARTS_DEPARTMENTS_URL = "https://www.torontomu.ca/arts/about/departments/"
ARTS_DEPARTMENTS_SECTION = "Departments"
ARTS_GRAD_PROGRAMS_URL = "https://www.torontomu.ca/arts/graduate/graduate-programs/"
ARTS_GRAD_PROGRAMS_SECTION = "Graduate programs"

_ARTS_UNDERGRAD_PROGRAMS: tuple[str, ...] = (
    "Arts and Contemporary Studies - BA (Hons)",
    "Criminology - BA (Hons)",
    "Economics and Finance - BA (Hons)",
    "English - BA (Hons)",
    "Environment and Urban Sustainability - BA (Hons)",
    "Geographic Analysis - BA (Hons)",
    "History - BA (Hons)",
    "Language and Intercultural Relations - BA (Hons)",
    "Philosophy - BA (Hons)",
    "Politics and Governance - BA (Hons)",
    "Psychology - BA (Hons)",
    "Public Administration and Governance - BA (Hons)",
    "Sociology - BA (Hons)",
    "Undeclared Arts - BA (Hons)",
)

_ARTS_DEPARTMENTS: tuple[str, ...] = (
    "Arts & Contemporary Studies",
    "Criminology",
    "Economics",
    "English",
    "Geography & Environmental Studies",
    "History",
    "Languages, Literatures & Cultures",
    "Philosophy",
    "Politics & Public Administration",
    "Psychology",
    "Sociology",
    "Undeclared Arts",
)

_ARTS_GRAD_PROGRAMS: tuple[str, ...] = (
    "Criminology and Social Justice MA",
    "Economics PhD",
    "Economics and Finance MA",
    "Literatures of Modernity MA",
    "Philosophy MA",
    "Policy Studies PhD",
    "Psychology: Clinical Stream MA & PhD",
    "Psychology: Psychological Science MA & PhD",
    "Public Policy & Administration MA",
    "Spatial Analysis MSA",
    "Youth and Student Development MA",
    "Immigration and Settlement Studies MA",
    "International Economics and Finance PhD",
)


@dataclass(frozen=True)
class CanonicalSource:
    url: str
    title: str
    section: Optional[str] = None


@dataclass(frozen=True)
class CanonicalAnswer:
    answer: str
    sources: List[CanonicalSource]


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _asks_for_list(q: str) -> bool:
    return any(token in q for token in ("list", "all", "every", "what are", "which are", "show"))


def _asks_for_count(q: str) -> bool:
    return "how many" in q or q.startswith("count ") or "number of" in q


def _contains_arts_context(q: str) -> bool:
    return any(token in q for token in ("faculty of arts", "arts", "tmu", "toronto metropolitan"))


def _is_arts_undergrad_program_query(question: str) -> bool:
    q = _normalize(question)
    if "graduate" in q:
        return False
    asks_programs = "undergraduate program" in q or "undergraduate programs" in q
    return asks_programs and (_asks_for_list(q) or _asks_for_count(q)) and _contains_arts_context(q)


def _is_arts_grad_program_query(question: str) -> bool:
    q = _normalize(question)
    asks_programs = "graduate program" in q or "graduate programs" in q
    return asks_programs and (_asks_for_list(q) or _asks_for_count(q)) and _contains_arts_context(q)


def _is_arts_department_query(question: str) -> bool:
    q = _normalize(question)
    asks_departments = "department" in q or "departments" in q
    return asks_departments and (_asks_for_list(q) or q.startswith("what departments") or q.startswith("which departments")) and _contains_arts_context(q)


def _enumerated_answer(intro: str, items: tuple[str, ...], source: CanonicalSource) -> CanonicalAnswer:
    lines = [intro, ""]
    for idx, name in enumerate(items, start=1):
        lines.append(f"{idx}. {name}")
    return CanonicalAnswer(answer="\n".join(lines), sources=[source])


def _undergrad_programs_list_answer() -> CanonicalAnswer:
    return _enumerated_answer(
        "The TMU Faculty of Arts undergraduate programs are:",
        _ARTS_UNDERGRAD_PROGRAMS,
        CanonicalSource(
            url=ARTS_UNDERGRAD_PROGRAMS_URL,
            title=ARTS_UNDERGRAD_PROGRAMS_URL,
            section=ARTS_UNDERGRAD_PROGRAMS_SECTION,
        ),
    )


def _undergrad_programs_count_answer() -> CanonicalAnswer:
    return CanonicalAnswer(
        answer=(
            f"The TMU Faculty of Arts has {len(_ARTS_UNDERGRAD_PROGRAMS)} undergraduate programs "
            f"listed on its undergraduate programs page."
        ),
        sources=[CanonicalSource(
            url=ARTS_UNDERGRAD_PROGRAMS_URL,
            title=ARTS_UNDERGRAD_PROGRAMS_URL,
            section=ARTS_UNDERGRAD_PROGRAMS_SECTION,
        )],
    )


def _graduate_programs_list_answer() -> CanonicalAnswer:
    return _enumerated_answer(
        "The TMU Faculty of Arts graduate programs are:",
        _ARTS_GRAD_PROGRAMS,
        CanonicalSource(
            url=ARTS_GRAD_PROGRAMS_URL,
            title=ARTS_GRAD_PROGRAMS_URL,
            section=ARTS_GRAD_PROGRAMS_SECTION,
        ),
    )


def _graduate_programs_count_answer() -> CanonicalAnswer:
    return CanonicalAnswer(
        answer=(
            f"The TMU Faculty of Arts has {len(_ARTS_GRAD_PROGRAMS)} graduate programs "
            f"listed in its graduate programs information."
        ),
        sources=[CanonicalSource(
            url=ARTS_GRAD_PROGRAMS_URL,
            title=ARTS_GRAD_PROGRAMS_URL,
            section=ARTS_GRAD_PROGRAMS_SECTION,
        )],
    )


def _departments_list_answer() -> CanonicalAnswer:
    return _enumerated_answer(
        "The TMU Faculty of Arts departments are:",
        _ARTS_DEPARTMENTS,
        CanonicalSource(
            url=ARTS_DEPARTMENTS_URL,
            title=ARTS_DEPARTMENTS_URL,
            section=ARTS_DEPARTMENTS_SECTION,
        ),
    )


def maybe_answer_canonical_finite_question(question: str, policy_label: Optional[str]) -> Optional[CanonicalAnswer]:
    q = _normalize(question)

    if policy_label == "ARTS_UNDERGRAD_PROGRAM_LIST" or _is_arts_undergrad_program_query(question):
        if _asks_for_count(q):
            return _undergrad_programs_count_answer()
        if _asks_for_list(q):
            return _undergrad_programs_list_answer()

    if policy_label == "ARTS_GRAD_PROGRAM_LIST" or _is_arts_grad_program_query(question):
        if _asks_for_count(q):
            return _graduate_programs_count_answer()
        if _asks_for_list(q):
            return _graduate_programs_list_answer()

    if policy_label == "ARTS_DEPARTMENTS_LIST" or _is_arts_department_query(question):
        if _asks_for_list(q) or q.startswith("what departments") or q.startswith("which departments"):
            return _departments_list_answer()

    return None
