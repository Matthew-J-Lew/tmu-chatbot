from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ProgramRecord:
    canonical_name: str
    aliases: tuple[str, ...]


_PROGRAMS: tuple[ProgramRecord, ...] = (
    ProgramRecord(
        canonical_name="Arts and Contemporary Studies",
        aliases=(
            "arts and contemporary studies",
            "arts contemporary studies",
            "acs",
            "arts and contemporary studies ba",
            "arts and contemporary studies ba hons",
        ),
    ),
    ProgramRecord(
        canonical_name="Criminology",
        aliases=("criminology", "criminology ba", "criminology ba hons"),
    ),
    ProgramRecord(
        canonical_name="Economics and Finance",
        aliases=(
            "economics and finance",
            "economics & finance",
            "econ and finance",
            "economics finance",
            "economics and finance ba",
            "economics and finance ba hons",
        ),
    ),
    ProgramRecord(
        canonical_name="English",
        aliases=("english", "english ba", "english ba hons"),
    ),
    ProgramRecord(
        canonical_name="Environmental and Urban Sustainability",
        aliases=(
            "environmental and urban sustainability",
            "environment and urban sustainability",
            "environment urban sustainability",
            "eus",
            "environmental and urban sustainability ba",
            "environmental and urban sustainability ba hons",
        ),
    ),
    ProgramRecord(
        canonical_name="Geographic Analysis",
        aliases=(
            "geographic analysis",
            "geography",
            "geographic analysis ba",
            "geographic analysis ba hons",
        ),
    ),
    ProgramRecord(
        canonical_name="History",
        aliases=("history", "history ba", "history ba hons"),
    ),
    ProgramRecord(
        canonical_name="Language and Intercultural Relations",
        aliases=(
            "language and intercultural relations",
            "languages and intercultural relations",
            "language intercultural relations",
            "lir",
            "language and intercultural relations ba",
            "language and intercultural relations ba hons",
        ),
    ),
    ProgramRecord(
        canonical_name="Philosophy",
        aliases=("philosophy", "philosophy ba", "philosophy ba hons"),
    ),
    ProgramRecord(
        canonical_name="Politics and Governance",
        aliases=(
            "politics and governance",
            "politics governance",
            "politics",
            "politics and governance ba",
            "politics and governance ba hons",
        ),
    ),
    ProgramRecord(
        canonical_name="Psychology",
        aliases=("psychology", "psychology ba", "psychology ba hons"),
    ),
    ProgramRecord(
        canonical_name="Public Administration and Governance",
        aliases=(
            "public administration and governance",
            "public administration",
            "pag",
            "public administration and governance ba",
            "public administration and governance ba hons",
        ),
    ),
    ProgramRecord(
        canonical_name="Sociology",
        aliases=("sociology", "sociology ba", "sociology ba hons"),
    ),
    ProgramRecord(
        canonical_name="Undeclared Arts",
        aliases=(
            "undeclared arts",
            "undeclared",
            "undeclared arts first year",
        ),
    ),
)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\b(honours|honors|hons?)\b", "", text)
    text = re.sub(r"\b(bachelor of arts|ba)\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _all_names(program: ProgramRecord) -> Iterable[str]:
    yield _normalize(program.canonical_name)
    for alias in program.aliases:
        yield _normalize(alias)


def list_programs() -> List[str]:
    return [p.canonical_name for p in _PROGRAMS]


def match_program(text: str) -> Optional[str]:
    """Return the canonical program name if the text clearly names one.

    Guardrails:
    - Prefer exact/substring matches over fuzzy matching.
    - Only use fuzzy matching for short replies (common in slot-filling turns).
    """
    normalized = _normalize(text)
    if not normalized:
        return None

    for program in _PROGRAMS:
        for name in _all_names(program):
            if normalized == name:
                return program.canonical_name

    for program in _PROGRAMS:
        for name in _all_names(program):
            if re.search(rf"\b{re.escape(name)}\b", normalized):
                return program.canonical_name

    word_count = len(normalized.split())
    if word_count > 5:
        return None

    best_name: Optional[str] = None
    best_score = 0.0
    for program in _PROGRAMS:
        for name in _all_names(program):
            score = SequenceMatcher(a=normalized, b=name).ratio()
            if score > best_score:
                best_score = score
                best_name = program.canonical_name

    if best_name and best_score >= 0.86:
        return best_name
    return None
