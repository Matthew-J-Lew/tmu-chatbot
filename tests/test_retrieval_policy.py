import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.api.canonical_facts import maybe_answer_canonical_finite_question
from app.api.retrieval_policy import choose_retrieval_policy


def test_undergrad_program_list_policy_and_canonical_answer():
    raw = "Can you list all undergraduate programs?"
    effective = "List every undergraduate program offered by the TMU Faculty of Arts."
    policy = choose_retrieval_policy(raw, effective)

    assert policy.label == "ARTS_UNDERGRAD_PROGRAM_LIST"
    assert policy.canonical_fallback is True
    assert "/arts/undergraduate/programs" in policy.preferred_urls
    assert policy.same_source_limit == 3

    canonical = maybe_answer_canonical_finite_question(raw, policy.label)
    assert canonical is not None
    assert "TMU Faculty of Arts undergraduate programs" in canonical.answer
    assert "1. Arts and Contemporary Studies - BA (Hons)" in canonical.answer
    assert "14. Undeclared Arts - BA (Hons)" in canonical.answer


def test_departments_policy_and_canonical_answer():
    raw = "What departments are in the Faculty of Arts?"
    policy = choose_retrieval_policy(raw, raw)

    assert policy.label == "ARTS_DEPARTMENTS_LIST"
    assert policy.canonical_fallback is True
    canonical = maybe_answer_canonical_finite_question(raw, policy.label)
    assert canonical is not None
    assert "TMU Faculty of Arts departments" in canonical.answer
    assert "Criminology" in canonical.answer
    assert canonical.sources[0].url.endswith("/arts/about/departments/")


def test_minor_declaration_policy_prefers_curriculum_pages():
    raw = "How do I declare a minor?"
    policy = choose_retrieval_policy(raw, raw)

    assert policy.label == "MINOR_DECLARATION"
    assert any("curriculum-advising" in frag for frag in policy.preferred_urls)
    assert "MyServiceHub" in policy.retrieval_query


def test_course_enrolment_policy_prefers_current_students_and_myservicehub():
    raw = "How do I enroll in classes?"
    policy = choose_retrieval_policy(raw, raw)

    assert policy.label == "COURSE_ENROLMENT"
    assert any("current-students/course-enrolment" in frag for frag in policy.preferred_urls)
    assert any("myservicehub-support/students/academics" in frag for frag in policy.preferred_urls)


def test_academic_consideration_policy_is_selected():
    raw = "What is an academic consideration request?"
    policy = choose_retrieval_policy(raw, raw)

    assert policy.label == "ACADEMIC_CONSIDERATION"
    assert any("academic-consideration" in frag for frag in policy.preferred_urls)


def test_one_shot_course_planning_prefers_program_calendar_policy():
    raw = "What courses should I pick for Criminology first year?"
    effective = "What courses should a first year student in the Criminology program take in TMU Faculty of Arts? Use the undergraduate calendar curriculum tables, first-year requirements, Program Overview/Curriculum Information, Full-Time, Four-Year Program, Full-Time, Five-Year Co-op Program, and Table I/II pages when relevant."
    policy = choose_retrieval_policy(raw, effective)

    assert policy.label == "COURSE_PLANNING_CALENDAR"
    assert policy.program_slug == "criminology"
    assert any("/calendar/2025-2026/programs/arts/criminology" in frag for frag in policy.preferred_urls)
    assert any("table i" in term.lower() for term in policy.preferred_section_terms)


def test_program_requirements_prefers_specific_calendar_program_policy():
    raw = "Criminology BA"
    effective = "What are the required courses, first-year requirements, degree requirements, curriculum groups, liberal studies, and electives for the Criminology program in TMU Faculty of Arts? Use the undergraduate calendar curriculum tables, Program Overview/Curriculum Information, Full-Time, Four-Year Program, Full-Time, Five-Year Co-op Program, and Table I/II pages when relevant."
    policy = choose_retrieval_policy(raw, effective)

    assert policy.label == "PROGRAM_REQUIREMENTS_CALENDAR"
    assert policy.program_slug == "criminology"
    assert policy.same_source_limit == 4
