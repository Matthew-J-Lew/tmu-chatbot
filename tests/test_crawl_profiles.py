import os

from app.crawler.crawl import load_profiles_yaml, parse_calendar_years, resolve_profile_config, url_in_scope


def test_parse_calendar_years_prefers_env_values():
    years = parse_calendar_years("2025-2026, 2026-2027,2025-2026", ["2024-2025"])
    assert years == ["2025-2026", "2026-2027"]


def test_parse_calendar_years_falls_back_to_defaults():
    years = parse_calendar_years("", ["2025-2026", "2026-2027"])
    assert years == ["2025-2026", "2026-2027"]


def test_resolve_arts_calendar_profile_builds_year_scoped_prefixes(monkeypatch):
    monkeypatch.setenv("CRAWL_CALENDAR_ALLOWED_YEARS", "2025-2026,2026-2027")
    cfg = resolve_profile_config(
        "arts_calendar",
        {
            "generated": {"kind": "arts_calendar", "years_env": "CRAWL_CALENDAR_ALLOWED_YEARS"},
            "allowed_domains": ["www.torontomu.ca"],
            "seeds": [],
            "allowed_path_prefixes": [],
            "allowed_path_regex": [],
            "deny_path_regex": [],
            "strip_query": True,
            "include_pdfs": False,
            "max_pages": 2500,
            "max_depth": 4,
            "use_sitemaps": False,
        },
    )
    assert "https://www.torontomu.ca/calendar/2025-2026/programs/arts/" in cfg["seeds"]
    assert "https://www.torontomu.ca/calendar/2025-2026/sitemap/" in cfg["seeds"]
    assert "/calendar/2026-2027/programs/arts" in cfg["allowed_path_prefixes"]
    assert "/content/ryerson/calendar/2026-2027/programs/arts" in cfg["allowed_path_prefixes"]
    assert any("/sitemap" in pat for pat in cfg["allowed_path_regex"])
    assert any("printpage" in pat for pat in cfg["deny_path_regex"])
    assert cfg["use_sitemaps"] is False


def test_url_in_scope_allows_calendar_sitemap_via_allow_regex():
    allowed, reason = url_in_scope(
        "https://www.torontomu.ca/calendar/2026-2027/sitemap/",
        ["www.torontomu.ca"],
        ["/calendar/2026-2027/programs/arts"],
        [r"^/calendar/2026-2027/sitemap(?:/|\.html)?$"],
        [r"^/calendar/\d{4}-\d{4}/courses(?:/|$)"],
    )
    assert allowed is True
    assert reason.startswith("allowed_by_regex:")


def test_load_profiles_yaml_resolves_generated_profile_from_env(monkeypatch):
    monkeypatch.setenv("CRAWL_CALENDAR_ALLOWED_YEARS", "2026-2027")
    data = load_profiles_yaml("app/crawler/profiles.yaml")
    profile = data["profiles"]["arts_calendar"]
    assert profile["seeds"] == [
        "https://www.torontomu.ca/calendar/2026-2027/programs/arts/",
        "https://www.torontomu.ca/calendar/2026-2027/sitemap/",
    ]
    assert "/calendar/2026-2027/programs/arts" in profile["allowed_path_prefixes"]
    assert any("/calendar/2026-2027/sitemap" in pat for pat in profile["allowed_path_regex"])
