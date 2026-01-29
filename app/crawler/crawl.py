"""TMU Chatbot crawler / URL discovery.

This module discovers in-scope URLs starting from profile seeds (and optional sitemaps),
then writes the approved crawl targets to Postgres for downstream ingestion.
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import psycopg2
import requests
import yaml
from bs4 import BeautifulSoup
from psycopg2.extras import Json

PGHOST = os.getenv("PGHOST", "pg")
PGUSER = os.getenv("PGUSER", "rag")
PGPASSWORD = os.getenv("PGPASSWORD", "rag")
PGDATABASE = os.getenv("PGDATABASE", "ragdb")

CRAWL_USER_AGENT = os.getenv("CRAWL_USER_AGENT", "TMU-FOA-Crawler/0.1 (+https://www.torontomu.ca/)")
DEFAULT_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT_SECONDS", "45"))


@dataclass
class CrawlProfile:
    """Configuration for a bounded crawl scope."""

    id: int
    name: str
    seeds: List[str]
    allowed_domains: List[str]
    allowed_path_prefixes: List[str]
    deny_path_regex: List[str]
    strip_query: bool
    include_pdfs: bool
    max_pages: int
    max_depth: int


class HostRateLimiter:
    """Simple per-host rate limiter to keep crawling polite."""

    def __init__(self, rps: float) -> None:
        self.rps = max(rps, 0.0)
        self._last: Dict[str, float] = {}

    def sleep_if_needed(self, host: str) -> None:
        if self.rps <= 0:
            return
        min_interval = 1.0 / self.rps
        now = time.time()
        last = self._last.get(host)
        if last is not None:
            wait = min_interval - (now - last)
            if wait > 0:
                time.sleep(wait)
        self._last[host] = time.time()


def connect():
    """Open a Postgres connection (psycopg2)."""
    return psycopg2.connect(host=PGHOST, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE)


def normalize_url(url: str, base_url: Optional[str] = None, strip_query: bool = True) -> Optional[str]:
    """Normalize a URL for deduping and stable identity (fragments removed, optional query stripping)."""
    if not url:
        return None

    # Resolve relative links against the base URL.
    if base_url:
        url = urllib.parse.urljoin(base_url, url)

    url, _frag = urllib.parse.urldefrag(url)
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Remove default ports.
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = parsed.path or "/"

    # Normalize repeated slashes and strip trailing slash (except root).
    path = re.sub(r"/+$", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query = ""
    if not strip_query and parsed.query:
        # Drop common tracking parameters.
        q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        q = [(k, v) for (k, v) in q if not k.lower().startswith("utm_") and k.lower() not in {"fbclid"}]
        query = urllib.parse.urlencode(sorted(q))

    rebuilt = urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))
    return rebuilt


def url_in_scope(url: str, allowed_domains: List[str], allowed_prefixes: List[str], deny_regex: List[str]) -> Tuple[bool, str]:
    """Return (allowed, reason) applying the profile's policy rules."""
    p = urllib.parse.urlparse(url)
    host = p.netloc.lower()
    path = p.path or "/"

    if allowed_domains and not any(host == d or host.endswith("." + d) for d in allowed_domains):
        return False, "domain_not_allowed"

    if allowed_prefixes and not any(path.startswith(pref) for pref in allowed_prefixes):
        return False, "path_prefix_not_allowed"

    for pat in deny_regex:
        try:
            if re.search(pat, path):
                return False, f"denied_by_regex:{pat}"
        except re.error:
            continue

    return True, "allowed"


def compute_priority(url: str, depth: int, from_sitemap: bool) -> float:
    """Heuristic priority score used to decide what to ingest first."""
    p = urllib.parse.urlparse(url)
    path = (p.path or "").lower()

    score = 0.35

    # Prefer more "evergreen" informational pages.
    positive = [
        "program", "programs", "department", "departments", "undergraduate", "graduate", "courses",
        "curriculum", "admissions", "future-students", "contact", "about", "faculty", "research",
    ]
    negative = [
        "news", "event", "events", "calendar", "story", "stories", "blog", "social", "gallery",
    ]

    for kw in positive:
        if kw in path:
            score += 0.08

    for kw in negative:
        if kw in path:
            score -= 0.12

    # Slightly prefer shallow URLs.
    score -= min(depth, 10) * 0.02

    # Give sitemap discoveries a small boost.
    if from_sitemap:
        score += 0.05

    # Penalize URLs with query strings (usually less stable).
    if p.query:
        score -= 0.1

    return float(max(0.0, min(1.0, score)))


def fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
    """Fetch a URL with a consistent crawler User-Agent."""
    headers = {"User-Agent": CRAWL_USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    return resp


def parse_links(html: str, base_url: str) -> Tuple[Optional[str], List[str]]:
    """Extract canonical URL (if present) and all outgoing links from an HTML page."""
    soup = BeautifulSoup(html, "lxml")

    canonical = None
    can = soup.find("link", attrs={"rel": re.compile("canonical", re.I)})
    if can and can.get("href"):
        canonical = can.get("href")

    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if href:
            links.append(urllib.parse.urljoin(base_url, href))

    return canonical, links


def parse_sitemap(content: bytes, base_url: str) -> List[str]:
    """Parse sitemap XML (or HTML sitemap pages) into a list of candidate URLs."""
    text = content.decode("utf-8", errors="ignore")

    # Attempt XML parsing first.
    try:
        root = ET.fromstring(text)
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        urls: List[str] = []
        for loc in root.findall(f".//{ns}loc"):
            if loc.text:
                urls.append(loc.text.strip())
        if urls:
            return urls
    except ET.ParseError:
        pass

    # Fallback: treat as HTML and extract anchors.
    soup = BeautifulSoup(text, "lxml")
    out: List[str] = []
    for a in soup.find_all("a", href=True):
        out.append(urllib.parse.urljoin(base_url, a.get("href")))
    return out


def load_profiles_yaml(path: str) -> dict:
    """Load crawl profiles from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_profile(cur, name: str, yaml_path: str) -> CrawlProfile:
    """Ensure a profile exists in DB; bootstrap it from YAML if missing."""
    cur.execute(
        """
        SELECT id, name, seeds, allowed_domains, allowed_path_prefixes, deny_path_regex,
               strip_query, include_pdfs, max_pages, max_depth
        FROM crawl_profiles
        WHERE name=%s;
        """,
        (name,),
    )
    row = cur.fetchone()
    if row:
        return CrawlProfile(
            id=int(row[0]),
            name=row[1],
            seeds=list(row[2] or []),
            allowed_domains=list(row[3] or []),
            allowed_path_prefixes=list(row[4] or []),
            deny_path_regex=list(row[5] or []),
            strip_query=bool(row[6]),
            include_pdfs=bool(row[7]),
            max_pages=int(row[8]),
            max_depth=int(row[9]),
        )

    cfg = load_profiles_yaml(yaml_path)
    prof = (cfg.get("profiles") or {}).get(name)
    if not prof:
        raise RuntimeError(f"Profile {name!r} not found in DB or YAML at {yaml_path}")

    cur.execute(
        """
        INSERT INTO crawl_profiles (name, seeds, allowed_domains, allowed_path_prefixes, deny_path_regex,
                                   strip_query, include_pdfs, max_pages, max_depth)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (
            name,
            Json(prof.get("seeds", [])),
            Json(prof.get("allowed_domains", [])),
            Json(prof.get("allowed_path_prefixes", [])),
            Json(prof.get("deny_path_regex", [])),
            bool(prof.get("strip_query", True)),
            bool(prof.get("include_pdfs", True)),
            int(prof.get("max_pages", 5000)),
            int(prof.get("max_depth", 6)),
        ),
    )
    pid = int(cur.fetchone()[0])
    return CrawlProfile(
        id=pid,
        name=name,
        seeds=list(prof.get("seeds", [])),
        allowed_domains=list(prof.get("allowed_domains", [])),
        allowed_path_prefixes=list(prof.get("allowed_path_prefixes", [])),
        deny_path_regex=list(prof.get("deny_path_regex", [])),
        strip_query=bool(prof.get("strip_query", True)),
        include_pdfs=bool(prof.get("include_pdfs", True)),
        max_pages=int(prof.get("max_pages", 5000)),
        max_depth=int(prof.get("max_depth", 6)),
    )


def create_crawl_run(cur, profile_id: int, meta: dict) -> int:
    """Create a crawl_runs row to record observability stats for this run."""
    cur.execute("INSERT INTO crawl_runs (profile_id, meta) VALUES (%s, %s) RETURNING id;", (profile_id, Json(meta)))
    return int(cur.fetchone()[0])


def finish_crawl_run(cur, run_id: int, discovered: int, approved: int, blocked: int, failed: int, meta: dict) -> None:
    """Finalize a crawl_runs row with counts and a finish timestamp."""
    cur.execute(
        """
        UPDATE crawl_runs
        SET finished_at=NOW(), discovered_count=%s, approved_count=%s, blocked_count=%s, failed_count=%s, meta=%s
        WHERE id=%s;
        """,
        (discovered, approved, blocked, failed, Json(meta), run_id),
    )


def upsert_target(
    cur,
    profile_id: int,
    url: str,
    normalized_url: str,
    status: str,
    priority: float,
    reason: str,
    meta: dict,
) -> None:
    """Upsert a crawl_targets row keyed by (profile_id, normalized_url)."""
    cur.execute(
        """
        INSERT INTO crawl_targets (profile_id, url, normalized_url, status, priority, last_seen_at, meta)
        VALUES (%s, %s, %s, %s, %s, NOW(), %s)
        ON CONFLICT (profile_id, normalized_url) DO UPDATE SET
          url = EXCLUDED.url,
          status = EXCLUDED.status,
          priority = GREATEST(crawl_targets.priority, EXCLUDED.priority),
          last_seen_at = NOW(),
          updated_at = NOW(),
          meta = COALESCE(crawl_targets.meta, '{}'::jsonb) || EXCLUDED.meta;
        """,
        (
            profile_id,
            url,
            normalized_url,
            status,
            float(priority),
            Json({"reason": reason, **(meta or {})}),
        ),
    )


def discover_sitemaps(allowed_domains: List[str]) -> List[str]:
    """Try to find sitemap URLs via robots.txt for each allowed domain."""
    out: List[str] = []
    for dom in allowed_domains:
        robots = f"https://{dom}/robots.txt"
        try:
            resp = fetch(robots)
            if resp.status_code != 200:
                continue
            for line in resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm = line.split(":", 1)[1].strip()
                    if sm:
                        out.append(sm)
        except Exception:
            continue
    return list(dict.fromkeys(out))


def run_crawl(profile: CrawlProfile, rps: float, enable_sitemaps: bool, max_pages_override: Optional[int] = None) -> Dict[str, int]:
    """Run discovery and write approved targets to DB; returns simple stats."""
    discovered = approved = blocked = failed = 0
    max_pages = max_pages_override or profile.max_pages

    limiter = HostRateLimiter(rps=rps)

    # robots parser per domain for basic politeness.
    robots: Dict[str, urllib.robotparser.RobotFileParser] = {}
    for dom in profile.allowed_domains:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"https://{dom}/robots.txt")
        try:
            rp.read()
        except Exception:
            pass
        robots[dom] = rp

    conn = connect()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            run_id = create_crawl_run(cur, profile.id, meta={"profile": profile.name, "enable_sitemaps": enable_sitemaps})

            queue: Deque[Tuple[str, int, bool]] = deque()  # (url, depth, from_sitemap)
            seen: Set[str] = set()

            # Seed with sitemaps if enabled.
            if enable_sitemaps:
                for sm_url in discover_sitemaps(profile.allowed_domains):
                    try:
                        norm_sm = normalize_url(sm_url, strip_query=profile.strip_query)
                        if not norm_sm:
                            continue
                        host = urllib.parse.urlparse(norm_sm).netloc
                        limiter.sleep_if_needed(host)
                        resp = fetch(norm_sm)
                        if resp.status_code != 200:
                            continue
                        for u in parse_sitemap(resp.content, norm_sm):
                            queue.append((u, 0, True))
                    except Exception:
                        continue

            # Always add seeds.
            for s in profile.seeds:
                queue.append((s, 0, False))

            while queue and len(seen) < max_pages:
                raw_url, depth, from_sitemap = queue.popleft()

                norm = normalize_url(raw_url, base_url=None, strip_query=profile.strip_query)
                if not norm or norm in seen:
                    continue

                allowed, reason = url_in_scope(norm, profile.allowed_domains, profile.allowed_path_prefixes, profile.deny_path_regex)
                status = "approved" if allowed else "blocked"
                prio = compute_priority(norm, depth, from_sitemap)

                # Record the target even if blocked (useful for QA).
                upsert_target(cur, profile.id, url=norm, normalized_url=norm, status=status, priority=prio, reason=reason, meta={"depth": depth, "from_sitemap": from_sitemap})
                discovered += 1
                if allowed:
                    approved += 1
                else:
                    blocked += 1

                seen.add(norm)

                # Do not expand blocked targets.
                if not allowed:
                    continue

                # Stop expanding if we hit depth limit.
                if depth >= profile.max_depth:
                    continue

                # Skip non-html unless profile explicitly includes PDFs.
                if norm.lower().endswith(".pdf") and not profile.include_pdfs:
                    continue

                # Respect robots.txt when possible.
                host = urllib.parse.urlparse(norm).netloc
                rp = robots.get(host)
                if rp and not rp.can_fetch(CRAWL_USER_AGENT, norm):
                    continue

                try:
                    limiter.sleep_if_needed(host)
                    resp = fetch(norm)
                    if resp.status_code >= 400:
                        continue

                    ctype = (resp.headers.get("Content-Type") or "").lower()
                    if "text/html" not in ctype and "application/xhtml" not in ctype:
                        continue

                    canonical, links = parse_links(resp.text, resp.url)
                    if canonical:
                        can_norm = normalize_url(canonical, base_url=resp.url, strip_query=profile.strip_query)
                        if can_norm and can_norm != norm:
                            # Store canonical URL mapping (as another target) and prefer it for future expansion.
                            can_allowed, can_reason = url_in_scope(
                                can_norm, profile.allowed_domains, profile.allowed_path_prefixes, profile.deny_path_regex
                            )
                            can_status = "approved" if can_allowed else "blocked"
                            upsert_target(
                                cur,
                                profile.id,
                                url=can_norm,
                                normalized_url=can_norm,
                                status=can_status,
                                priority=max(prio, compute_priority(can_norm, depth, from_sitemap)),
                                reason=f"canonical:{can_reason}",
                                meta={"canonical_of": norm},
                            )
                            if can_allowed and can_norm not in seen:
                                queue.appendleft((can_norm, depth, from_sitemap))

                    for link in links:
                        norm_link = normalize_url(link, base_url=resp.url, strip_query=profile.strip_query)
                        if not norm_link or norm_link in seen:
                            continue
                        queue.append((norm_link, depth + 1, False))

                except Exception:
                    failed += 1
                    continue

            finish_crawl_run(
                cur,
                run_id,
                discovered=discovered,
                approved=approved,
                blocked=blocked,
                failed=failed,
                meta={"max_pages": max_pages, "max_depth": profile.max_depth, "seen": len(seen)},
            )
            conn.commit()

    finally:
        conn.close()

    return {"discovered": discovered, "approved": approved, "blocked": blocked, "failed": failed}


def main() -> None:
    """CLI entrypoint for running the crawler."""
    parser = argparse.ArgumentParser(description="Discover TMU URLs and write approved targets to Postgres.")
    parser.add_argument("--profile", default=os.getenv("CRAWL_PROFILE", "arts"))
    parser.add_argument("--profiles-yaml", default=os.getenv("CRAWL_PROFILES_YAML", "/app/app/crawler/profiles.yaml"))
    parser.add_argument("--rps", type=float, default=float(os.getenv("CRAWL_RPS", "1.0")))
    parser.add_argument("--enable-sitemaps", action="store_true", default=os.getenv("CRAWL_ENABLE_SITEMAPS", "true").lower() in {"1","true","yes"})
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("CRAWL_MAX_PAGES", "0")))
    args = parser.parse_args()

    conn = connect()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            profile = ensure_profile(cur, args.profile, args.profiles_yaml)
            conn.commit()
    finally:
        conn.close()

    stats = run_crawl(profile, rps=args.rps, enable_sitemaps=args.enable_sitemaps, max_pages_override=(args.max_pages or None))
    print(json.dumps({"profile": profile.name, **stats}, indent=2))


if __name__ == "__main__":
    main()
