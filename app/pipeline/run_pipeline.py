"""Run the crawl + ingestion pipeline.

This module ties together URL discovery (crawler) and document ingestion in a single, repeatable run.
It is designed to be used in scheduled jobs (cron, CI, or the included scheduler container).
"""

import argparse
import json
import os
from typing import List

from app.crawler.crawl import ensure_profile, run_crawl, connect as pg_connect
from app.ingestion.ingest import run_db_mode


def _split_profiles(s: str) -> List[str]:
    """Parse a comma-separated profile list."""
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def run_pipeline_once(profiles: List[str], *, crawl_rps: float, enable_sitemaps: bool, ingest_limit: int) -> dict:
    """Run one full pipeline cycle for the given profiles."""
    out = {"profiles": {}}

    # Bootstrap profiles (if needed) using the default profiles.yaml.
    profiles_yaml = os.getenv("CRAWL_PROFILES_YAML", "/app/app/crawler/profiles.yaml")

    for name in profiles:
        with pg_connect() as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                profile = ensure_profile(cur, name, profiles_yaml)
                conn.commit()

        crawl_stats = run_crawl(profile, rps=crawl_rps, enable_sitemaps=enable_sitemaps, max_pages_override=None)

        # Ingest the highest-priority approved pages.
        with pg_connect() as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                run_db_mode(cur, profile_name=name, limit=ingest_limit, sleep_seconds=float(os.getenv("INGEST_SLEEP_SECONDS", "0.3")))
                conn.commit()

        out["profiles"][name] = {"crawl": crawl_stats, "ingest": {"limit": ingest_limit}}

    return out


def main() -> None:
    """CLI entrypoint for one-shot pipeline execution."""
    parser = argparse.ArgumentParser(description="Run crawl + ingestion in a single command.")
    parser.add_argument("--profiles", default=os.getenv("PIPELINE_PROFILES", "arts"), help="Comma-separated crawl profile names")
    parser.add_argument("--crawl-rps", type=float, default=float(os.getenv("CRAWL_RPS", "1.0")))
    parser.add_argument("--enable-sitemaps", action="store_true", default=os.getenv("CRAWL_ENABLE_SITEMAPS", "true").lower() in {"1","true","yes"})
    parser.add_argument("--ingest-limit", type=int, default=int(os.getenv("INGEST_LIMIT", "200")))
    args = parser.parse_args()

    profiles = _split_profiles(args.profiles)
    if not profiles:
        raise SystemExit("No profiles provided")

    result = run_pipeline_once(profiles, crawl_rps=args.crawl_rps, enable_sitemaps=args.enable_sitemaps, ingest_limit=args.ingest_limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
