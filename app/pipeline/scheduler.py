"""Simple in-container scheduler for the crawl + ingestion pipeline.

This module runs the pipeline in a loop with a fixed interval, which is useful for a Docker-only deployment.
In production you can also run the one-shot pipeline via cron (recommended for finer control).
"""

import os
import time

from app.pipeline.run_pipeline import _split_profiles, run_pipeline_once


def main() -> None:
    """Run the pipeline forever with a fixed sleep interval."""
    profiles = _split_profiles(os.getenv("PIPELINE_PROFILES", "arts"))
    interval = int(os.getenv("PIPELINE_INTERVAL_SECONDS", "21600"))  # 6 hours
    crawl_rps = float(os.getenv("CRAWL_RPS", "1.0"))
    enable_sitemaps = os.getenv("CRAWL_ENABLE_SITEMAPS", "true").lower() in {"1", "true", "yes"}
    ingest_limit = int(os.getenv("INGEST_LIMIT", "200"))

    if not profiles:
        raise SystemExit("PIPELINE_PROFILES is empty")

    while True:
        try:
            run_pipeline_once(profiles, crawl_rps=crawl_rps, enable_sitemaps=enable_sitemaps, ingest_limit=ingest_limit)
        except Exception as e:
            # Keep the scheduler alive even if a single run fails.
            print(f"Pipeline run failed: {e}")

        time.sleep(max(1, interval))


if __name__ == "__main__":
    main()
