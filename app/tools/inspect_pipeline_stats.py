"""Inspect crawler/ingestion observability tables.

This is a lightweight QA helper that prints recent crawl/ingest runs and per-status URL counts.
Run it inside the api container:
  python -m app.tools.inspect_pipeline_stats
"""

import asyncio

from app.api.db import init_db_pool, close_db_pool, get_pool


async def _run() -> None:
    await init_db_pool()
    pool = get_pool()

    async with pool.acquire() as conn:
        profiles = await conn.fetch(
            """
            SELECT id, name FROM crawl_profiles ORDER BY name ASC;
            """
        )
        print("\n=== CRAWL PROFILES ===")
        for p in profiles:
            print(f"- {p['name']} (id={p['id']})")

        print("\n=== CRAWL TARGET STATUS COUNTS ===")
        rows = await conn.fetch(
            """
            SELECT cp.name, ct.status, COUNT(*) AS n
            FROM crawl_targets ct
            JOIN crawl_profiles cp ON cp.id = ct.profile_id
            GROUP BY cp.name, ct.status
            ORDER BY cp.name ASC, n DESC;
            """
        )
        cur_name = None
        for r in rows:
            if r["name"] != cur_name:
                cur_name = r["name"]
                print(f"\nProfile: {cur_name}")
            print(f"  {r['status']}: {r['n']}")

        print("\n=== RECENT CRAWL RUNS (last 5) ===")
        runs = await conn.fetch(
            """
            SELECT cp.name, cr.started_at, cr.finished_at, cr.discovered_count, cr.approved_count, cr.blocked_count, cr.failed_count
            FROM crawl_runs cr
            JOIN crawl_profiles cp ON cp.id = cr.profile_id
            ORDER BY cr.started_at DESC
            LIMIT 5;
            """
        )
        for r in runs:
            print(
                f"- {r['name']} start={r['started_at']} discovered={r['discovered_count']} approved={r['approved_count']} blocked={r['blocked_count']} failed={r['failed_count']}"
            )

        print("\n=== RECENT FAILED TARGETS (last 20) ===")
        failed_targets = await conn.fetch(
            """
            SELECT cp.name, ct.url, ct.last_http_status, ct.error, ct.updated_at, ct.meta->'last_ingest' AS last_ingest
            FROM crawl_targets ct
            JOIN crawl_profiles cp ON cp.id = ct.profile_id
            WHERE ct.status = 'failed'
            ORDER BY ct.updated_at DESC
            LIMIT 20;
            """
        )
        if not failed_targets:
            print("(none)")
        for ft in failed_targets:
            li = ft["last_ingest"] or {}
            err_type = li.get("error_type") or ""
            fetcher = li.get("fetcher") or ""
            status = ft["last_http_status"] or li.get("http_status") or ""
            print(f"- {ft['name']} status={status} fetcher={fetcher} type={err_type} url={ft['url']}")
            if ft["error"]:
                print(f"    error: {ft['error']}")
            stage = li.get("stage")
            if stage:
                print(f"    stage: {stage}")
            dur = li.get("duration_ms")
            if dur is not None:
                print(f"    duration_ms: {dur}")

        print("\n=== RECENT INGEST RUNS (last 5) ===")
        iruns = await conn.fetch(
            """
            SELECT cp.name, ir.started_at, ir.finished_at, ir.selected_count, ir.ingested_count, ir.skipped_count, ir.failed_count
            FROM ingest_runs ir
            JOIN crawl_profiles cp ON cp.id = ir.profile_id
            ORDER BY ir.started_at DESC
            LIMIT 5;
            """
        )
        for r in iruns:
            print(
                f"- {r['name']} start={r['started_at']} selected={r['selected_count']} ingested={r['ingested_count']} skipped={r['skipped_count']} failed={r['failed_count']}"
            )

    await close_db_pool()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
