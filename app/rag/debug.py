# app/rag/debug.py
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from app.api.db import init_db_pool, close_db_pool, get_pool
from .retrieval import retrieve


def debug_retrieve(
    query: str,
    k: int = 8,
    num_candidates: int = 30,
    show_chunk_chars: int = 400,
) -> None:
    asyncio.run(_debug_retrieve_async(query, k, num_candidates, show_chunk_chars))


async def _debug_retrieve_async(
    query: str,
    k: int,
    num_candidates: int,
    show_chunk_chars: int,
) -> None:
    await init_db_pool()
    pool = get_pool()

    t0 = time.time()
    results = await retrieve(pool=pool, query=query, k=k, num_candidates=num_candidates)
    t1 = time.time()

    print("=" * 80)
    print(f"Query: {query!r}")
    print(f"Total latency: {t1 - t0:.3f} s")
    print(f"Returned {len(results)} chunks (k={k}, num_candidates={num_candidates})")
    print("=" * 80)

    for idx, r in enumerate(results, start=1):
        print(f"\n[{idx}] chunk_id={r['id']}  source_url={r['source_url']}")
        print(f"chunk_url:  {r['chunk_url']}")
        section = r.get("section")
        if section:
            print(f"section:    {section}")
        print(
            f"rerank_score={r.get('rerank_score'):.4f}  "
            f"hybrid_score={r.get('hybrid_score'):.4f}"
        )
        preview = (r["chunk"] or "").replace("\n", " ")
        if len(preview) > show_chunk_chars:
            preview = preview[: show_chunk_chars - 3] + "..."
        print("Preview:")
        print(preview)

    print("\n" + "=" * 80 + "\n")
    await close_db_pool()
