# app/rag/debug.py
"""
Debug / evaluation helpers for the retrieval pipeline.

Usage example (inside the API container or your dev env):

    from app.rag.debug import debug_retrieve
    debug_retrieve("How do I apply to the Faculty of Arts?", k=6)

This will:
  - run the full pipeline (embed -> hybrid search -> rerank)
  - print total latency
  - print previews of the top-k chunks and their scores
"""

from __future__ import annotations

import time
from typing import Optional

from .retrieval import retrieve


def debug_retrieve(
    query: str,
    k: int = 8,
    num_candidates: int = 30,
    show_chunk_chars: int = 400,
) -> None:
    """
    Run the full retrieval pipeline for a single query and print details.

    Args:
        query: The user's question.
        k: Final number of chunks to return after reranking.
        num_candidates: Number of candidates to pull from Postgres
                        before reranking.
        show_chunk_chars: Number of characters to show from each chunk
                          as a preview in the console.
    """
    t0 = time.time()
    results = retrieve(query, k=k, num_candidates=num_candidates)
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
