# app/rag/reranker.py
"""
Cross-encoder reranker for the RAG system.

Pipeline role:
  1) Hybrid retrieval in Postgres returns a set of candidate chunks
     (vector + keyword search) using rag_hybrid_search().
  2) This module scores each (query, chunk) pair with a cross-encoder
     and reorders the candidates.
  3) The top-k reranked chunks are passed to the LLM as context.

So we basically take the top candidate chunks and do an additional screening to return the top k BEST chunks from the candidate chunks

"""

from __future__ import annotations

from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

# A lightweight, fast reranker that works well with MS MARCO style data.
# You can experiment with other models later if needed.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load the model once at import time for this worker process.
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker

# Reranks candidate chunks using a cross encoder
# Args:
#        query: The user's original question or search query.
#        candidates: List of candidate chunks, each a dict at least with:
#            {
#              "chunk": str,
#              ... (other fields from rag_hybrid_search)
#            }
#        top_k: Number of top results to keep after reranking.
#
#    Returns:
#        A list of candidate dicts sorted by rerank_score (descending),
#        truncated to top_k. Each dict will be augmented with:
#            "rerank_score": float
def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
   
    if not candidates:
        return []

    reranker = _get_reranker()

    # Prepare (query, chunk-text) pairs for the model
    pairs = [(query, c["chunk"]) for c in candidates]

    # Single batched forward pass for efficiency
    scores = reranker.predict(pairs)

    # Attach scores to candidate dicts
    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    # Sort by rerank_score descending
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Return only the top_k
    return candidates[:top_k]
