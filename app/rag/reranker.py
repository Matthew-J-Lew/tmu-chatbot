"""
Cross-encoder reranker for the RAG system.

Why this exists (plain English):
- The database search gives us a shortlist of likely relevant passages.
- The reranker does a deeper comparison (question vs passage) to sort the shortlist.
- This usually improves answer quality, but it costs CPU/GPU time.
"""

from __future__ import annotations

from typing import Any, Dict, List
import os

from sentence_transformers import CrossEncoder

_RERANKER = None


def _get_model() -> CrossEncoder:
    """
    Load the reranker model once per process.
    Keeping it loaded makes repeated requests much faster.
    """
    global _RERANKER
    if _RERANKER is None:
        model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        # CrossEncoder will use GPU automatically if available (via PyTorch config)
        _RERANKER = CrossEncoder(model_name)
    return _RERANKER


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Score each candidate with a cross-encoder and return the best top_k.
    """
    if not candidates:
        return []

    reranker = _get_model()
    pairs = [(query, c.get("chunk", "")) for c in candidates]
    scores = reranker.predict(pairs)

    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return candidates[:top_k]
