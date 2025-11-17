# app/rag/embeddings.py
"""
Embedding utilities for the RAG system.

This module is responsible for turning text into dense vectors that match
the `embedding VECTOR(384)` column in the `chunks` table (see schema.sql).

Important:
- The SAME model used here for query embeddings should also be used when
  creating chunk embeddings during ingestion, otherwise retrieval quality
  will be poor.
- This file is used at RUNTIME (when the user asks a question) as part
  of the retrieval pipeline.

Pipeline role:
  User question -> embed_query() -> rag_hybrid_search() in Postgres
  -> cross-encoder reranker -> LLM answer

What is the file doing?
1.  This file loads the all-MiniLM-L6-v2 language/AI model
2.  The file then turns sentences/questions into 384-dimensional vectors (lists of 384 numbers). Those numbers represent the meaning of the sentence.
3.  Performs a safety check to ensure the model is making 384 dimensional vectors, ** if you change the data type you HAVE tpo re-ingest with the new datatype **

"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

# 384-dim model, matches VECTOR(384) in the DB.
# This can be changed later but we MUST:
# - re-embed all chunks with the new model
# - update the VECTOR dimension in schema.sql if needed
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# Does a lazy load on the sentence transformer model so we can cache it and re-use it instead of reloading for each request
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

# Calls model.encode on a batch of strings
# Returns a NumPy array shaped (batch_size, 384)
# Encodes text into embeddings
# Args:
#        texts: A sequence of strings to embed.
#        normalize: If True, L2-normalize embeddings so cosine similarity
#                   is equivalent to dot product, which works well with
#                   pgvector's cosine ops.
def _encode_texts(texts: Sequence[str], *, normalize: bool = True) -> np.ndarray:

    if not texts:
        # Return an empty (0, EMBEDDING_DIM) array for convenience
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    model = _get_model()

    embeddings = model.encode(
        list(texts),
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    # Safety check: ensure the dimension matches the DB schema
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
            f"got {embeddings.shape[1]} from model '{EMBEDDING_MODEL_NAME}'. "
            "Update EMBEDDING_DIM and the DB schema if you intentionally "
            "switched models."
        )

    return embeddings.astype("float32")


# Embed a single user query string into a 384-dim vector.
# This is what the retrieval pipeline calls at runtime before passing the embedding into Postgres' rag_hybrid_search().
# Args: text = the user's question or search query
# Returns A list[float] of length EMBEDDING_DIM suitable for passing as a `VECTOR(384)` parameter in SQL (e.g. `%s::vector`).
def embed_query(text: str) -> List[float]:
   
    # Handle empty queries gracefully by returning a zero vector
    if not text:
        return [0.0] * EMBEDDING_DIM

    embeddings = _encode_texts([text])
    return embeddings[0].tolist()

# Embeds a batch of texts
# Args: texts = sequence of strings to embed
# Returns a list of floats, one embedding per input text
def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    
    embeddings = _encode_texts(texts)
    return embeddings.tolist()
