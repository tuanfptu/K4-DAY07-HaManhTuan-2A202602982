"""Dense vector retriever using embedding models and cosine similarity."""

from __future__ import annotations

import math
from typing import Any, Callable

# Load MockEmbedder from src.embeddings or fallback
try:
    from src.embeddings import MockEmbedder
except ImportError:
    try:
        from embeddings import MockEmbedder  # type: ignore
    except ImportError:
        import hashlib

        class MockEmbedder:  # type: ignore
            """Deterministic fallback mock embedder."""

            def __init__(self, dim: int = 64) -> None:
                self.dim = dim

            def __call__(self, text: str) -> list[float]:
                digest = hashlib.md5(text.encode()).hexdigest()
                seed = int(digest, 16)
                vec = []
                for _ in range(self.dim):
                    seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
                    vec.append((seed / 0xFFFFFFFF) * 2 - 1)
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                return [v / norm for v in vec]


def _normalize(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length (L2 norm)."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-12:
        return vector
    return [v / norm for v in vector]


def _dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


class DenseRetriever:
    """Dense retriever computing cosine similarity via dot products on normalized embeddings."""

    def __init__(self, embedding_fn: Callable[[str], list[float]] | None = None) -> None:
        """Initialize DenseRetriever with an embedding function.

        Args:
            embedding_fn: Callable that converts a string into a list of floats.
                          If None, defaults to MockEmbedder from src.embeddings.
        """
        if embedding_fn is not None:
            self.embedding_fn = embedding_fn
        else:
            self.embedding_fn = MockEmbedder()

        self.chunks: list[dict] = []
        self.embeddings: list[list[float]] = []

    def _extract_content(self, chunk: Any) -> str:
        """Extract text from chunk dictionary or object."""
        if isinstance(chunk, dict):
            return str(chunk.get("retrieval_content") or chunk.get("content") or "")
        if hasattr(chunk, "retrieval_content") and getattr(chunk, "retrieval_content"):
            return str(getattr(chunk, "retrieval_content"))
        if hasattr(chunk, "content"):
            return str(getattr(chunk, "content") or "")
        return str(chunk)

    def index(self, chunks: list[dict]) -> None:
        """Embed all chunks using retrieval_content and store unit-normalized embeddings.

        Args:
            chunks: List of chunk dictionaries containing 'retrieval_content' or 'content'.
        """
        self.chunks = list(chunks)
        self.embeddings = []

        # Identify chunks needing embedding
        needed_indices = []
        needed_texts = []
        for idx, chunk in enumerate(self.chunks):
            if isinstance(chunk, dict) and "_embedding" in chunk:
                continue
            if isinstance(chunk, dict) and "embedding" in chunk and chunk["embedding"]:
                chunk["_embedding"] = _normalize(list(chunk["embedding"]))
                continue
            needed_indices.append(idx)
            needed_texts.append(self._extract_content(chunk))

        if needed_texts:
            if hasattr(self.embedding_fn, "embed_batch"):
                batch_vectors = self.embedding_fn.embed_batch(needed_texts)
                for chunk_idx, raw_vec in zip(needed_indices, batch_vectors):
                    norm_vec = _normalize(list(raw_vec))
                    if isinstance(self.chunks[chunk_idx], dict):
                        self.chunks[chunk_idx]["_embedding"] = norm_vec
            else:
                for chunk_idx, text in zip(needed_indices, needed_texts):
                    raw_vec = self.embedding_fn(text)
                    norm_vec = _normalize(list(raw_vec))
                    if isinstance(self.chunks[chunk_idx], dict):
                        self.chunks[chunk_idx]["_embedding"] = norm_vec

        for chunk in self.chunks:
            if isinstance(chunk, dict) and "_embedding" in chunk:
                self.embeddings.append(chunk["_embedding"])
            else:
                text = self._extract_content(chunk)
                norm_vec = _normalize(list(self.embedding_fn(text)))
                self.embeddings.append(norm_vec)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Embed query, compute cosine similarity via dot product, and return top_k matches.

        Args:
            query: Search query text.
            top_k: Maximum number of ranked results to return (default: 20).

        Returns:
            List of (chunk_index, score) pairs sorted descending by cosine similarity score.
        """
        if not query or not self.chunks or not self.embeddings or top_k <= 0:
            return []

        raw_query_vec = self.embedding_fn(query)
        query_vec = _normalize(list(raw_query_vec))

        scored: list[tuple[int, float]] = []
        for idx, emb in enumerate(self.embeddings):
            score = _dot_product(query_vec, emb)
            scored.append((idx, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
