"""Retrieve small child chunks and expand them to heading-aligned parents."""

from __future__ import annotations

from typing import Any

from .hybrid_retriever import HybridRetriever


class ParentChildRetriever:
    """Adapter that performs child-level search and parent-level context expansion.

    Ranking stays precise because BM25/dense search indexes ``retrieval_content`` of
    children.  Returned chunks contain the complete parent section, grouped by
    ``parent_id`` so the generator receives coherent evidence without duplicates.
    """

    def __init__(self, child_retriever: HybridRetriever) -> None:
        self.child_retriever = child_retriever
        self.chunks: list[dict[str, Any]] = []

    def index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = list(chunks)
        self.child_retriever.index(self.chunks)

    def search(
        self,
        query: str,
        *,
        child_top_k: int = 20,
        parent_top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return unique expanded parents in first-hit ranking order."""
        if not query or parent_top_k <= 0:
            return []

        ranked = self.child_retriever.search(
            query,
            bm25_top_k=max(child_top_k, parent_top_k),
            dense_top_k=max(child_top_k, parent_top_k),
            final_top_k=max(child_top_k, parent_top_k),
        )
        parents: list[dict[str, Any]] = []
        seen: set[str] = set()
        for child_index, score in ranked:
            if not 0 <= child_index < len(self.chunks):
                continue
            child = self.chunks[child_index]
            parent_id = str(child.get("parent_id") or child.get("id") or child_index)
            if parent_id in seen:
                continue
            seen.add(parent_id)
            metadata = dict(child.get("chunk_metadata", child.get("metadata", {})))
            parents.append(
                {
                    "id": parent_id,
                    "parent_id": parent_id,
                    "content": child.get("parent_content") or child.get("content", ""),
                    "retrieval_content": child.get("retrieval_content", child.get("content", "")),
                    "matched_child_id": child.get("id", ""),
                    "matched_child_content": child.get("content", ""),
                    "score": float(score),
                    "chunk_metadata": metadata,
                    "metadata": metadata,
                }
            )
            if len(parents) >= parent_top_k:
                break
        return parents
