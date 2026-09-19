"""Hybrid retriever combining BM25 sparse retrieval and dense embeddings via RRF."""

from __future__ import annotations

from typing import Any

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Combines BM25 sparse keyword search and dense semantic search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
    ) -> None:
        """Initialize HybridRetriever with BM25 and Dense retriever instances.

        Args:
            bm25: Initialized BM25Retriever instance.
            dense: Initialized DenseRetriever instance.
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (default: 60).
        """
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.chunks: list[dict] = []

    def index(self, chunks: list[dict]) -> None:
        """Index chunks in both BM25 and dense retrievers and store chunk references.

        Args:
            chunks: List of chunk dictionaries to index.
        """
        self.chunks = list(chunks)
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(
        self,
        query: str,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        final_top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Perform hybrid search by querying BM25 and dense retrievers and fusing with RRF.

        Args:
            query: Search query string.
            bm25_top_k: Number of candidates to retrieve from BM25 (default: 20).
            dense_top_k: Number of candidates to retrieve from Dense retriever (default: 20).
            final_top_k: Number of fused results to return (default: 10).

        Returns:
            List of (chunk_index, rrf_score) pairs sorted descending by RRF score.
        """
        if not query or not self.chunks or final_top_k <= 0:
            return []

        bm25_results = self.bm25.search(query, top_k=bm25_top_k)
        dense_results = self.dense.search(query, top_k=dense_top_k)

        fused = reciprocal_rank_fusion(
            [bm25_results, dense_results],
            k=self.rrf_k,
        )

        return fused[:final_top_k]

    def get_chunk(self, index: int) -> dict | None:
        """Retrieve chunk dictionary by index for convenient lookup.

        Args:
            index: Integer index of the chunk.

        Returns:
            The chunk dictionary if index is valid, None otherwise.
        """
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None
