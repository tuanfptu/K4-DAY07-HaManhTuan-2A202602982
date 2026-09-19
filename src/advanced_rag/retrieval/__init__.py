"""Retrieval module for Advanced RAG.

Exports sparse, dense, hybrid, multi-query, and HyDE retrievers along with RRF fusion
and metadata filtering utilities.
"""

from __future__ import annotations

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .hyde import HyDERetriever
from .hybrid_retriever import HybridRetriever
from .metadata_filter import filter_by_audience, filter_chunks
from .multi_query import MultiQueryRetriever
from .parent_child_retriever import ParentChildRetriever
from .rrf import reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "reciprocal_rank_fusion",
    "HybridRetriever",
    "filter_chunks",
    "filter_by_audience",
    "MultiQueryRetriever",
    "HyDERetriever",
    "ParentChildRetriever",
]
