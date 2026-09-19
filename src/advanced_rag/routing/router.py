"""RAG pipeline router.

Selects the optimal retrieval strategy based on query analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .query_analyzer import QueryAnalyzer, QueryAnalysisResult, QueryType


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    DENSE_ONLY = "dense_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    HYBRID_RERANKED = "hybrid_reranked"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    DECOMPOSED = "decomposed"


@dataclass
class RoutingDecision:
    """Decision made by the router."""
    strategy: RetrievalStrategy
    query_analysis: QueryAnalysisResult
    use_metadata_filter: bool = True
    use_reranker: bool = False
    use_query_rewrite: bool = False
    use_multi_query: bool = False
    use_hyde: bool = False
    use_decomposition: bool = False
    explanation: str = ""


class Router:
    """Route queries to optimal retrieval strategies.

    Uses rule-based routing for determinism and explainability.
    """

    def __init__(
        self,
        analyzer: QueryAnalyzer | None = None,
        reranker_available: bool = False,
        llm_available: bool = False,
    ) -> None:
        self.analyzer = analyzer or QueryAnalyzer()
        self.reranker_available = reranker_available
        self.llm_available = llm_available

    def route(self, query: str) -> RoutingDecision:
        """Analyze a query and determine the best retrieval strategy."""
        analysis = self.analyzer.analyze(query)

        match analysis.query_type:
            case QueryType.FACTUAL:
                return self._route_factual(analysis)
            case QueryType.PROCEDURAL:
                return self._route_procedural(analysis)
            case QueryType.MULTI_HOP:
                return self._route_multi_hop(analysis)
            case QueryType.GLOBAL:
                return self._route_global(analysis)
            case QueryType.AMBIGUOUS:
                return self._route_ambiguous(analysis)
            case QueryType.OUT_OF_SCOPE:
                return self._route_out_of_scope(analysis)
            case _:
                return self._route_factual(analysis)

    def _route_factual(self, analysis: QueryAnalysisResult) -> RoutingDecision:
        """Factual → metadata filter + hybrid + optional rerank."""
        return RoutingDecision(
            strategy=(
                RetrievalStrategy.HYBRID_RERANKED
                if self.reranker_available
                else RetrievalStrategy.HYBRID
            ),
            query_analysis=analysis,
            use_metadata_filter=True,
            use_reranker=self.reranker_available,
            explanation=(
                "Factual query: using hybrid retrieval with metadata filter"
                + (" and reranking" if self.reranker_available else "")
            ),
        )

    def _route_procedural(self, analysis: QueryAnalysisResult) -> RoutingDecision:
        """Procedural → rewrite + hybrid + rerank."""
        return RoutingDecision(
            strategy=(
                RetrievalStrategy.HYBRID_RERANKED
                if self.reranker_available
                else RetrievalStrategy.HYBRID
            ),
            query_analysis=analysis,
            use_metadata_filter=True,
            use_reranker=self.reranker_available,
            use_query_rewrite=True,
            explanation="Procedural query: rewriting for formal terms + hybrid retrieval",
        )

    def _route_multi_hop(self, analysis: QueryAnalysisResult) -> RoutingDecision:
        """Multi-hop → decompose + multiple retrievals + fuse + rerank."""
        return RoutingDecision(
            strategy=RetrievalStrategy.DECOMPOSED,
            query_analysis=analysis,
            use_metadata_filter=True,
            use_reranker=self.reranker_available,
            use_decomposition=True,
            explanation="Multi-hop query: decomposing into sub-queries for independent retrieval",
        )

    def _route_global(self, analysis: QueryAnalysisResult) -> RoutingDecision:
        """Global → multi-query or broader retrieval."""
        return RoutingDecision(
            strategy=RetrievalStrategy.MULTI_QUERY,
            query_analysis=analysis,
            use_metadata_filter=False,  # Don't restrict for global queries
            use_reranker=self.reranker_available,
            use_multi_query=True,
            explanation="Global query: using multi-query retrieval without metadata restrictions",
        )

    def _route_ambiguous(self, analysis: QueryAnalysisResult) -> RoutingDecision:
        """Ambiguous → multi-query or HyDE."""
        if self.llm_available:
            return RoutingDecision(
                strategy=RetrievalStrategy.HYDE,
                query_analysis=analysis,
                use_metadata_filter=True,
                use_hyde=True,
                explanation="Ambiguous query: using HyDE for better embedding alignment",
            )
        return RoutingDecision(
            strategy=RetrievalStrategy.MULTI_QUERY,
            query_analysis=analysis,
            use_metadata_filter=True,
            use_multi_query=True,
            explanation="Ambiguous query: using multi-query retrieval for broader coverage",
        )

    def _route_out_of_scope(self, analysis: QueryAnalysisResult) -> RoutingDecision:
        """Out of scope → still try, but with low confidence."""
        return RoutingDecision(
            strategy=RetrievalStrategy.HYBRID,
            query_analysis=analysis,
            use_metadata_filter=False,
            explanation="Possibly out of scope: attempting broad hybrid retrieval",
        )
