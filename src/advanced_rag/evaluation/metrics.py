"""Retrieval evaluation metrics and benchmarking.

Implements standard IR metrics: Hit@k, MRR, nDCG@k, Recall@k.
Also handles benchmark execution and comparison reporting.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    query_id: str
    query: str
    gold_doc: str
    gold_section: str
    retrieved_docs: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    reciprocal_rank: float = 0.0
    retrieval_method: str = ""
    latency_ms: float = 0.0
    agent_answer: str = ""
    answer_correct: bool = False


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    strategy_name: str
    query_results: list[QueryResult] = field(default_factory=list)
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    avg_latency_ms: float = 0.0
    total_queries: int = 0


def _dcg(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved_docs: list[str], gold_doc: str, k: int = 5) -> float:
    """Compute nDCG@k for a single query.

    Binary relevance: 1 if doc matches gold_doc, 0 otherwise.
    """
    relevances = [
        1.0 if doc == gold_doc else 0.0
        for doc in retrieved_docs[:k]
    ]

    dcg = _dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = _dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved_docs: list[str], gold_doc: str, k: int) -> bool:
    """Check if gold document appears in top-k retrieved docs."""
    return gold_doc in retrieved_docs[:k]


def reciprocal_rank(retrieved_docs: list[str], gold_doc: str) -> float:
    """Compute Reciprocal Rank for a single query."""
    for i, doc in enumerate(retrieved_docs):
        if doc == gold_doc:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    retrieved_docs: list[str],
    gold_docs: list[str],
    k: int = 5,
) -> float:
    """Compute Recall@k: fraction of gold docs found in top-k."""
    if not gold_docs:
        return 0.0
    found = sum(1 for g in gold_docs if g in retrieved_docs[:k])
    return found / len(gold_docs)


class BenchmarkRunner:
    """Run benchmark queries and compute evaluation metrics.

    Loads benchmark queries from JSON, runs them through a retrieval
    pipeline, and computes standard IR metrics.
    """

    def __init__(self, benchmark_path: str | Path = "data/benchmark_queries.json") -> None:
        self.benchmark_path = Path(benchmark_path)
        self.queries: list[dict] = []

    def load_queries(self) -> list[dict]:
        """Load benchmark queries from JSON file."""
        if not self.benchmark_path.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: {self.benchmark_path}"
            )

        with open(self.benchmark_path, encoding="utf-8") as f:
            self.queries = json.load(f)

        return self.queries

    def evaluate_single(
        self,
        query_def: dict,
        retrieved_docs: list[str],
        retrieved_scores: list[float] | None = None,
        retrieval_method: str = "",
        latency_ms: float = 0.0,
        agent_answer: str = "",
    ) -> QueryResult:
        """Evaluate retrieval for a single query."""
        gold_doc = query_def.get("expected_doc", "")
        scores = retrieved_scores or [0.0] * len(retrieved_docs)

        return QueryResult(
            query_id=query_def.get("id", ""),
            query=query_def.get("query", ""),
            gold_doc=gold_doc,
            gold_section=query_def.get("expected_section", ""),
            retrieved_docs=retrieved_docs,
            retrieved_scores=scores,
            hit_at_1=hit_at_k(retrieved_docs, gold_doc, 1),
            hit_at_3=hit_at_k(retrieved_docs, gold_doc, 3),
            hit_at_5=hit_at_k(retrieved_docs, gold_doc, 5),
            reciprocal_rank=reciprocal_rank(retrieved_docs, gold_doc),
            retrieval_method=retrieval_method,
            latency_ms=latency_ms,
            agent_answer=agent_answer,
        )

    def compute_report(
        self,
        results: list[QueryResult],
        strategy_name: str = "default",
    ) -> EvalReport:
        """Compute aggregated metrics from query results."""
        n = len(results)
        if n == 0:
            return EvalReport(strategy_name=strategy_name)

        report = EvalReport(
            strategy_name=strategy_name,
            query_results=results,
            total_queries=n,
            hit_at_1=sum(r.hit_at_1 for r in results) / n,
            hit_at_3=sum(r.hit_at_3 for r in results) / n,
            recall_at_5=sum(
                1.0 if r.hit_at_5 else 0.0 for r in results
            ) / n,
            mrr=sum(r.reciprocal_rank for r in results) / n,
            ndcg_at_5=sum(
                ndcg_at_k(r.retrieved_docs, r.gold_doc, 5)
                for r in results
            ) / n,
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
        )

        return report

    def format_report(self, report: EvalReport) -> str:
        """Format an evaluation report as readable text."""
        lines: list[str] = [
            f"\n{'='*60}",
            f"  Evaluation Report: {report.strategy_name}",
            f"{'='*60}",
            f"  Total queries: {report.total_queries}",
            f"  Hit@1:         {report.hit_at_1:.2%}",
            f"  Hit@3:         {report.hit_at_3:.2%}",
            f"  Recall@5:      {report.recall_at_5:.2%}",
            f"  MRR:           {report.mrr:.4f}",
            f"  nDCG@5:        {report.ndcg_at_5:.4f}",
            f"  Avg latency:   {report.avg_latency_ms:.1f}ms",
            f"{'='*60}",
            "",
            "  Per-query results:",
        ]

        for r in report.query_results:
            status = "✓" if r.hit_at_3 else "✗"
            lines.append(
                f"  {status} [{r.query_id}] RR={r.reciprocal_rank:.2f} "
                f"| {r.query[:50]}..."
            )
            if r.retrieved_docs:
                lines.append(f"    Top-1: {r.retrieved_docs[0]}")
                lines.append(f"    Gold:  {r.gold_doc}")

        return "\n".join(lines)

    def compare_strategies(
        self,
        reports: list[EvalReport],
    ) -> str:
        """Generate a comparison table of multiple strategies."""
        lines: list[str] = [
            f"\n{'='*80}",
            "  Strategy Comparison",
            f"{'='*80}",
            f"  {'Strategy':<30} {'Hit@1':>7} {'Hit@3':>7} {'R@5':>7} {'MRR':>7} {'nDCG@5':>7} {'Latency':>9}",
            f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9}",
        ]

        for r in reports:
            lines.append(
                f"  {r.strategy_name:<30} "
                f"{r.hit_at_1:>6.0%} "
                f"{r.hit_at_3:>6.0%} "
                f"{r.recall_at_5:>6.0%} "
                f"{r.mrr:>6.3f} "
                f"{r.ndcg_at_5:>6.3f} "
                f"{r.avg_latency_ms:>7.1f}ms"
            )

        lines.append(f"{'='*80}")
        return "\n".join(lines)
