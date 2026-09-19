"""Comprehensive evaluation runner for FPTU HCM Student RAG.

Evaluates:
1. Recall@1 (Hit@1)
2. Recall@5 (Hit@5)
3. MRR (Mean Reciprocal Rank)
4. nDCG@5 (Normalized Discounted Cumulative Gain at 5)
5. Full Evidence@5 (Top-5 contains expected doc AND section evidence)
6. Faithfulness / Agent Accuracy (Agent answer contains gold keywords)
7. Audience Match Rate (Retrieved chunks match expected audience, e.g. student vs faculty)

Usage:
    python scripts/run_advanced_eval.py --dataset dev
    python scripts/run_advanced_eval.py --dataset test
    python scripts/run_advanced_eval.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.advanced_rag.preprocessing.parser import load_corpus
from src.advanced_rag.preprocessing.cleaner import clean_document
from src.advanced_rag.chunking.parent_child_chunker import ParentChildChunker
from src.advanced_rag.retrieval.bm25_retriever import BM25Retriever
from src.advanced_rag.retrieval.dense_retriever import DenseRetriever
from src.advanced_rag.retrieval.hybrid_retriever import HybridRetriever
from src.advanced_rag.retrieval.parent_child_retriever import ParentChildRetriever
from src.advanced_rag.retrieval.metadata_filter import filter_chunks
from src.advanced_rag.routing.query_rewriter import QueryRewriter
from src.advanced_rag.generation.prompt_builder import AnswerGenerator
from src.embeddings import get_embedder


@dataclass
class QueryEvalResult:
    query_id: str
    query: str
    expected_doc: str
    expected_docs: list[str]
    expected_section: str
    expected_audience: str
    gold_keywords: list[str]
    retrieved_docs: list[str] = field(default_factory=list)
    retrieved_audiences: list[str] = field(default_factory=list)
    retrieved_contents: list[str] = field(default_factory=list)
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    reciprocal_rank: float = 0.0
    ndcg_at_5: float = 0.0
    full_evidence_at_5: float = 0.0
    faithfulness_score: float = 0.0
    audience_match_rate: float = 0.0
    agent_answer: str = ""
    latency_ms: float = 0.0


def compute_dcg(relevances: list[float], k: int = 5) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg_at_k(retrieved_docs: list[str], expected_docs: list[str], k: int = 5) -> float:
    gold = set(expected_docs)
    # A source document only contributes once, even if several chunks were retrieved.
    seen: set[str] = set()
    rel = []
    for doc in retrieved_docs[:k]:
        is_new_relevant = doc in gold and doc not in seen
        rel.append(1.0 if is_new_relevant else 0.0)
        seen.add(doc)
    dcg = compute_dcg(rel, k)
    idcg = compute_dcg([1.0] * min(len(gold), k), k)
    return dcg / idcg if idcg > 0.0 else 0.0


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    lowered = text.casefold()
    return bool(terms) and all(str(term).casefold() in lowered for term in terms)


def evaluate_query(
    qdef: dict,
    top_chunks: list[dict],
    agent_answer: str,
    latency_ms: float,
) -> QueryEvalResult:
    expected_docs = qdef.get("source_doc_ids") or [qdef.get("expected_doc", "")]
    expected_docs = [str(d) for d in expected_docs if d]
    expected_doc = expected_docs[0] if expected_docs else ""
    expected_section = qdef.get("expected_section", "")
    expected_audience = qdef.get("expected_audience", "student").lower()
    gold_keywords = qdef.get("gold_keywords", [])

    retrieved_docs = []
    retrieved_audiences = []
    retrieved_contents = []

    for c in top_chunks:
        meta = c.get("chunk_metadata", c.get("metadata", {}))
        retrieved_docs.append(meta.get("doc_id", "unknown"))
        retrieved_audiences.append(str(meta.get("audience", "all")).lower())
        retrieved_contents.append(str(c.get("content", "")))

    # 1. Recall@1
    recall_at_1 = (
        len(set(retrieved_docs[:1]) & set(expected_docs)) / len(expected_docs)
        if expected_docs else 0.0
    )

    # 2. Recall@5
    recall_at_5 = (
        len(set(retrieved_docs[:5]) & set(expected_docs)) / len(expected_docs)
        if expected_docs else 0.0
    )

    # 3. MRR
    rr = 0.0
    for idx, doc in enumerate(retrieved_docs[:5]):
        if doc in expected_docs:
            rr = 1.0 / (idx + 1)
            break

    # 4. nDCG@5
    ndcg_5 = compute_ndcg_at_k(retrieved_docs, expected_docs, k=5)

    # 5. Full Evidence@5: Check if top 5 contains expected doc AND section mention
    full_evidence = 0.0
    evidence_groups = qdef.get("evidence", [])
    combined_top5 = "\n".join(retrieved_contents[:5]).casefold()
    if evidence_groups:
        covered = [
            any(str(phrase).casefold() in combined_top5 for phrase in group.get("phrases", []))
            for group in evidence_groups
        ]
        full_evidence = 1.0 if covered and all(covered) else 0.0
    elif recall_at_5 > 0.0:
        sec_keywords = [w.lower() for w in expected_section.replace("Điều", "").replace("CHƯƠNG", "").replace(".", "").split() if len(w) > 2]
        for doc, content in zip(retrieved_docs[:5], retrieved_contents[:5]):
            if doc in expected_docs:
                content_lower = content.lower()
                # Check if section number or title is present
                if any(kw in content_lower for kw in sec_keywords) or len(sec_keywords) == 0:
                    full_evidence = 1.0
                    break

    # 6. Faithfulness / Agent Accuracy: Tỷ lệ gold keywords xuất hiện trong câu trả lời
    faithfulness = 0.0
    answer_criteria = qdef.get("answer_criteria", [])
    if answer_criteria and agent_answer:
        matched = sum(
            1 for criterion in answer_criteria
            if _contains_all_terms(agent_answer, criterion.get("all_terms", []))
        )
        faithfulness = matched / len(answer_criteria)
    elif gold_keywords and agent_answer:
        answer_lower = agent_answer.lower()
        matched = sum(1 for kw in gold_keywords if kw.lower() in answer_lower)
        faithfulness = matched / len(gold_keywords)
    elif not gold_keywords:
        faithfulness = 1.0 if recall_at_1 > 0 else 0.5

    # 7. Audience Match Rate (Quan trọng nhất riêng cho lớp L3A)
    # Tỷ lệ chunk tìm về có audience đúng (student hoặc all), không nhặt nhầm faculty/staff
    if retrieved_audiences:
        valid_audiences = {expected_audience, "all"}
        matched_aud = sum(1 for aud in retrieved_audiences[:5] if aud in valid_audiences)
        audience_match = matched_aud / min(5, len(retrieved_audiences))
    else:
        audience_match = 1.0

    return QueryEvalResult(
        query_id=qdef.get("id", "unknown"),
        query=qdef.get("query", ""),
        expected_doc=expected_doc,
        expected_docs=expected_docs,
        expected_section=expected_section,
        expected_audience=expected_audience,
        gold_keywords=gold_keywords,
        retrieved_docs=retrieved_docs,
        retrieved_audiences=retrieved_audiences,
        retrieved_contents=retrieved_contents,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        reciprocal_rank=rr,
        ndcg_at_5=ndcg_5,
        full_evidence_at_5=full_evidence,
        faithfulness_score=faithfulness,
        audience_match_rate=audience_match,
        agent_answer=agent_answer,
        latency_ms=latency_ms,
    )


def run_eval(
    dataset_name: str = "dev",
    mode: str = "advanced",
    embedding_provider: str = "auto",
    use_llm: bool = True,
    corpus_dir: str = "data/university",
    output_file: str | None = None,
    benchmark_file: str | None = None,
) -> dict:
    """Execute benchmark evaluation and output metrics."""
    if benchmark_file:
        dataset_name = Path(benchmark_file).stem
    print("=" * 80)
    print(f"  FPTU HCM RAG BENCHMARK EVALUATION — Dataset: {dataset_name.upper()}")
    print(f"  Mode: {mode} | Embedding: {embedding_provider} | LLM Generation: {use_llm}")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Load Queries
    queries: list[dict] = []
    if benchmark_file:
        with open(benchmark_file, encoding="utf-8") as f:
            queries = json.load(f)
    elif dataset_name in ("dev", "all"):
        dev_file = Path("data/benchmark_dev_70.json")
        if dev_file.exists():
            with open(dev_file, encoding="utf-8") as f:
                queries.extend(json.load(f))
    if dataset_name in ("test", "all"):
        test_file = Path("data/benchmark_test_30.json")
        if test_file.exists():
            with open(test_file, encoding="utf-8") as f:
                queries.extend(json.load(f))

    print(f"\n[1/4] Loaded {len(queries)} evaluation queries from dataset '{dataset_name}'.")
    if not queries:
        raise ValueError("Benchmark contains no queries; check --dataset/--benchmark-file.")

    # 2. Ingest Corpus
    print(f"[2/4] Ingesting and chunking corpus from {corpus_dir}...")
    corpus = load_corpus(corpus_dir)
    chunker = ParentChildChunker(parent_chunk_size=1800, child_chunk_size=450, child_overlap=60)
    all_chunks: list[dict] = []
    for meta, body in corpus:
        clean_body = clean_document(body)
        chunks = chunker.chunk(clean_body, metadata=meta)
        all_chunks.extend(chunks)
    print(f"      Total corpus chunks created: {len(all_chunks)}")

    # 3. Build Retrievers
    print("[3/4] Initializing retrievers (BM25 + Dense + RRF)...")
    if embedding_provider == "auto":
        embedding_provider = "gemini" if os.getenv("GEMINI_API_KEY") else "mock"
    embed_fn = get_embedder(embedding_provider)

    bm25 = BM25Retriever()
    dense = DenseRetriever(embedding_fn=embed_fn)
    hybrid = HybridRetriever(bm25=bm25, dense=dense)
    parent_retriever = ParentChildRetriever(hybrid)
    parent_retriever.index(all_chunks)

    # 4. Evaluate queries
    print(f"[4/4] Evaluating {len(queries)} queries across 7 target metrics...\n")
    rewriter = QueryRewriter()
    generator = AnswerGenerator(use_gemini=use_llm)

    results: list[QueryEvalResult] = []
    t_start_all = time.time()

    for idx, qdef in enumerate(queries, start=1):
        q_text = qdef.get("query", "")
        t0 = time.time()

        # Step A: Filter by audience if specified
        req_filter = qdef.get("metadata_filter")
        working_chunks = filter_chunks(all_chunks, req_filter) if req_filter else all_chunks

        # Step B: Rewrite query
        search_query = rewriter.rewrite(q_text)

        # Step C: Retrieve top 5 chunks
        if len(working_chunks) != len(all_chunks):
            # Query-specific filtered search
            filtered_bm25 = BM25Retriever()
            filtered_dense = DenseRetriever(embedding_fn=embed_fn)
            filtered_hybrid = HybridRetriever(bm25=filtered_bm25, dense=filtered_dense)
            filtered_parent = ParentChildRetriever(filtered_hybrid)
            filtered_parent.index(working_chunks)
            retrieved_chunk_dicts = filtered_parent.search(search_query, child_top_k=20, parent_top_k=5)
        else:
            retrieved_chunk_dicts = parent_retriever.search(search_query, child_top_k=20, parent_top_k=5)
        retrieval_latency = (time.time() - t0) * 1000

        # Step D: Generate answer
        contexts = [
            {
                "content": c.get("content", ""),
                "metadata": c.get("chunk_metadata", c.get("metadata", {})),
            }
            for c in retrieved_chunk_dicts
        ]
        # For fast evaluation of 70-100 queries, generate concise answer
        gen_res = generator.generate(q_text, contexts, include_citations=True)
        agent_answer = gen_res["answer"]
        total_latency = (time.time() - t0) * 1000

        # Step E: Evaluate metrics
        qr = evaluate_query(
            qdef=qdef,
            top_chunks=retrieved_chunk_dicts,
            agent_answer=agent_answer,
            latency_ms=total_latency,
        )
        results.append(qr)

        if idx % 10 == 0 or idx == len(queries):
            status = "PASS" if qr.recall_at_1 > 0 else ("TOP5" if qr.recall_at_5 > 0 else "MISS")
            print(f"  [{idx:02d}/{len(queries):02d}] {status} [{qr.query_id}] R@1={qr.recall_at_1:.0f} R@5={qr.recall_at_5:.0f} "
                  f"Faith={qr.faithfulness_score:.2f} AudMatch={qr.audience_match_rate:.2f} | {q_text[:45]}...")

    # Aggregated metrics calculation
    n = len(results)
    avg_r1 = sum(r.recall_at_1 for r in results) / n
    avg_r5 = sum(r.recall_at_5 for r in results) / n
    avg_mrr = sum(r.reciprocal_rank for r in results) / n
    avg_ndcg5 = sum(r.ndcg_at_5 for r in results) / n
    avg_full_ev = sum(r.full_evidence_at_5 for r in results) / n
    avg_faith = sum(r.faithfulness_score for r in results) / n
    avg_aud_match = sum(r.audience_match_rate for r in results) / n
    avg_lat = sum(r.latency_ms for r in results) / n
    total_duration = time.time() - t_start_all

    # Print summary table
    print("\n" + "=" * 80)
    print(f"  AGGREGATED BENCHMARK REPORT: {dataset_name.upper()} SET ({n} QUERIES)")
    print("=" * 80)
    print(f"  1. Recall@1                     : {avg_r1:>7.2%}")
    print(f"  2. Recall@5                     : {avg_r5:>7.2%}")
    print(f"  3. MRR (Mean Reciprocal Rank)   : {avg_mrr:>7.4f}")
    print(f"  4. nDCG@5                       : {avg_ndcg5:>7.4f}")
    print(f"  5. Full Evidence@5              : {avg_full_ev:>7.2%}")
    print(f"  6. Faithfulness / Agent Accuracy: {avg_faith:>7.2%}  <-- (Số 1 trong yêu cầu)")
    print(f"  7. Audience Match Rate (L3A)    : {avg_aud_match:>7.2%}  <-- (Số 2 trong yêu cầu)")
    print(f"  -  Avg Query Latency            : {avg_lat:>7.1f} ms")
    print(f"  -  Total Benchmark Time         : {total_duration:>7.1f} s")
    print("=" * 80)

    # Save to file
    out_path = Path(output_file or f"ket_qua_eval_{dataset_name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"FPTU HCM RAG BENCHMARK EVALUATION: {dataset_name.upper()}\n")
        f.write(f"Total Queries: {n}\n")
        f.write(f"Recall@1: {avg_r1:.4f}\n")
        f.write(f"Recall@5: {avg_r5:.4f}\n")
        f.write(f"MRR: {avg_mrr:.4f}\n")
        f.write(f"nDCG@5: {avg_ndcg5:.4f}\n")
        f.write(f"Full Evidence@5: {avg_full_ev:.4f}\n")
        f.write(f"Faithfulness / Agent Accuracy: {avg_faith:.4f}\n")
        f.write(f"Audience Match Rate: {avg_aud_match:.4f}\n")
        f.write(f"Avg Latency: {avg_lat:.1f}ms\n\n")
        f.write("DETAILS PER QUERY:\n")
        for r in results:
            f.write(f"[{r.query_id}] R@1={r.recall_at_1} R@5={r.recall_at_5} MRR={r.reciprocal_rank:.2f} "
                    f"Faith={r.faithfulness_score:.2f} AudMatch={r.audience_match_rate:.2f}\n")
            f.write(f"  Q: {r.query}\n")
            f.write(f"  Expected: {r.expected_doc} | Top-1: {r.retrieved_docs[0] if r.retrieved_docs else 'N/A'}\n")
            f.write(f"  Answer: {r.agent_answer[:150]}...\n\n")

    print(f"\n[OK] Results successfully exported to: {out_path}")
    return {
        "dataset": dataset_name,
        "n_queries": n,
        "recall_at_1": avg_r1,
        "recall_at_5": avg_r5,
        "mrr": avg_mrr,
        "ndcg_at_5": avg_ndcg5,
        "full_evidence_at_5": avg_full_ev,
        "faithfulness": avg_faith,
        "audience_match_rate": avg_aud_match,
        "avg_latency_ms": avg_lat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG with 7 metrics")
    parser.add_argument("--dataset", choices=["dev", "test", "all"], default="dev", help="Dataset to evaluate")
    parser.add_argument("--mode", default="advanced", help="Retrieval mode")
    parser.add_argument("--embedding", default="auto", help="Embedding provider")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM generation (use fast mock generator)")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--benchmark-file", default=None, help="Custom benchmark JSON (supports source_doc_ids/evidence/answer_criteria)")
    args = parser.parse_args()

    run_eval(
        dataset_name=args.dataset,
        mode=args.mode,
        embedding_provider=args.embedding,
        use_llm=not args.no_llm,
        output_file=args.output,
        benchmark_file=args.benchmark_file,
    )


if __name__ == "__main__":
    main()
