
#!/usr/bin/env python3
"""Benchmark script for FPTU Student Policy RAG system.

Loads the university corpus, chunks documents, builds the retrieval index,
runs 5 benchmark queries, and outputs evaluation metrics.

Usage:
    python bench.py
    python bench.py --mode baseline
    python bench.py --mode advanced --output ket_qua_benchmark.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_corpus_and_chunk(corpus_dir: str, mode: str) -> list[dict]:
    """Load and chunk the corpus."""
    from src.advanced_rag.preprocessing.parser import load_corpus
    from src.advanced_rag.preprocessing.cleaner import clean_document

    corpus = load_corpus(corpus_dir)
    all_chunks: list[dict] = []

    if mode == "baseline":
        from src.chunking import FixedSizeChunker
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        for metadata, body in corpus:
            body = clean_document(body)
            doc_id = metadata.get("doc_id", Path(metadata.get("title", "unknown")).stem)
            chunks = chunker.chunk(body)
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc_id}#{i}",
                    "content": chunk_text,
                    "retrieval_content": chunk_text,
                    "chunk_metadata": {**metadata, "doc_id": doc_id},
                    "metadata": {**metadata, "doc_id": doc_id},
                })
    else:
        from src.advanced_rag.chunking.heading_chunker import HeadingChunker
        chunker = HeadingChunker(chunk_size=800)
        for metadata, body in corpus:
            body = clean_document(body)
            chunks = chunker.chunk(body, metadata=metadata)
            all_chunks.extend(chunks)

    return all_chunks


def run_benchmark(
    mode: str = "advanced",
    embedding_provider: str = "auto",
    corpus_dir: str = "data/university",
    benchmark_path: str = "data/benchmark_queries.json",
    output_file: str | None = None,
) -> str:
    """Run the full benchmark and return results as text."""
    output_lines: list[str] = []

    def log(msg: str = "") -> None:
        output_lines.append(msg)
        print(msg)

    log("=" * 70)
    log(f"  FPTU Student Policy RAG — Benchmark Results")
    log(f"  Mode: {mode}")
    log(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # Load corpus
    log(f"\n  Loading corpus from {corpus_dir}...")
    t0 = time.time()
    chunks = load_corpus_and_chunk(corpus_dir, mode)
    load_time = (time.time() - t0) * 1000
    log(f"  Loaded {len(chunks)} chunks in {load_time:.0f}ms")

    # Load benchmark queries
    bm_path = Path(benchmark_path)
    if not bm_path.exists():
        log(f"\n  ERROR: Benchmark file not found: {benchmark_path}")
        return "\n".join(output_lines)

    with open(bm_path, encoding="utf-8") as f:
        queries = json.load(f)

    log(f"  Benchmark queries: {len(queries)}")

    # Resolve embedding provider
    if embedding_provider == "auto":
        import os
        embedding_provider = "gemini" if os.getenv("GEMINI_API_KEY") else "mock"

    log(f"  Embedding provider: {embedding_provider}")

    from src.embeddings import get_embedder
    embed_fn = get_embedder(embedding_provider)

    # Build index
    log(f"\n  Building retrieval index (mode={mode})...")
    t0 = time.time()

    from src.advanced_rag.retrieval.metadata_filter import filter_chunks

    if mode == "baseline":
        from src.advanced_rag.retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(embedding_fn=embed_fn)
        retriever.index(chunks)
    else:
        from src.advanced_rag.retrieval.bm25_retriever import BM25Retriever
        from src.advanced_rag.retrieval.dense_retriever import DenseRetriever
        from src.advanced_rag.retrieval.hybrid_retriever import HybridRetriever

        bm25 = BM25Retriever()
        dense = DenseRetriever(embedding_fn=embed_fn)
        retriever = HybridRetriever(bm25=bm25, dense=dense)
        retriever.index(chunks)

    index_time = (time.time() - t0) * 1000
    log(f"  Index built in {index_time:.0f}ms")

    # Run queries
    log(f"\n{'─' * 70}")
    log(f"  Running {len(queries)} benchmark queries...")
    log(f"{'─' * 70}\n")

    from src.advanced_rag.evaluation.metrics import BenchmarkRunner

    runner = BenchmarkRunner(benchmark_path)
    eval_results = []

    for qdef in queries:
        query = qdef.get("query", qdef.get("query_vi", ""))
        query_id = qdef.get("id", "unknown")

        # Apply metadata filter if required
        meta_filter = qdef.get("metadata_filter") if qdef.get("requires_filter") else None
        working_chunks = filter_chunks(chunks, meta_filter)

        # Re-index filtered chunks if filtering changed the set
        if meta_filter and len(working_chunks) != len(chunks):
            if mode == "baseline":
                filtered_retriever = DenseRetriever(embedding_fn=embed_fn)
                filtered_retriever.index(working_chunks)
            else:
                filtered_bm25 = BM25Retriever()
                filtered_dense = DenseRetriever(embedding_fn=embed_fn)
                filtered_retriever = HybridRetriever(
                    bm25=filtered_bm25, dense=filtered_dense,
                )
                filtered_retriever.index(working_chunks)
            search_retriever = filtered_retriever
        else:
            search_retriever = retriever
            working_chunks = chunks

        # Search
        t0 = time.time()
        results = search_retriever.search(query, top_k=5) if mode == "baseline" else search_retriever.search(query, final_top_k=5)
        latency = (time.time() - t0) * 1000

        # Map results to doc_ids
        retrieved_docs = []
        retrieved_scores = []
        for idx, score in results:
            if idx < len(working_chunks):
                chunk = working_chunks[idx]
                doc_id = chunk.get("chunk_metadata", chunk.get("metadata", {})).get("doc_id", "unknown")
                retrieved_docs.append(doc_id)
                retrieved_scores.append(score)

        # Evaluate
        qr = runner.evaluate_single(
            query_def=qdef,
            retrieved_docs=retrieved_docs,
            retrieved_scores=retrieved_scores,
            retrieval_method=mode,
            latency_ms=latency,
        )
        eval_results.append(qr)

        hit = "✓" if qr.hit_at_3 else "✗"
        log(f"  {hit} [{query_id}]")
        log(f"    Q: {query}")
        log(f"    Gold: {qr.gold_doc} | Top-1: {retrieved_docs[0] if retrieved_docs else 'N/A'}")
        log(f"    RR: {qr.reciprocal_rank:.2f} | Hit@3: {qr.hit_at_3} | Latency: {latency:.0f}ms")
        if meta_filter:
            log(f"    Filter: {meta_filter}")
        log()

    # Compute report
    report = runner.compute_report(eval_results, strategy_name=mode)
    report_text = runner.format_report(report)
    log(report_text)

    # Summary
    log(f"\n{'=' * 70}")
    log(f"  Summary")
    log(f"{'=' * 70}")
    log(f"  Strategy:   {mode}")
    log(f"  Chunks:     {len(chunks)}")
    log(f"  Hit@1:      {report.hit_at_1:.0%}")
    log(f"  Hit@3:      {report.hit_at_3:.0%}")
    log(f"  MRR:        {report.mrr:.4f}")
    log(f"  nDCG@5:     {report.ndcg_at_5:.4f}")
    log(f"  Avg latency: {report.avg_latency_ms:.0f}ms")
    log(f"{'=' * 70}")

    result_text = "\n".join(output_lines)

    if output_file:
        Path(output_file).write_text(result_text, encoding="utf-8")
        print(f"\n  Results saved to {output_file}")

    return result_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument(
        "--mode",
        choices=["baseline", "advanced"],
        default="advanced",
        help="Retrieval mode",
    )
    parser.add_argument(
        "--embedding",
        choices=["auto", "gemini", "mock", "local", "openai"],
        default="auto",
        help="Embedding provider (default: auto -> gemini if API key present, else mock)",
    )
    parser.add_argument("--output", default="ket_qua_benchmark.txt", help="Output file")
    parser.add_argument("--corpus-dir", default="data/university", help="Corpus directory")
    parser.add_argument("--benchmark", default="data/benchmark_queries.json", help="Benchmark JSON")
    args = parser.parse_args()

    run_benchmark(
        mode=args.mode,
        embedding_provider=args.embedding,
        corpus_dir=args.corpus_dir,
        benchmark_path=args.benchmark,
        output_file=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
