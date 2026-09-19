"""Advanced RAG CLI for FPTU Student Policy & Services.

Usage:
    python -m src.advanced_rag.cli "Nếu tôi trượt môn bắt buộc thì phải làm gì?"
    python -m src.advanced_rag.cli --mode baseline "Học phí ngành CNTT?"
    python -m src.advanced_rag.cli --mode advanced --audience student --show-sources "Điều kiện đi OJT?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FPTU Student Policy RAG — Advanced CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="The question to ask (Vietnamese)",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "hybrid", "advanced"],
        default="advanced",
        help="Retrieval mode (default: advanced)",
    )
    parser.add_argument(
        "--embedding",
        choices=["auto", "gemini", "mock", "local", "openai"],
        default="auto",
        help="Embedding provider (default: auto -> gemini if API key present, else mock)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--audience", default=None, help="Filter by audience")
    parser.add_argument("--campus", default=None, help="Filter by campus")
    parser.add_argument("--show-sources", action="store_true", help="Show source documents")
    parser.add_argument("--show-debug", action="store_true", help="Show debug info")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark queries")
    parser.add_argument("--corpus-dir", default="data/university", help="Corpus directory")
    return parser


def _load_and_chunk_corpus(corpus_dir: str, mode: str = "advanced") -> list[dict]:
    """Load corpus and chunk documents."""
    from src.advanced_rag.preprocessing.parser import load_corpus
    from src.advanced_rag.preprocessing.cleaner import clean_document

    corpus = load_corpus(corpus_dir)
    all_chunks: list[dict] = []

    if mode == "baseline":
        # Use original FixedSizeChunker
        from src.chunking import FixedSizeChunker
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        for metadata, body in corpus:
            body = clean_document(body)
            doc_id = metadata.get("doc_id", "unknown")
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
        # Advanced mode indexes compact children and expands hits to coherent,
        # heading-aligned parent sections during retrieval.
        from src.advanced_rag.chunking.parent_child_chunker import ParentChildChunker
        chunker = ParentChildChunker(parent_chunk_size=1800, child_chunk_size=450, child_overlap=60)
        for metadata, body in corpus:
            body = clean_document(body)
            chunks = chunker.chunk(body, metadata=metadata)
            all_chunks.extend(chunks)

    return all_chunks


def _run_query(
    query: str,
    chunks: list[dict],
    mode: str,
    top_k: int,
    audience: str | None,
    campus: str | None,
    show_debug: bool,
    embedding: str = "auto",
) -> dict:
    """Run a single query through the retrieval pipeline."""
    from src.advanced_rag.retrieval.metadata_filter import filter_chunks
    from src.advanced_rag.routing.query_analyzer import QueryAnalyzer
    from src.advanced_rag.routing.query_rewriter import QueryRewriter
    from src.advanced_rag.generation.prompt_builder import AnswerGenerator
    from src.embeddings import get_embedder

    start_time = time.time()
    debug_info: dict = {}

    # Resolve embedding
    if embedding == "auto":
        import os
        embedding = "gemini" if os.getenv("GEMINI_API_KEY") else "mock"
    embed_fn = get_embedder(embedding)
    debug_info["embedding_provider"] = embedding

    # Build metadata filter
    filters: dict | None = None
    if audience or campus:
        filters = {}
        if audience:
            filters["audience"] = audience
        if campus:
            filters["campus"] = campus

    # Apply metadata pre-filter
    working_chunks = filter_chunks(chunks, filters)
    debug_info["chunks_after_filter"] = len(working_chunks)

    if show_debug:
        print(f"\n[DEBUG] Total chunks: {len(chunks)}")
        print(f"[DEBUG] After filter: {len(working_chunks)}")
        print(f"[DEBUG] Embedding provider: {embedding}")

    if mode == "baseline":
        # Dense-only retrieval
        from src.advanced_rag.retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(embedding_fn=embed_fn)
        retriever.index(working_chunks)
        results = retriever.search(query, top_k=top_k)
        debug_info["method"] = "dense_only"
    elif mode == "hybrid":
        # Hybrid BM25 + Dense
        from src.advanced_rag.retrieval.bm25_retriever import BM25Retriever
        from src.advanced_rag.retrieval.dense_retriever import DenseRetriever
        from src.advanced_rag.retrieval.hybrid_retriever import HybridRetriever

        bm25 = BM25Retriever()
        dense = DenseRetriever(embedding_fn=embed_fn)
        hybrid = HybridRetriever(bm25=bm25, dense=dense)
        hybrid.index(working_chunks)
        results = hybrid.search(query, final_top_k=top_k)
        debug_info["method"] = "hybrid_rrf"
    else:
        # Advanced: analyze → route → retrieve
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze(query)

        if show_debug:
            print(f"[DEBUG] Query type: {analysis.query_type.value}")
            print(f"[DEBUG] Detected topics: {analysis.detected_topics}")
            print(f"[DEBUG] Metadata filters: {analysis.metadata_filters}")

        # Rewrite query
        rewriter = QueryRewriter()
        rewritten = rewriter.rewrite(query)
        if show_debug and rewritten != query:
            print(f"[DEBUG] Rewritten query: {rewritten}")

        # Hybrid child retrieval followed by parent expansion.
        from src.advanced_rag.retrieval.bm25_retriever import BM25Retriever
        from src.advanced_rag.retrieval.dense_retriever import DenseRetriever
        from src.advanced_rag.retrieval.hybrid_retriever import HybridRetriever
        from src.advanced_rag.retrieval.parent_child_retriever import ParentChildRetriever

        bm25 = BM25Retriever()
        dense = DenseRetriever(embedding_fn=embed_fn)
        hybrid = HybridRetriever(bm25=bm25, dense=dense)
        parent_retriever = ParentChildRetriever(hybrid)
        parent_retriever.index(working_chunks)
        expanded = parent_retriever.search(rewritten, child_top_k=top_k * 4, parent_top_k=top_k)
        # Normalize to the index/score contract used below while replacing the
        # working set with expanded parent contexts.
        working_chunks = expanded
        results = [(i, item["score"]) for i, item in enumerate(expanded)]

        debug_info["method"] = "heading_parent_child_hybrid"
        debug_info["query_type"] = analysis.query_type.value
        debug_info["rewritten_query"] = rewritten

    latency = (time.time() - start_time) * 1000

    # Build context from results
    contexts = []
    for idx, score in results[:top_k]:
        if idx < len(working_chunks):
            chunk = working_chunks[idx]
            contexts.append({
                "content": chunk.get("content", ""),
                "metadata": chunk.get("chunk_metadata", chunk.get("metadata", {})),
                "score": score,
            })

    # Generate answer
    generator = AnswerGenerator()
    answer_result = generator.generate(query, contexts, include_citations=True)

    debug_info["latency_ms"] = latency
    debug_info["n_results"] = len(contexts)

    return {
        "query": query,
        "answer": answer_result["answer"],
        "contexts": contexts,
        "debug": debug_info,
        "latency_ms": latency,
    }


def _run_benchmark(chunks: list[dict], mode: str, top_k: int) -> None:
    """Run all benchmark queries and report results."""
    from src.advanced_rag.evaluation.metrics import BenchmarkRunner

    runner = BenchmarkRunner()
    try:
        queries = runner.load_queries()
    except FileNotFoundError:
        print("ERROR: data/benchmark_queries.json not found.")
        return

    print(f"\n{'='*60}")
    print(f"  Running benchmark: {len(queries)} queries, mode={mode}")
    print(f"{'='*60}\n")

    results = []
    for qdef in queries:
        query = qdef.get("query", qdef.get("query_vi", ""))
        result = _run_query(
            query=query,
            chunks=chunks,
            mode=mode,
            top_k=top_k,
            audience=qdef.get("metadata_filter", {}).get("audience") if qdef.get("requires_filter") else None,
            campus=None,
            show_debug=False,
        )

        # Extract doc_ids from retrieved contexts
        retrieved_docs = [
            ctx["metadata"].get("doc_id", "unknown")
            for ctx in result["contexts"]
        ]
        retrieved_scores = [ctx["score"] for ctx in result["contexts"]]

        qr = runner.evaluate_single(
            query_def=qdef,
            retrieved_docs=retrieved_docs,
            retrieved_scores=retrieved_scores,
            retrieval_method=result["debug"].get("method", mode),
            latency_ms=result["latency_ms"],
        )
        results.append(qr)

        status = "✓" if qr.hit_at_3 else "✗"
        print(f"  {status} Q: {query[:60]}...")
        print(f"    Gold: {qr.gold_doc} | Top-1: {qr.retrieved_docs[0] if qr.retrieved_docs else 'N/A'}")
        print(f"    RR: {qr.reciprocal_rank:.2f} | Latency: {qr.latency_ms:.0f}ms")
        print()

    report = runner.compute_report(results, strategy_name=f"{mode}")
    print(runner.format_report(report))


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.query and not args.benchmark:
        parser.print_help()
        return 1

    print(f"\n{'='*60}")
    print("  FPTU Student Policy RAG — Advanced Pipeline")
    print(f"  Mode: {args.mode}")
    print(f"{'='*60}")

    # Load and chunk corpus
    print("\n  Loading corpus...")
    chunks = _load_and_chunk_corpus(args.corpus_dir, args.mode)
    print(f"  Loaded {len(chunks)} chunks from {args.corpus_dir}")

    if args.benchmark:
        _run_benchmark(chunks, args.mode, args.top_k)
        return 0

    if args.query:
        result = _run_query(
            query=args.query,
            chunks=chunks,
            mode=args.mode,
            top_k=args.top_k,
            audience=args.audience,
            campus=args.campus,
            show_debug=args.show_debug,
            embedding=args.embedding,
        )

        print(f"\n  Question: {result['query']}")
        print(f"  Latency: {result['latency_ms']:.0f}ms")
        print(f"\n  Answer:")
        print(f"  {result['answer']}")

        if args.show_sources:
            print(f"\n  {'─'*40}")
            print("  Retrieved sources:")
            for i, ctx in enumerate(result["contexts"], 1):
                meta = ctx.get("metadata", {})
                print(
                    f"  [{i}] {meta.get('doc_id', 'unknown')} "
                    f"(score: {ctx.get('score', 0):.3f})"
                )
                print(f"      {ctx['content'][:120]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
