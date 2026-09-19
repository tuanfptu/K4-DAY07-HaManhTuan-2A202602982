"""BM25 sparse retriever with Vietnamese language support and graceful fallback."""

from __future__ import annotations

import math
import re
from typing import Any

# Optional dependency: rank_bm25
try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_RANK_BM25 = True
except ImportError:
    BM25Okapi = None
    _HAS_RANK_BM25 = False


class BM25Retriever:
    """Sparse retriever implementing BM25 ranking with Vietnamese text tokenization.

    Supports indexing chunks containing 'retrieval_content' or 'content'.
    Attempts to use rank_bm25 if installed, otherwise falls back to a clean,
    built-in pure-Python BM25 implementation.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 parameters.

        Args:
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter (default: 0.75).
        """
        self.k1 = float(k1)
        self.b = float(b)
        self.chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25_engine: Any = None

        # Fallback engine state
        self._corpus_size: int = 0
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._doc_term_freqs: list[dict[str, int]] = []
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}

    def _extract_content(self, chunk: Any) -> str:
        """Extract text from chunk dictionary or object."""
        if isinstance(chunk, dict):
            return str(chunk.get("retrieval_content") or chunk.get("content") or "")
        if hasattr(chunk, "retrieval_content") and getattr(chunk, "retrieval_content"):
            return str(getattr(chunk, "retrieval_content"))
        if hasattr(chunk, "content"):
            return str(getattr(chunk, "content") or "")
        return str(chunk)

    def _tokenize(self, text: str) -> list[str]:
        """Simple Vietnamese-aware tokenization.

        Converts text to lowercase and extracts word tokens by splitting on
        whitespace and punctuation, preserving Vietnamese accented characters.

        Args:
            text: Input text string.

        Returns:
            List of lowercased word tokens.
        """
        if not text or not isinstance(text, str):
            return []

        # Split on whitespace and punctuation using Unicode word regex.
        # \w matches alphanumeric characters and letters across scripts including Vietnamese diacritics.
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return tokens

    def index(self, chunks: list[dict]) -> None:
        """Build the BM25 index from a list of chunk dictionaries.

        Each chunk is expected to have 'content' and/or 'retrieval_content'.
        Preference is given to 'retrieval_content' when present.

        Args:
            chunks: List of chunk dictionaries to index.
        """
        self.chunks = list(chunks)
        self._corpus_size = len(self.chunks)
        self._tokenized_corpus = [
            self._tokenize(self._extract_content(c)) for c in self.chunks
        ]

        if self._corpus_size == 0:
            self._bm25_engine = None
            self._doc_lengths = []
            self._avgdl = 0.0
            self._doc_term_freqs = []
            self._doc_freqs = {}
            self._idf = {}
            return

        if _HAS_RANK_BM25 and BM25Okapi is not None:
            try:
                self._bm25_engine = BM25Okapi(
                    self._tokenized_corpus,
                    k1=self.k1,
                    b=self.b,
                )
            except Exception:
                self._bm25_engine = None
        else:
            self._bm25_engine = None

        # Build fallback / internal index structures
        self._doc_lengths = [len(tokens) for tokens in self._tokenized_corpus]
        total_tokens = sum(self._doc_lengths)
        self._avgdl = total_tokens / self._corpus_size if self._corpus_size > 0 else 0.0

        self._doc_term_freqs = []
        self._doc_freqs = {}
        for tokens in self._tokenized_corpus:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._doc_term_freqs.append(tf)
            for token in tf:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        # Smooth Robertson / BM25+ IDF formula guarantees non-negative values for all terms
        self._idf = {}
        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log(
                1.0 + (self._corpus_size - df + 0.5) / (df + 0.5)
            )

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search indexed chunks with BM25 ranking.

        Args:
            query: Search query string.
            top_k: Maximum number of ranked results to return (default: 20).

        Returns:
            List of (chunk_index, score) pairs sorted descending by score.
        """
        if not query or self._corpus_size == 0 or top_k <= 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # If rank_bm25 engine is available and active
        if self._bm25_engine is not None:
            try:
                raw_scores = self._bm25_engine.get_scores(query_tokens)
                scored = [
                    (i, float(s))
                    for i, s in enumerate(raw_scores)
                    if s > 0.0
                ]
                if not scored:
                    # In case of small corpus where standard Okapi IDF yielded <= 0,
                    # retain documents that actually contain query tokens
                    scored = [
                        (i, float(s))
                        for i, s in enumerate(raw_scores)
                        if any(t in self._tokenized_corpus[i] for t in query_tokens)
                    ]
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]
            except Exception:
                # Fall back to built-in implementation if rank_bm25 fails
                pass

        # Built-in BM25 calculation (accurate, non-negative IDF, Vietnamese-friendly)
        scored_results: list[tuple[int, float]] = []
        for i, tf in enumerate(self._doc_term_freqs):
            doc_len = self._doc_lengths[i]
            if doc_len == 0:
                continue

            score = 0.0
            doc_len_ratio = (doc_len / self._avgdl) if self._avgdl > 0 else 1.0

            for q_term in query_tokens:
                if q_term not in tf:
                    continue
                term_freq = tf[q_term]
                idf = self._idf.get(q_term, 0.0)
                numerator = term_freq * (self.k1 + 1.0)
                denominator = term_freq + self.k1 * (1.0 - self.b + self.b * doc_len_ratio)
                if denominator > 0:
                    score += idf * (numerator / denominator)

            if score > 0.0:
                scored_results.append((i, float(score)))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
