"""Reranking module with optional model-based reranking.

Supports:
- Cross-encoder reranking (Qwen3-Reranker or similar)
- Fallback to score-based passthrough
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieval results for improved relevance.

    Uses a cross-encoder model when available, falls back to
    passthrough (keeping original ranking) when not.
    """

    def __init__(
        self,
        provider: str = "none",
        model_name: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self._rerank_fn: Callable | None = None
        self._available = False

        if provider == "none":
            return

        if provider == "qwen":
            self._init_qwen(model_name or "Qwen/Qwen3-Reranker-0.6B")
        elif provider == "cohere":
            self._init_cohere()
        else:
            logger.warning(f"Unknown reranker provider: {provider}, using passthrough")

    def _init_qwen(self, model_name: str) -> None:
        """Initialize Qwen reranker (requires transformers)."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            logger.info(f"Loading Qwen reranker: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True,
            )
            self._model.eval()
            self._available = True
            logger.info("Qwen reranker loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Qwen reranker: {e}. Using passthrough.")
            self._available = False

    def _init_cohere(self) -> None:
        """Initialize Cohere reranker (requires API key)."""
        try:
            import cohere
            import os

            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                logger.warning("COHERE_API_KEY not set. Using passthrough.")
                return

            self._cohere_client = cohere.Client(api_key)
            self._available = True
        except Exception as e:
            logger.warning(f"Could not init Cohere reranker: {e}. Using passthrough.")

    @property
    def is_available(self) -> bool:
        """Check if a reranker model is loaded and available."""
        return self._available

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidates and return top_k.

        Each candidate should have at least 'content' and optionally 'score'.

        Args:
            query: The search query.
            candidates: List of candidate dicts with 'content' key.
            top_k: Number of results to return.

        Returns:
            Reranked list of candidates, each with updated 'rerank_score'.
        """
        if not candidates:
            return []

        if not self._available:
            return self._passthrough(candidates, top_k)

        if self.provider == "qwen":
            return self._rerank_qwen(query, candidates, top_k)
        elif self.provider == "cohere":
            return self._rerank_cohere(query, candidates, top_k)

        return self._passthrough(candidates, top_k)

    def _passthrough(
        self,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Passthrough: keep original order/scores."""
        results = []
        for c in candidates[:top_k]:
            result = dict(c)
            result["rerank_score"] = c.get("score", 0.0)
            results.append(result)
        return results

    def _rerank_qwen(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank using Qwen cross-encoder."""
        try:
            import torch

            pairs = [
                (query, c.get("content", c.get("retrieval_content", "")))
                for c in candidates
            ]

            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                scores = self._model(**inputs).logits.squeeze(-1).tolist()

            if isinstance(scores, float):
                scores = [scores]

            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            results = []
            for c, score in scored[:top_k]:
                result = dict(c)
                result["rerank_score"] = float(score)
                results.append(result)

            return results

        except Exception as e:
            logger.warning(f"Qwen reranking failed: {e}. Falling back.")
            return self._passthrough(candidates, top_k)

    def _rerank_cohere(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank using Cohere API."""
        try:
            documents = [
                c.get("content", c.get("retrieval_content", ""))
                for c in candidates
            ]

            response = self._cohere_client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
            )

            results = []
            for item in response.results:
                c = dict(candidates[item.index])
                c["rerank_score"] = item.relevance_score
                results.append(c)

            return results

        except Exception as e:
            logger.warning(f"Cohere reranking failed: {e}. Falling back.")
            return self._passthrough(candidates, top_k)
