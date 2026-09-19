"""Evidence verification and confidence scoring.

Verifies that generated answers are supported by retrieved evidence.
Provides confidence scoring and abstention when evidence is insufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceItem:
    """A piece of evidence supporting or contradicting a claim."""
    claim: str = ""
    supported: bool = False
    source: str = ""
    section: str = ""
    score: float = 0.0
    matching_text: str = ""


@dataclass
class ConfidenceAssessment:
    """Overall confidence assessment for a RAG response."""
    retrieval_confidence: float = 0.0
    evidence_support: float = 0.0
    overall_confidence: float = 0.0
    should_abstain: bool = False
    abstention_reason: str = ""
    signals: dict[str, float] = field(default_factory=dict)


class EvidenceChecker:
    """Check whether retrieved evidence supports the answer.

    Uses text-overlap heuristics (not an LLM) for verification.
    """

    def check_evidence(
        self,
        answer: str,
        contexts: list[dict[str, Any]],
    ) -> list[EvidenceItem]:
        """Check if key claims in the answer are supported by context.

        Extracts key phrases from the answer and checks if they appear
        in the retrieved contexts.
        """
        if not answer or not contexts:
            return []

        # Extract key phrases (numbers, dates, proper nouns, policy terms)
        key_phrases = self._extract_key_phrases(answer)

        evidence_items: list[EvidenceItem] = []
        for phrase in key_phrases:
            item = self._find_support(phrase, contexts)
            evidence_items.append(item)

        return evidence_items

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract verifiable key phrases from text."""
        phrases: list[str] = []

        # Numbers with context (credits, GPA, money, percentages)
        num_patterns = re.findall(
            r"(\d+(?:[.,]\d+)?)\s*(?:tín chỉ|điểm|%|đồng|VNĐ|triệu|học phần|tuần|ngày|tháng|năm|lần)",
            text,
        )
        for match in num_patterns:
            # Find surrounding context
            idx = text.find(match)
            start = max(0, idx - 30)
            end = min(len(text), idx + len(match) + 30)
            phrases.append(text[start:end].strip())

        # Điều/Khoản references
        dieu_refs = re.findall(r"Điều\s+\d+[a-zA-Z]?", text)
        phrases.extend(dieu_refs)

        # Dates
        dates = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", text)
        phrases.extend(dates)

        return phrases[:10]  # Limit to avoid excessive checking

    def _find_support(
        self,
        phrase: str,
        contexts: list[dict[str, Any]],
    ) -> EvidenceItem:
        """Find support for a phrase in the contexts."""
        phrase_lower = phrase.lower().strip()

        for ctx in contexts:
            content = ctx.get("content", "").lower()
            metadata = ctx.get("metadata", ctx.get("chunk_metadata", {}))

            if phrase_lower in content:
                return EvidenceItem(
                    claim=phrase,
                    supported=True,
                    source=metadata.get("doc_id", "unknown"),
                    section=self._get_section(metadata),
                    score=1.0,
                    matching_text=phrase,
                )

            # Partial match: check if key numbers/terms overlap
            numbers_in_phrase = re.findall(r"\d+(?:[.,]\d+)?", phrase)
            if numbers_in_phrase:
                for num in numbers_in_phrase:
                    if num in content:
                        return EvidenceItem(
                            claim=phrase,
                            supported=True,
                            source=metadata.get("doc_id", "unknown"),
                            section=self._get_section(metadata),
                            score=0.7,
                            matching_text=num,
                        )

        return EvidenceItem(
            claim=phrase,
            supported=False,
            score=0.0,
        )

    def _get_section(self, metadata: dict) -> str:
        """Build section reference from metadata."""
        parts: list[str] = []
        for key in ("chapter", "article", "section"):
            if val := metadata.get(key):
                parts.append(val)
        return " > ".join(parts) if parts else "N/A"


class ConfidenceScorer:
    """Score confidence of RAG responses.

    Combines multiple signals to produce a confidence score.
    NOT a calibrated probability — just a relative quality indicator.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def score(
        self,
        contexts: list[dict[str, Any]],
        evidence_items: list[EvidenceItem] | None = None,
        retrieval_scores: list[float] | None = None,
    ) -> ConfidenceAssessment:
        """Compute confidence assessment.

        Signals used:
        - Number of retrieved results
        - Top retrieval score
        - Score gap between top results
        - Evidence support ratio
        - Metadata match quality
        """
        signals: dict[str, float] = {}

        # Signal 1: Result count
        n_results = len(contexts)
        signals["result_count"] = min(n_results / 3.0, 1.0)

        # Signal 2: Top score
        scores = retrieval_scores or [
            c.get("score", c.get("rerank_score", 0.0))
            for c in contexts
        ]
        top_score = max(scores) if scores else 0.0
        signals["top_score"] = min(max(top_score, 0.0), 1.0)

        # Signal 3: Score gap (top vs 2nd)
        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            gap = sorted_scores[0] - sorted_scores[1]
            signals["score_gap"] = min(gap * 5, 1.0)  # Normalize
        else:
            signals["score_gap"] = 0.5

        # Signal 4: Evidence support
        if evidence_items:
            supported = sum(1 for e in evidence_items if e.supported)
            total = len(evidence_items) if evidence_items else 1
            signals["evidence_support"] = supported / total
        else:
            signals["evidence_support"] = 0.5  # Unknown

        # Signal 5: Metadata relevance
        if contexts:
            has_metadata = sum(
                1 for c in contexts
                if c.get("metadata", c.get("chunk_metadata", {}))
            )
            signals["metadata_quality"] = has_metadata / len(contexts)
        else:
            signals["metadata_quality"] = 0.0

        # Weighted combination
        weights = {
            "result_count": 0.15,
            "top_score": 0.30,
            "score_gap": 0.10,
            "evidence_support": 0.30,
            "metadata_quality": 0.15,
        }

        overall = sum(
            signals.get(k, 0.0) * w for k, w in weights.items()
        )

        retrieval_conf = (
            signals["result_count"] * 0.3
            + signals["top_score"] * 0.5
            + signals["score_gap"] * 0.2
        )

        evidence_conf = signals.get("evidence_support", 0.5)

        should_abstain = overall < self.threshold
        abstention_reason = ""
        if should_abstain:
            if signals["result_count"] < 0.33:
                abstention_reason = "Quá ít kết quả tìm kiếm liên quan."
            elif signals["top_score"] < 0.2:
                abstention_reason = "Điểm tương đồng quá thấp."
            elif signals.get("evidence_support", 1.0) < 0.3:
                abstention_reason = "Thiếu bằng chứng hỗ trợ cho câu trả lời."
            else:
                abstention_reason = "Độ tin cậy tổng thể thấp."

        return ConfidenceAssessment(
            retrieval_confidence=retrieval_conf,
            evidence_support=evidence_conf,
            overall_confidence=overall,
            should_abstain=should_abstain,
            abstention_reason=abstention_reason,
            signals=signals,
        )


class CorrectiveRAG:
    """Corrective retrieval-augmented generation.

    Checks retrieval quality and retries with different strategies
    if results are poor.
    """

    def __init__(
        self,
        confidence_scorer: ConfidenceScorer | None = None,
        max_retries: int = 2,
    ) -> None:
        self.scorer = confidence_scorer or ConfidenceScorer()
        self.max_retries = max_retries

    def should_retry(
        self,
        contexts: list[dict[str, Any]],
        attempt: int = 0,
    ) -> tuple[bool, str]:
        """Check if retrieval should be retried.

        Returns:
            (should_retry, suggestion) tuple.
        """
        if attempt >= self.max_retries:
            return False, "Maximum retry depth reached"

        assessment = self.scorer.score(contexts)

        if assessment.should_abstain:
            suggestions = []
            if assessment.signals.get("result_count", 0) < 0.33:
                suggestions.append("broaden_filters")
            if assessment.signals.get("top_score", 0) < 0.2:
                suggestions.append("rewrite_query")
            if not suggestions:
                suggestions.append("multi_query")

            return True, suggestions[0] if suggestions else "rewrite_query"

        return False, ""

    def get_retry_strategy(self, suggestion: str) -> dict[str, Any]:
        """Get modified retrieval parameters for retry.

        Returns:
            Dict of parameter overrides for the retry attempt.
        """
        strategies = {
            "broaden_filters": {
                "use_metadata_filter": False,
                "top_k_multiplier": 2,
            },
            "rewrite_query": {
                "use_query_rewrite": True,
                "use_multi_query": False,
            },
            "multi_query": {
                "use_multi_query": True,
                "top_k_multiplier": 1.5,
            },
            "hyde": {
                "use_hyde": True,
            },
        }
        return strategies.get(suggestion, strategies["rewrite_query"])
