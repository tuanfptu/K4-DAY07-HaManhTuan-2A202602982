"""Query decomposition for multi-hop questions.

Breaks complex multi-part questions into simpler sub-queries
that can be retrieved and answered independently.
"""

from __future__ import annotations

import re
from typing import Callable


class QueryDecomposer:
    """Decompose complex queries into simpler sub-queries.

    Uses Vietnamese connectors and clause patterns to split multi-hop
    questions. Optionally uses an LLM for better decomposition.
    """

    # Vietnamese connectors that often indicate multi-part questions
    _CONNECTORS = [
        r"\bvà\b",
        r"\brồi\b",
        r"\bsau đó\b",
        r"\bđồng thời\b",
        r"\bnếu\s+.{5,}?\s+thì\b",
        r"\bvừa\s+.{5,}?\s+vừa\b",
        r"\bngoài ra\b",
        r"\bthêm vào đó\b",
        r"\bcùng lúc\b",
    ]

    def __init__(self, llm_fn: Callable[[str], str] | None = None) -> None:
        self.llm_fn = llm_fn

    def should_decompose(self, query: str) -> bool:
        """Check if a query would benefit from decomposition."""
        query_lower = query.lower()

        # Count connector matches
        connector_count = sum(
            1 for pattern in self._CONNECTORS
            if re.search(pattern, query_lower)
        )

        # Count distinct topic areas
        topics = set()
        topic_keywords = {
            "học phí": "tuition", "học bổng": "scholarship",
            "ojt": "ojt", "thực tập": "ojt",
            "bảo lưu": "preservation", "nghỉ học": "leave",
            "trượt": "fail", "thi lại": "retake",
            "tốt nghiệp": "graduation", "chuyển ngành": "transfer",
            "cải thiện": "improve", "phúc khảo": "review",
            "đăng ký": "registration",
        }
        for kw, topic in topic_keywords.items():
            if kw in query_lower:
                topics.add(topic)

        return connector_count >= 1 and len(topics) >= 2

    def decompose(self, query: str) -> list[str]:
        """Decompose a query into sub-queries."""
        if self.llm_fn:
            return self._llm_decompose(query)
        return self._deterministic_decompose(query)

    def _deterministic_decompose(self, query: str) -> list[str]:
        """Split query on Vietnamese connectors."""
        query_lower = query.lower().strip()

        # Try splitting on common connectors
        parts = re.split(
            r"(?:\bvà\b|\brồi\b|\bsau đó\b|\bđồng thời\b|\bngoài ra\b|\bcùng lúc\b)",
            query_lower,
        )

        # Handle "nếu...thì..." pattern
        ntt_match = re.search(r"nếu\s+(.+?)\s+thì\s+(.+)", query_lower)
        if ntt_match:
            parts = [ntt_match.group(1).strip(), ntt_match.group(2).strip()]

        # Filter out too-short parts
        sub_queries = [
            p.strip() for p in parts
            if p.strip() and len(p.strip()) > 10
        ]

        # Add context to each sub-query
        contextualized = []
        for sq in sub_queries:
            # Capitalize
            sq = sq[0].upper() + sq[1:] if sq else sq
            # Add question mark if missing
            if sq and sq[-1] not in ".?!":
                sq += "?"
            contextualized.append(sq)

        if len(contextualized) <= 1:
            return [query]

        return contextualized

    def _llm_decompose(self, query: str) -> list[str]:
        """Use LLM for decomposition."""
        assert self.llm_fn is not None
        prompt = (
            "Phân tách câu hỏi phức tạp sau thành các câu hỏi đơn giản, "
            "mỗi câu có thể tra cứu độc lập trong quy chế Đại học FPT. "
            "Mỗi câu hỏi con một dòng, không đánh số.\n\n"
            f"Câu hỏi gốc: {query}\n"
            "Các câu hỏi con:"
        )
        try:
            response = self.llm_fn(prompt).strip()
            sub_queries = [
                line.strip()
                for line in response.splitlines()
                if line.strip() and len(line.strip()) > 10
            ]
            return sub_queries if len(sub_queries) >= 2 else [query]
        except Exception:
            return self._deterministic_decompose(query)
