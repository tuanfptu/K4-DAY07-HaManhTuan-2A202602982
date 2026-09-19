"""Multi-query retrieval with LLM variant generation and deterministic fallback."""

from __future__ import annotations

import re
from typing import Any, Callable

from .rrf import reciprocal_rank_fusion


class MultiQueryRetriever:
    """Expands user queries into multiple perspectives and fuses results via RRF."""

    def __init__(
        self,
        retriever: Any,
        llm_fn: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize MultiQueryRetriever.

        Args:
            retriever: Any retriever instance implementing search(query, top_k).
            llm_fn: Optional callable taking a prompt string and returning LLM generated text.
        """
        self.retriever = retriever
        self.llm_fn = llm_fn

    def _deterministic_variants(self, query: str) -> list[str]:
        """Generate deterministic query variants using domain terms and synonyms.

        Args:
            query: Cleaned query text.

        Returns:
            List of domain-expanded query variant strings.
        """
        variants: list[str] = []
        q_lower = query.lower()

        # 1. Add domain institution term
        if "fpt" not in q_lower:
            variants.append(f"{query} Đại học FPT")
        else:
            variants.append(f"{query} cơ sở HCM")

        # 2. Add regulatory / policy context
        if "quy chế" not in q_lower and "quy định" not in q_lower:
            variants.append(f"quy chế {query}")
        else:
            variants.append(f"thủ tục quy định {query}")

        # 3. Add student target audience term
        if "sinh viên" not in q_lower:
            variants.append(f"sinh viên {query}")
        else:
            variants.append(f"hướng dẫn học vụ {query}")

        # 4. Synonym replacement for key university terms
        synonym_map = {
            "học phí": "chi phí đào tạo",
            "học bổng": "chính sách học bổng",
            "ojt": "thực tập doanh nghiệp",
            "thi lại": "học lại",
            "fap": "cổng thông tin đào tạo",
            "bảo lưu": "tạm hoãn học tập",
            "phúc khảo": "khiếu nại điểm thi",
            "điểm danh": "chuyên cần vắng học",
        }
        for term, syn in synonym_map.items():
            if term in q_lower:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                rephrased = pattern.sub(syn, query)
                if rephrased.lower() != q_lower:
                    variants.append(rephrased)
                break

        return variants

    def generate_queries(self, query: str) -> list[str]:
        """Generate query variants using LLM if available, or deterministic fallback.

        Args:
            query: Input user query.

        Returns:
            List of query variant strings including the original query.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        variants: list[str] = [clean_query]

        # Try LLM-based query generation if llm_fn is provided
        if self.llm_fn is not None:
            try:
                prompt = (
                    f"Bạn là trợ lý tìm kiếm thông tin quy chế đại học.\n"
                    f"Hãy tạo 3 câu hỏi tìm kiếm tương đương hoặc mở rộng cho câu hỏi sau:\n"
                    f"\"{clean_query}\"\n"
                    f"Trả về mỗi câu hỏi trên một dòng riêng biệt, không đánh số thứ tự."
                )
                response = self.llm_fn(prompt)
                if isinstance(response, str) and response.strip():
                    lines = response.splitlines()
                    for line in lines:
                        cleaned_line = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
                        if (
                            cleaned_line
                            and cleaned_line.lower() != clean_query.lower()
                            and cleaned_line not in variants
                        ):
                            variants.append(cleaned_line)
            except Exception:
                pass

        # If LLM did not generate additional variants, use deterministic fallback
        if len(variants) <= 1:
            variants.extend(self._deterministic_variants(clean_query))

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for q in variants:
            norm = q.lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(q.strip())

        return deduped

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Retrieve using all generated query variants and fuse results using RRF.

        Args:
            query: Input search query.
            top_k: Number of final fused results to return (default: 10).

        Returns:
            List of (chunk_index, rrf_score) pairs sorted descending by RRF score.
        """
        if not query or not query.strip() or top_k <= 0:
            return []

        query_variants = self.generate_queries(query)
        if not query_variants:
            return []

        rankings: list[list[tuple[int, float]]] = []
        candidate_k = max(top_k * 2, 20)

        for variant in query_variants:
            results = self.retriever.search(variant, top_k=candidate_k)
            if results:
                rankings.append(results)

        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings)
        return fused[:top_k]
