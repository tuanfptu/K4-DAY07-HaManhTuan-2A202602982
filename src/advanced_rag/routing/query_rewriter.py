"""Query rewriting for improved retrieval.

Transforms informal/ambiguous queries into formal, retrieval-optimized forms.
Works deterministically without an LLM, with optional LLM enhancement.
"""

from __future__ import annotations

import re
from typing import Callable


# Vietnamese informal → formal mappings for university domain
_INFORMAL_FORMAL_MAP: dict[str, str] = {
    "trượt môn": "không đạt học phần bắt buộc",
    "trượt": "không đạt",
    "rớt": "không đạt",
    "thi lại": "thi kết thúc học phần lần 2",
    "thi trượt": "không đạt kỳ thi kết thúc học phần",
    "học lại": "đăng ký học lại học phần không đạt",
    "học cải thiện": "đăng ký học cải thiện điểm",
    "bảo lưu": "bảo lưu kết quả học tập tạm dừng học",
    "nghỉ học": "tạm dừng học tập xin nghỉ",
    "đuổi học": "buộc thôi học",
    "cảnh cáo": "cảnh cáo học vụ kết quả học tập kém",
    "nợ môn": "chưa hoàn thành học phần bắt buộc",
    "GPA thấp": "điểm trung bình tích lũy thấp cảnh cáo",
    "xin điểm": "phúc khảo bài thi kết quả học phần",
    "miễn môn": "miễn học phần công nhận tín chỉ",
    "chuyển ngành": "chuyển ngành đào tạo",
    "chuyển trường": "chuyển cơ sở đào tạo",
    "ra trường": "tốt nghiệp cấp bằng đại học",
    "intern": "thực tập OJT On-the-Job Training",
    "đi thực tập": "đăng ký OJT thực tập tại doanh nghiệp",
}

# Domain-specific expansion terms
_DOMAIN_EXPANSIONS: dict[str, list[str]] = {
    "học phí": ["mức thu", "đóng tiền", "tài chính", "khoản phí"],
    "học bổng": ["hỗ trợ tài chính", "miễn giảm", "xét duyệt"],
    "ojt": ["thực tập", "On-the-Job Training", "doanh nghiệp"],
    "tốt nghiệp": ["cấp bằng", "hoàn thành chương trình", "ra trường"],
    "điểm": ["kết quả học tập", "GPA", "trung bình tích lũy"],
    "đăng ký": ["ghi danh", "thủ tục", "quy trình"],
}


class QueryRewriter:
    """Rewrite queries for better retrieval performance.

    Uses Vietnamese-specific informal→formal mappings and domain expansion.
    Optionally uses an LLM for more sophisticated rewriting.
    """

    def __init__(self, llm_fn: Callable[[str], str] | None = None) -> None:
        self.llm_fn = llm_fn

    def rewrite(self, query: str) -> str:
        """Rewrite a query into a more retrieval-friendly form."""
        if self.llm_fn:
            return self._llm_rewrite(query)
        return self._deterministic_rewrite(query)

    def generate_variants(self, query: str, max_variants: int = 3) -> list[str]:
        """Generate multiple query variants for multi-query retrieval."""
        variants: list[str] = [query]

        # Formal rewrite
        formal = self._deterministic_rewrite(query)
        if formal != query and formal not in variants:
            variants.append(formal)

        # Domain-enriched version
        enriched = self._enrich_with_domain(query)
        if enriched != query and enriched not in variants:
            variants.append(enriched)

        # Context-prefixed version
        prefixed = f"Quy định Đại học FPT: {query}"
        if prefixed not in variants:
            variants.append(prefixed)

        if self.llm_fn and len(variants) < max_variants:
            try:
                llm_variants = self._llm_generate_variants(query)
                for v in llm_variants:
                    if v not in variants and len(variants) < max_variants + 1:
                        variants.append(v)
            except Exception:
                pass  # Graceful fallback to deterministic variants

        return variants[:max_variants + 1]

    def _deterministic_rewrite(self, query: str) -> str:
        """Rewrite using keyword substitution."""
        result = query.lower()

        for informal, formal in _INFORMAL_FORMAL_MAP.items():
            if informal in result:
                result = result.replace(informal, formal)

        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]

        return result

    def _enrich_with_domain(self, query: str) -> str:
        """Add domain-specific expansion terms."""
        query_lower = query.lower()
        expansions: list[str] = []

        for keyword, terms in _DOMAIN_EXPANSIONS.items():
            if keyword in query_lower:
                for term in terms:
                    if term.lower() not in query_lower:
                        expansions.append(term)
                        break  # One expansion per keyword

        if expansions:
            return f"{query} ({', '.join(expansions)})"
        return query

    def _llm_rewrite(self, query: str) -> str:
        """Use LLM for sophisticated rewriting."""
        assert self.llm_fn is not None
        prompt = (
            "Viết lại câu hỏi sau thành dạng chính thức, rõ ràng, "
            "phù hợp để tra cứu trong quy chế đào tạo Đại học FPT. "
            "Chỉ trả về câu hỏi đã viết lại, không giải thích.\n\n"
            f"Câu hỏi gốc: {query}\n"
            "Câu hỏi viết lại:"
        )
        try:
            return self.llm_fn(prompt).strip()
        except Exception:
            return self._deterministic_rewrite(query)

    def _llm_generate_variants(self, query: str) -> list[str]:
        """Use LLM to generate query variants."""
        assert self.llm_fn is not None
        prompt = (
            "Tạo 2 câu hỏi khác cùng ý nghĩa với câu hỏi bên dưới, "
            "phù hợp để tra cứu quy chế Đại học FPT. "
            "Mỗi câu một dòng, không đánh số.\n\n"
            f"Câu hỏi: {query}\n"
            "Các câu tương đương:"
        )
        try:
            response = self.llm_fn(prompt).strip()
            return [
                line.strip()
                for line in response.splitlines()
                if line.strip() and len(line.strip()) > 10
            ][:2]
        except Exception:
            return []
