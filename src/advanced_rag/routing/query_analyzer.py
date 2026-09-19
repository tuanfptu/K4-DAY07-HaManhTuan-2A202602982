"""Query analysis and classification for routing decisions.

Classifies queries into types (FACTUAL, PROCEDURAL, MULTI_HOP, etc.)
and extracts metadata hints for filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryType(Enum):
    """Classification of query intent."""
    FACTUAL = "FACTUAL"
    PROCEDURAL = "PROCEDURAL"
    MULTI_HOP = "MULTI_HOP"
    GLOBAL = "GLOBAL"
    AMBIGUOUS = "AMBIGUOUS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass
class QueryAnalysisResult:
    """Result of analyzing a user query."""
    original_query: str
    query_type: QueryType = QueryType.FACTUAL
    rewritten_queries: list[str] = field(default_factory=list)
    metadata_filters: dict = field(default_factory=dict)
    decomposed_queries: list[str] = field(default_factory=list)
    confidence: float = 0.5
    detected_entities: list[str] = field(default_factory=list)
    detected_topics: list[str] = field(default_factory=list)


# Vietnamese keywords for classification
_PROCEDURAL_KEYWORDS = [
    "làm thế nào", "cách", "thủ tục", "quy trình", "hướng dẫn",
    "đăng ký", "nộp", "xin", "phải làm gì", "bước", "cần làm",
    "liên hệ", "gửi", "khai báo", "yêu cầu",
]

_FACTUAL_KEYWORDS = [
    "bao nhiêu", "là gì", "mấy", "khi nào", "ở đâu",
    "ai", "điểm", "tín chỉ", "học phí", "hạn", "thời gian",
    "điều kiện", "quy định", "yêu cầu",
]

_MULTI_HOP_CONNECTORS = [
    " và ", " rồi ", " sau đó ", " nếu ", " thì ",
    " đồng thời ", " vừa ", " cùng lúc ",
]

_GLOBAL_KEYWORDS = [
    "tổng quan", "tất cả", "toàn bộ", "liệt kê", "danh sách",
    "so sánh", "khác nhau", "giống nhau",
]

_SCOPE_KEYWORDS = [
    "fpt", "đại học", "trường", "sinh viên", "học phần",
    "tín chỉ", "gpa", "ojt", "học bổng", "học phí",
    "ký túc", "thư viện", "campus", "fap", "bảo lưu",
    "thực tập", "tốt nghiệp", "điểm", "phúc khảo",
]

# Topic → metadata mapping
_TOPIC_AUDIENCE_MAP = {
    "sinh viên": "student",
    "giảng viên": "faculty",
    "nhân viên": "staff",
    "giáo viên": "faculty",
    "thầy": "faculty",
    "cô": "faculty",
}

_TOPIC_DEPARTMENT_MAP = {
    "học phí": "finance",
    "tài chính": "finance",
    "học bổng": "scholarship",
    "ojt": "ojt",
    "thực tập": "ojt",
    "trao đổi": "international",
    "quốc tế": "international",
    "exchange": "international",
}

_TOPIC_CATEGORY_MAP = {
    "quy chế": "academic_regulation",
    "quy định": "academic_regulation",
    "điều": "academic_regulation",
    "học phí": "tuition",
    "học bổng": "scholarship",
    "ojt": "ojt",
    "thực tập": "ojt",
    "trao đổi": "international_exchange",
    "cơ sở": "campus_services",
    "phòng": "support_routing",
    "ban": "support_routing",
    "liên hệ": "support_routing",
}


class QueryAnalyzer:
    """Analyze and classify incoming queries.

    Uses keyword-based heuristics for deterministic, reproducible
    classification without requiring an LLM.
    """

    def analyze(self, query: str) -> QueryAnalysisResult:
        """Analyze a query and return classification + metadata hints."""
        query_lower = query.lower().strip()

        result = QueryAnalysisResult(original_query=query)

        # Detect if query is in scope
        in_scope = any(kw in query_lower for kw in _SCOPE_KEYWORDS)
        if not in_scope and len(query_lower) > 5:
            # Short queries might still be in scope
            result.query_type = QueryType.OUT_OF_SCOPE
            result.confidence = 0.3
            return result

        # Classify query type
        result.query_type = self._classify_type(query_lower)

        # Extract metadata filters
        result.metadata_filters = self._extract_filters(query_lower)

        # Detect entities and topics
        result.detected_entities = self._detect_entities(query_lower)
        result.detected_topics = self._detect_topics(query_lower)

        # Generate rewritten queries
        result.rewritten_queries = self._rewrite(query, query_lower)

        # Decompose multi-hop queries
        if result.query_type == QueryType.MULTI_HOP:
            result.decomposed_queries = self._decompose(query_lower)

        result.confidence = self._estimate_confidence(result)

        return result

    def _classify_type(self, query_lower: str) -> QueryType:
        """Classify query type based on keyword patterns."""
        # Check multi-hop first (has connectors + multiple topic areas)
        connector_count = sum(
            1 for c in _MULTI_HOP_CONNECTORS if c in query_lower
        )
        if connector_count >= 1:
            topic_count = sum(
                1 for cat in _TOPIC_CATEGORY_MAP
                if cat in query_lower
            )
            if topic_count >= 2 or connector_count >= 2:
                return QueryType.MULTI_HOP

        # Global queries
        if any(kw in query_lower for kw in _GLOBAL_KEYWORDS):
            return QueryType.GLOBAL

        # Procedural queries
        procedural_hits = sum(
            1 for kw in _PROCEDURAL_KEYWORDS if kw in query_lower
        )
        if procedural_hits >= 1:
            return QueryType.PROCEDURAL

        # Factual queries
        factual_hits = sum(
            1 for kw in _FACTUAL_KEYWORDS if kw in query_lower
        )
        if factual_hits >= 1:
            return QueryType.FACTUAL

        # Short or vague queries
        if len(query_lower.split()) <= 3:
            return QueryType.AMBIGUOUS

        return QueryType.FACTUAL

    def _extract_filters(self, query_lower: str) -> dict:
        """Extract metadata filter hints from query text."""
        filters: dict = {}

        # Audience detection
        for keyword, audience in _TOPIC_AUDIENCE_MAP.items():
            if keyword in query_lower:
                filters["audience"] = audience
                break

        # Default to student for most queries
        if "audience" not in filters:
            filters["audience"] = "student"

        # Campus detection
        if "hcm" in query_lower or "hồ chí minh" in query_lower:
            filters["campus"] = "hcm"

        # Department detection
        for keyword, dept in _TOPIC_DEPARTMENT_MAP.items():
            if keyword in query_lower:
                filters["department"] = dept
                break

        return filters

    def _detect_entities(self, query_lower: str) -> list[str]:
        """Detect named entities and key terms."""
        entities: list[str] = []

        # Detect Điều references
        dieu_match = re.findall(r"điều\s+(\d+)", query_lower)
        for num in dieu_match:
            entities.append(f"Điều {num}")

        # Detect GPA/credit numbers
        num_match = re.findall(r"(\d+(?:\.\d+)?)\s*(?:tín chỉ|điểm|gpa)", query_lower)
        for num in num_match:
            entities.append(f"số: {num}")

        # Detect semester references
        if any(s in query_lower for s in ["fall", "spring", "summer"]):
            entities.append("học kỳ")

        return entities

    def _detect_topics(self, query_lower: str) -> list[str]:
        """Detect topic areas from query."""
        topics: list[str] = []
        for keyword, category in _TOPIC_CATEGORY_MAP.items():
            if keyword in query_lower and category not in topics:
                topics.append(category)
        return topics

    def _rewrite(self, query: str, query_lower: str) -> list[str]:
        """Generate query variants for better retrieval."""
        variants: list[str] = [query]

        # Add formal version
        informal_to_formal = {
            "trượt môn": "không đạt học phần",
            "rớt": "không đạt",
            "thi lại": "thi kết thúc học phần lần 2",
            "học lại": "đăng ký học lại học phần không đạt",
            "bảo lưu": "bảo lưu kết quả học tập",
            "nghỉ học": "tạm dừng học tập",
            "đuổi học": "buộc thôi học",
            "cảnh cáo": "cảnh cáo học vụ",
        }
        rewritten = query_lower
        for informal, formal in informal_to_formal.items():
            if informal in query_lower:
                rewritten = rewritten.replace(informal, formal)

        if rewritten != query_lower:
            variants.append(rewritten)

        # Add context-enriched version
        enriched = f"Quy định của Đại học FPT về: {query}"
        variants.append(enriched)

        return variants

    def _decompose(self, query_lower: str) -> list[str]:
        """Decompose a multi-hop query into sub-queries."""
        sub_queries: list[str] = []

        # Split on connectors
        parts = re.split(
            r"(?:và|rồi|sau đó|đồng thời|nếu.*?thì)",
            query_lower,
        )

        for part in parts:
            part = part.strip()
            if len(part) > 10:
                sub_queries.append(part)

        # If decomposition didn't help, return original
        if len(sub_queries) <= 1:
            return [query_lower]

        return sub_queries

    def _estimate_confidence(self, result: QueryAnalysisResult) -> float:
        """Estimate how confident we are in the analysis."""
        score = 0.5

        if result.query_type == QueryType.OUT_OF_SCOPE:
            return 0.3

        if result.detected_topics:
            score += 0.1 * min(len(result.detected_topics), 3)

        if result.detected_entities:
            score += 0.1

        if result.metadata_filters:
            score += 0.1

        return min(score, 1.0)
