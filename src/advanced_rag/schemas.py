"""Data schemas and transfer models for the Advanced RAG system."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """Classification of user query intent."""

    FACTUAL = "FACTUAL"
    PROCEDURAL = "PROCEDURAL"
    MULTI_HOP = "MULTI_HOP"
    GLOBAL = "GLOBAL"
    AMBIGUOUS = "AMBIGUOUS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

    def __str__(self) -> str:
        return self.value


@dataclass
class ChunkMetadata:
    """Rich metadata associated with a document chunk.

    Attributes:
        doc_id: Unique identifier of the source document.
        chunk_id: Unique identifier of the chunk within the document.
        title: Title of the source document.
        chapter: Optional chapter name or number (e.g., 'CHƯƠNG I').
        article: Optional article number or name (e.g., 'Điều 1').
        section: Optional section heading or path.
        audience: Target audience (e.g., 'student', 'faculty', 'all').
        campus: Campus scope (e.g., 'hcm', 'all').
        department: Relevant university department or office.
        category: Document category (e.g., 'academic_regulation', 'tuition').
        source_url: Canonical URL where document is published.
        document_version: Version identifier or year.
        language: ISO language code ('vi', 'en').
        source_type: Type of source (e.g., 'official_public_web').
        retrieved_at: Date or timestamp when source was crawled/retrieved.
        extra: Additional arbitrary key-value metadata.
    """

    doc_id: str = ""
    chunk_id: str = ""
    title: str = ""
    chapter: str | None = None
    article: str | None = None
    section: str | None = None
    audience: str = "student"
    campus: str = "all"
    department: str = ""
    category: str = ""
    source_url: str = ""
    document_version: str = ""
    language: str = "vi"
    source_type: str = "official_public_web"
    retrieved_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkMetadata:
        """Create ChunkMetadata from a dictionary, mapping known fields and storing rest in extra."""
        if not data:
            return cls()

        known_fields = {f.name for f in cls.__dataclass_fields__.values() if f.name != "extra"}
        init_kwargs: dict[str, Any] = {}
        extra_kwargs: dict[str, Any] = {}

        for k, v in data.items():
            if k in known_fields:
                init_kwargs[k] = v
            else:
                extra_kwargs[k] = v

        if extra_kwargs:
            init_kwargs["extra"] = extra_kwargs

        return cls(**init_kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return asdict(self)


@dataclass
class Chunk:
    """A granular unit of indexed text.

    Attributes:
        id: Unique identifier for the chunk (e.g., 'doc_01_chunk_004').
        content: Raw text content of the chunk for generation.
        retrieval_content: Enriched text representation used for sparse/dense search.
        metadata: Structured ChunkMetadata.
        embedding: Optional precomputed vector embedding.
    """

    id: str
    content: str
    retrieval_content: str = ""
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.retrieval_content:
            self.retrieval_content = self.content

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a dictionary."""
        return asdict(self)


@dataclass
class RetrievalResult:
    """Individual chunk match returned by a retriever.

    Attributes:
        chunk: The matched Chunk instance.
        score: Relevance or fusion score (higher is more relevant).
        method: Method used ('bm25', 'dense', 'hybrid', 'reranked').
    """

    chunk: Chunk
    score: float
    method: str = "hybrid"

    def to_dict(self) -> dict[str, Any]:
        """Serialize retrieval result to a dictionary."""
        return asdict(self)


@dataclass
class QueryAnalysis:
    """Semantic analysis and classification of a user query.

    Attributes:
        original_query: Raw user input text.
        query_type: Classified QueryType intent.
        rewritten_queries: Alternative query variations (e.g., formal terms).
        metadata_filters: Extracted key-value filters (e.g., campus, audience).
        decomposed_queries: Sub-queries for multi-hop retrieval.
        confidence: Estimated classification confidence (0.0 - 1.0).
        detected_entities: Specific entities found in query.
        detected_topics: Topic categories detected in query.
    """

    original_query: str
    query_type: str | QueryType = QueryType.FACTUAL
    rewritten_queries: list[str] = field(default_factory=list)
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    decomposed_queries: list[str] = field(default_factory=list)
    confidence: float = 0.5
    detected_entities: list[str] = field(default_factory=list)
    detected_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize query analysis to a dictionary."""
        res = asdict(self)
        if isinstance(self.query_type, Enum):
            res["query_type"] = self.query_type.value
        return res


@dataclass
class Evidence:
    """An individual claim verified against retrieved source context.

    Attributes:
        claim: Factual claim statement evaluated.
        supported: Whether the claim is directly supported by cited text.
        source: Name or identifier of the supporting source document.
        section: Section or heading where evidence was found.
        score: Confidence or verification score.
    """

    claim: str
    supported: bool
    source: str
    section: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize evidence to a dictionary."""
        return asdict(self)


@dataclass
class RAGResponse:
    """Complete end-to-end response from the Advanced RAG pipeline.

    Attributes:
        answer: Generated answer text.
        sources: Ranked list of retrieval results used.
        evidences: Verified evidence items backing up the answer.
        confidence: Overall confidence score in the generated answer.
        query_analysis: Optional QueryAnalysis detailing query interpretation.
        retrieval_method: Retrieval mechanism used ('bm25', 'dense', 'hybrid', 'reranked').
        latency_ms: Total pipeline processing duration in milliseconds.
    """

    answer: str
    sources: list[RetrievalResult] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    query_analysis: QueryAnalysis | None = None
    retrieval_method: str = "hybrid"
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize RAG response to a dictionary."""
        res = asdict(self)
        if self.query_analysis and isinstance(self.query_analysis.query_type, Enum):
            res["query_analysis"]["query_type"] = self.query_analysis.query_type.value
        return res
