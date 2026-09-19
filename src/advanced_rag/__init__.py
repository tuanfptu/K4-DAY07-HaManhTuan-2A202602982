"""Advanced RAG architecture and pipelines for Vietnamese university documents."""

from __future__ import annotations

from .config import RAGConfig, load_config
from .preprocessing.cleaner import (
    clean_document,
    normalize_whitespace,
    remove_boilerplate,
)
from .preprocessing.parser import (
    load_corpus,
    parse_frontmatter,
    parse_markdown_file,
)
from .preprocessing.validator import (
    validate_corpus,
    validate_document,
)
from .schemas import (
    Chunk,
    ChunkMetadata,
    Evidence,
    QueryAnalysis,
    QueryType,
    RAGResponse,
    RetrievalResult,
)

__all__ = [
    # Configuration
    "RAGConfig",
    "load_config",
    # Data Models & Schemas
    "ChunkMetadata",
    "Chunk",
    "RetrievalResult",
    "QueryAnalysis",
    "QueryType",
    "Evidence",
    "RAGResponse",
    # Preprocessing Parser
    "parse_frontmatter",
    "parse_markdown_file",
    "load_corpus",
    # Preprocessing Cleaner
    "clean_document",
    "remove_boilerplate",
    "normalize_whitespace",
    # Preprocessing Validator
    "validate_document",
    "validate_corpus",
]
