"""
Contextual retrieval chunker for Vietnamese university policy documents.
"""

from __future__ import annotations

from typing import Any

from .heading_chunker import HeadingChunker
from .utils import parse_frontmatter


class ContextualChunker:
    """
    Deterministic contextual retrieval chunker.

    Implements Anthropic's Contextual Retrieval concept without requiring LLM inference:
    For each chunk, prepends document context (title, chapter, article, audience)
    derived deterministically from metadata and document heading hierarchy.

    Format of retrieval_content:
        "Document: {title}. Chapter: {chapter}. Article: {article}. Audience: {audience}.\n\n{original_text}"

    This dramatically improves vector retrieval accuracy for Vietnamese regulations
    by grounding ambiguous clauses (e.g. "Khoản 2: Sinh viên phải...") with their
    complete regulatory context (which policy, chapter, article, and audience).
    """

    def __init__(
        self,
        base_chunker: Any = None,
        chunk_size: int = 800,
        default_audience: str = "sinh viên",
    ) -> None:
        """
        Initialize the ContextualChunker.

        Args:
            base_chunker: Underlying chunker instance (defaults to HeadingChunker).
            chunk_size: Target chunk size in characters if base_chunker is created.
            default_audience: Fallback audience if not specified in document metadata.
        """
        self.chunk_size = chunk_size
        self.default_audience = default_audience

        if base_chunker is not None:
            self.base_chunker = base_chunker
        else:
            # Default to HeadingChunker with prefix disabled on base chunks
            # since ContextualChunker constructs its own rich contextual header
            self.base_chunker = HeadingChunker(
                chunk_size=self.chunk_size,
                include_context_prefix=False,
                min_chunk_size=min(200, self.chunk_size),
            )

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Chunk document and prepend deterministic contextual metadata to each chunk.

        Args:
            text: Raw document text (Markdown or plain text).
            metadata: Document-level metadata dictionary.

        Returns:
            List of chunk dictionaries with contextualized 'retrieval_content'.
        """
        if not text or not text.strip():
            return []

        doc_meta: dict[str, Any] = dict(metadata or {})

        # Parse YAML frontmatter if present in raw text
        front_meta, clean_text = parse_frontmatter(text)
        for k, v in front_meta.items():
            if k not in doc_meta:
                doc_meta[k] = v

        clean_text = clean_text.strip()
        if not clean_text:
            return []

        doc_id = str(doc_meta.get("doc_id") or doc_meta.get("id") or "doc").strip()

        # Run base chunker
        try:
            raw_chunks = self.base_chunker.chunk(clean_text, metadata=doc_meta)
        except TypeError:
            raw_chunks = self.base_chunker.chunk(clean_text)

        if not raw_chunks:
            return []

        enriched_chunks: list[dict[str, Any]] = []

        for idx, item in enumerate(raw_chunks, start=1):
            if isinstance(item, dict):
                original_text = item.get("content", "").strip()
                chunk_id = item.get("id") or item.get("chunk_id") or f"{doc_id}__part_{idx}"
                chunk_meta = dict(item.get("chunk_metadata", {}))
            else:
                original_text = str(item).strip()
                chunk_id = f"{doc_id}__part_{idx}"
                chunk_meta = dict(doc_meta)

            if not original_text:
                continue

            # Extract context dimensions
            title = (
                chunk_meta.get("title")
                or doc_meta.get("title")
                or doc_id
            )
            chapter = chunk_meta.get("chapter") or doc_meta.get("chapter") or "N/A"
            article = chunk_meta.get("article") or doc_meta.get("article") or "N/A"
            audience = (
                doc_meta.get("audience")
                or chunk_meta.get("audience")
                or self.default_audience
            )

            # Deterministic contextual header
            # Format: "Document: {title}. Chapter: {chapter}. Article: {article}. Audience: {audience}."
            context_header = (
                f"Document: {title}. Chapter: {chapter}. Article: {article}. Audience: {audience}."
            )

            retrieval_content = f"{context_header}\n\n{original_text}"

            # Update chunk metadata with context information
            chunk_meta.update(
                {
                    "title": title,
                    "chapter": chapter,
                    "article": article,
                    "audience": audience,
                    "context_header": context_header,
                }
            )

            chunk_dict: dict[str, Any] = {
                "id": chunk_id,
                "chunk_id": chunk_id,
                "content": original_text,
                "retrieval_content": retrieval_content,
                "context_header": context_header,
                "chunk_metadata": chunk_meta,
            }

            # If item was a parent-child chunk, preserve parent_id and parent_content
            if isinstance(item, dict) and "parent_content" in item:
                chunk_dict["parent_content"] = item["parent_content"]
                chunk_dict["parent_id"] = item.get("parent_id", "")

            enriched_chunks.append(chunk_dict)

        return enriched_chunks
