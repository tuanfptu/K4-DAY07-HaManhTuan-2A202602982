"""
Parent-Child chunker for Vietnamese university policy documents.
"""

from __future__ import annotations

from typing import Any

from .heading_chunker import HeadingChunker
from .utils import build_context_prefix, recursive_split_text


class ParentChildChunker:
    """
    Parent-Child chunking strategy.

    Splits documents into larger parent chunks (aligned with heading boundaries such as
    Articles or Chapters) and smaller child chunks.

    Retrieval uses the smaller, more semantically dense child chunks (or their retrieval_content),
    while LLM generation has immediate access to the full parent_content for complete context.

    Each chunk dict contains:
        - 'id': Child chunk ID (e.g. '{parent_id}__child_1')
        - 'chunk_id': Same as 'id'
        - 'parent_id': Parent chunk ID
        - 'content': Child chunk text
        - 'parent_content': Full text of the parent section
        - 'retrieval_content': Child chunk enriched with section context prefix
        - 'chunk_metadata': Metadata including hierarchy path, parent_id, and child_index
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        child_overlap: int = 50,
        include_context_prefix: bool = True,
    ) -> None:
        """
        Initialize the ParentChildChunker.

        Args:
            parent_chunk_size: Target size for parent chunks in characters.
            child_chunk_size: Target size for child chunks in characters.
            child_overlap: Overlap in characters between child chunks.
            include_context_prefix: Whether to prepend context prefix to retrieval_content.
        """
        self.parent_chunk_size = max(child_chunk_size, parent_chunk_size)
        self.child_chunk_size = max(100, child_chunk_size)
        self.child_overlap = min(child_overlap, self.child_chunk_size // 2)
        self.include_context_prefix = include_context_prefix

        # Base heading chunker used to extract parent units along heading boundaries
        self._parent_chunker = HeadingChunker(
            chunk_size=self.parent_chunk_size,
            include_context_prefix=False,
            min_chunk_size=min(300, self.parent_chunk_size),
        )

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Split document into parent-child chunks.

        Args:
            text: Raw document text (Markdown or plain text).
            metadata: Document metadata dictionary.

        Returns:
            List of child chunk dictionaries with 'content', 'parent_content',
            'retrieval_content', and 'chunk_metadata'.
        """
        if not text or not text.strip():
            return []

        doc_metadata = dict(metadata or {})

        # Step 1: Use heading boundaries to segment text into parent chunks
        parent_chunks = self._parent_chunker.chunk(text=text, metadata=doc_metadata)

        if not parent_chunks:
            return []

        results: list[dict[str, Any]] = []

        # Step 2: Split each parent into smaller child chunks
        for parent in parent_chunks:
            parent_id = parent["id"]
            parent_content = parent["content"].strip()
            parent_meta = parent["chunk_metadata"]

            if not parent_content:
                continue

            # Split parent body into child chunks
            if len(parent_content) <= self.child_chunk_size:
                child_bodies = [parent_content]
            else:
                child_bodies = recursive_split_text(
                    text=parent_content,
                    max_chunk_size=self.child_chunk_size,
                    overlap=self.child_overlap,
                )

            total_children = len(child_bodies)
            doc_title = parent_meta.get("title", "")
            section_path = parent_meta.get("section_path", [])
            prefix = build_context_prefix(doc_title, section_path) if self.include_context_prefix else ""

            for child_idx, child_body in enumerate(child_bodies, start=1):
                child_body = child_body.strip()
                if not child_body:
                    continue

                child_id = f"{parent_id}__child_{child_idx}"

                if self.include_context_prefix and prefix:
                    retrieval_content = f"{prefix}\n\n{child_body}"
                else:
                    retrieval_content = child_body

                child_meta = dict(parent_meta)
                child_meta.update(
                    {
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "child_index": child_idx,
                        "total_children": total_children,
                        "parent_chunk_size": self.parent_chunk_size,
                        "child_chunk_size": self.child_chunk_size,
                    }
                )

                results.append(
                    {
                        "id": child_id,
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "content": child_body,
                        "parent_content": parent_content,
                        "retrieval_content": retrieval_content,
                        "chunk_metadata": child_meta,
                    }
                )

        return results

    @staticmethod
    def extract_parents(chunks: list[dict[str, Any]]) -> dict[str, str]:
        """
        Extract unique parent chunks mapping {parent_id: parent_content}.

        Useful for expanding retrieved child chunks to their full parent context
        for LLM generation without duplicates.

        Args:
            chunks: List of child chunk dicts.

        Returns:
            Dictionary mapping parent_id to parent_content string.
        """
        parents: dict[str, str] = {}
        for c in chunks:
            pid = c.get("parent_id")
            p_content = c.get("parent_content")
            if pid and p_content and pid not in parents:
                parents[pid] = p_content
        return parents

    @staticmethod
    def get_parent(chunk: dict[str, Any]) -> str:
        """
        Retrieve the full parent content of a given chunk dict.

        Args:
            chunk: Child chunk dictionary.

        Returns:
            Parent content string.
        """
        return chunk.get("parent_content", "") or chunk.get("content", "")
