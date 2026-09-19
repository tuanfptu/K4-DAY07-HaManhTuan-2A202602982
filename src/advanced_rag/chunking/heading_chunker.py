"""
Heading-aware chunker for Vietnamese university policy and administrative documents.
"""

from __future__ import annotations

import re
from typing import Any

from .utils import (
    build_context_prefix,
    clean_heading,
    parse_frontmatter,
    parse_number_or_roman,
    recursive_split_text,
    slugify,
)


class HeadingChunker:
    """
    Structure-aware chunker optimized for Vietnamese university policy documents.

    Features:
        - Detects document hierarchy: # Title, ## CHƯƠNG, ### Điều, #### Khoản, markdown headings.
        - Produces retrieval_content with structured context prefix:
            [SECTION]
            Document: {title} > CHƯƠNG II > Điều 6

            {content}
        - Keeps original text intact in 'content'.
        - Recursively splits oversized section bodies while preserving heading context.
        - Generates structured chunk_id: {doc_id}__chapter_{c}__article_{a}__part_{p}.
        - Populates chunk_metadata with chapter, article, section path, and document metadata.
    """

    # Chapter regex: "## CHƯƠNG I: ...", "#### **CHƯƠNG I**", "CHƯƠNG II", "Chương 1"
    CHAPTER_RE = re.compile(
        r"^\s*(?:#{1,6}\s+)?(?:\*\*)?\s*"
        r"(CHƯƠNG\s+([IVXLCDM0-9]+)(?:[:\.\s\-]+(.*))?)"
        r"\s*(?:\*\*)?\s*$",
        re.IGNORECASE,
    )

    # Section / Mục regex: "### Mục 1. ...", "Mục 2: ...", "**Mục 1**"
    SECTION_RE = re.compile(
        r"^\s*(?:#{1,6}\s+)?(?:\*\*)?\s*"
        r"(Mục\s+(\d+[a-zA-Z]?)(?:[:\.\s\-]+(.*))?)"
        r"\s*(?:\*\*)?\s*$",
        re.IGNORECASE,
    )

    # Article / Điều regex: "### Điều 6. ...", "**Điều 1. Phạm vi...**", "Điều 6"
    ARTICLE_RE = re.compile(
        r"^\s*(?:#{1,6}\s+)?(?:\*\*)?\s*"
        r"(Điều\s+(\d+[a-zA-Z]?)(?:[:\.\s\-]+(.*))?)"
        r"\s*(?:\*\*)?\s*$",
        re.IGNORECASE,
    )

    # Clause / Khoản regex: "#### Khoản 1. ...", "Khoản 1:", "**Khoản 2**"
    CLAUSE_RE = re.compile(
        r"^\s*(?:#{1,6}\s+)?(?:\*\*)?\s*"
        r"(Khoản\s+(\d+[a-zA-Z]?)(?:[:\.\s\-]+(.*))?)"
        r"\s*(?:\*\*)?\s*$",
        re.IGNORECASE,
    )

    # Standard Markdown headings: "# Title", "## Subsection"
    MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def __init__(
        self,
        chunk_size: int = 1000,
        include_context_prefix: bool = True,
        min_chunk_size: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        """
        Initialize the HeadingChunker.

        Args:
            chunk_size: Target maximum chunk size in characters.
            include_context_prefix: Whether to prepend [SECTION] context prefix to retrieval_content.
            min_chunk_size: Minimum character size for recursive sub-chunks.
            separators: Separators for recursive splitting (default: paragraphs, newlines, sentences).
        """
        self.chunk_size = max(min_chunk_size, chunk_size)
        self.include_context_prefix = include_context_prefix
        self.min_chunk_size = min_chunk_size
        self.separators = separators

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Chunk a Vietnamese policy document into structured chunks.

        Args:
            text: Markdown or plain text of the document.
            metadata: Optional dictionary with document metadata (title, doc_id/id, audience, etc.).

        Returns:
            List of chunk dictionaries with 'id', 'chunk_id', 'content', 'retrieval_content',
            and 'chunk_metadata'.
        """
        if not text or not text.strip():
            return []

        doc_meta: dict[str, Any] = dict(metadata or {})

        # Handle YAML frontmatter if present in raw text
        front_meta, clean_text = parse_frontmatter(text)
        for k, v in front_meta.items():
            if k not in doc_meta:
                doc_meta[k] = v

        clean_text = clean_text.strip()
        if not clean_text:
            return []

        doc_id = str(doc_meta.get("doc_id") or doc_meta.get("id") or "doc").strip()
        doc_title = str(doc_meta.get("title") or "").strip()

        # Parse the document into hierarchical sections
        sections = self._extract_sections(clean_text, default_title=doc_title)

        # Update doc_title if found during section extraction
        if not doc_title and sections:
            first_title = sections[0].get("detected_title")
            if first_title:
                doc_title = first_title
                doc_meta["title"] = doc_title

        if not doc_title:
            doc_title = doc_id

        chunks: list[dict[str, Any]] = []

        for sec in sections:
            body = sec["body"].strip()
            if not body:
                continue

            section_path = sec["section_path"]
            chapter = sec["chapter"]
            chapter_num = sec["chapter_num"]
            article = sec["article"]
            article_num = sec["article_num"]
            clause = sec["clause"]
            clause_num = sec["clause_num"]
            section_slug = sec["section_slug"]

            # Construct the context prefix for retrieval
            prefix = build_context_prefix(doc_title, section_path) if self.include_context_prefix else ""
            prefix_cost = len(prefix) + 2 if prefix else 0

            available_size = max(
                self.min_chunk_size,
                self.chunk_size - prefix_cost if self.include_context_prefix else self.chunk_size,
            )

            # Determine if body needs recursive splitting
            if len(body) <= available_size:
                sub_bodies = [body]
            else:
                sub_bodies = recursive_split_text(
                    text=body,
                    max_chunk_size=available_size,
                    separators=self.separators,
                )

            total_parts = len(sub_bodies)

            for part_idx, sub_body in enumerate(sub_bodies, start=1):
                sub_body = sub_body.strip()
                if not sub_body:
                    continue

                # Build retrieval content with context prefix
                if self.include_context_prefix and prefix:
                    retrieval_content = f"{prefix}\n\n{sub_body}"
                else:
                    retrieval_content = sub_body

                # Generate structured chunk ID
                chunk_id = self._build_chunk_id(
                    doc_id=doc_id,
                    chapter_num=chapter_num,
                    article_num=article_num,
                    clause_num=clause_num,
                    section_slug=section_slug,
                    part_index=part_idx,
                )

                # Assemble chunk metadata
                chunk_meta = dict(doc_meta)
                chunk_meta.update(
                    {
                        "doc_id": doc_id,
                        "title": doc_title,
                        "chapter": chapter,
                        "chapter_number": int(chapter_num) if (chapter_num and chapter_num.isdigit()) else chapter_num,
                        "article": article,
                        "article_number": int(article_num) if (article_num and article_num.isdigit()) else article_num,
                        "clause": clause,
                        "clause_number": int(clause_num) if (clause_num and clause_num.isdigit()) else clause_num,
                        "section_path": list(section_path),
                        "part_index": part_idx,
                        "total_parts": total_parts,
                    }
                )

                chunks.append(
                    {
                        "id": chunk_id,
                        "chunk_id": chunk_id,
                        "content": sub_body,
                        "retrieval_content": retrieval_content,
                        "chunk_metadata": chunk_meta,
                    }
                )

        return chunks

    @staticmethod
    def _build_chunk_id(
        doc_id: str,
        chapter_num: str | None,
        article_num: str | None,
        clause_num: str | None,
        section_slug: str | None,
        part_index: int,
    ) -> str:
        """
        Generate structured chunk_id like:
            {doc_id}__chapter_2__article_6__part_1
        """
        clean_doc = re.sub(r"[^a-zA-Z0-9_\-]+", "_", doc_id).strip("_") or "doc"
        parts = [clean_doc]

        if chapter_num:
            parts.append(f"chapter_{chapter_num}")

        if article_num:
            parts.append(f"article_{article_num}")
        if clause_num:
            parts.append(f"clause_{clause_num}")
        elif not article_num and not chapter_num and section_slug:
            parts.append(f"sec_{section_slug}")

        parts.append(f"part_{part_index}")
        return "__".join(parts)

    def _extract_sections(
        self,
        text: str,
        default_title: str = "",
    ) -> list[dict[str, Any]]:
        """
        Scan text line-by-line and extract structural sections with hierarchy.
        """
        lines = text.splitlines()

        detected_title = default_title
        headings_map: dict[int, str] = {}

        current_chapter: str | None = None
        current_chapter_num: str | None = None
        current_section: str | None = None
        current_article: str | None = None
        current_article_num: str | None = None
        current_clause: str | None = None
        current_clause_num: str | None = None
        current_slug: str | None = None

        sections: list[dict[str, Any]] = []
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            body = "\n".join(buffer).strip()
            buffer = []
            if not body:
                return

            # Build ordered list of active headings
            path: list[str] = []
            for lvl in sorted(headings_map):
                h = headings_map[lvl].strip()
                if h and (not path or path[-1].casefold() != h.casefold()):
                    path.append(h)

            sections.append(
                {
                    "body": body,
                    "section_path": path,
                    "chapter": current_chapter,
                    "chapter_num": current_chapter_num,
                    "section": current_section,
                    "article": current_article,
                    "article_num": current_article_num,
                    "clause": current_clause,
                    "clause_num": current_clause_num,
                    "section_slug": current_slug,
                    "detected_title": detected_title,
                }
            )

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                buffer.append(raw_line)
                continue

            # ----------------------------------------------------
            # 1. Check Chapter (CHƯƠNG)
            # ----------------------------------------------------
            chapter_match = self.CHAPTER_RE.match(line)
            if chapter_match:
                flush()
                raw_heading = chapter_match.group(1)
                cleaned = clean_heading(raw_heading)
                ch_num = parse_number_or_roman(chapter_match.group(2))

                # Clear levels >= 2
                for lvl in list(headings_map):
                    if lvl >= 2:
                        del headings_map[lvl]

                headings_map[2] = cleaned
                current_chapter = cleaned
                current_chapter_num = ch_num
                current_section = None
                current_article = None
                current_article_num = None
                current_clause = None
                current_clause_num = None
                current_slug = None
                continue

            # ----------------------------------------------------
            # 2. Check Section (Mục)
            # ----------------------------------------------------
            section_match = self.SECTION_RE.match(line)
            if section_match:
                flush()
                raw_heading = section_match.group(1)
                cleaned = clean_heading(raw_heading)

                # Clear levels >= 3
                for lvl in list(headings_map):
                    if lvl >= 3:
                        del headings_map[lvl]

                headings_map[3] = cleaned
                current_section = cleaned
                current_article = None
                current_article_num = None
                current_clause = None
                current_clause_num = None
                current_slug = None
                continue

            # ----------------------------------------------------
            # 3. Check Article (Điều)
            # ----------------------------------------------------
            article_match = self.ARTICLE_RE.match(line)
            if article_match:
                flush()
                raw_heading = article_match.group(1)
                cleaned = clean_heading(raw_heading)
                art_num = parse_number_or_roman(article_match.group(2))

                # Clear levels >= 4
                for lvl in list(headings_map):
                    if lvl >= 4:
                        del headings_map[lvl]

                headings_map[4] = cleaned
                current_article = cleaned
                current_article_num = art_num
                current_clause = None
                current_clause_num = None
                current_slug = None
                continue

            # ----------------------------------------------------
            # 4. Check Clause (Khoản)
            # ----------------------------------------------------
            clause_match = self.CLAUSE_RE.match(line)
            if clause_match:
                flush()
                raw_heading = clause_match.group(1)
                cleaned = clean_heading(raw_heading)
                cl_num = parse_number_or_roman(clause_match.group(2))

                # Clear levels >= 5
                for lvl in list(headings_map):
                    if lvl >= 5:
                        del headings_map[lvl]

                headings_map[5] = cleaned
                current_clause = cleaned
                current_clause_num = cl_num
                current_slug = None
                continue

            # ----------------------------------------------------
            # 5. Check Markdown Headings (#, ##, ###, ...)
            # ----------------------------------------------------
            md_match = self.MARKDOWN_HEADING_RE.match(line)
            if md_match:
                flush()
                level = len(md_match.group(1))
                heading_text = clean_heading(md_match.group(2))

                if level == 1:
                    # Document Level 1 Title
                    if not detected_title:
                        detected_title = heading_text
                    headings_map.clear()
                    headings_map[1] = heading_text
                    current_chapter = None
                    current_chapter_num = None
                    current_section = None
                    current_article = None
                    current_article_num = None
                    current_clause = None
                    current_clause_num = None
                    current_slug = None
                else:
                    # Clear this level and lower
                    for lvl in list(headings_map):
                        if lvl >= level:
                            del headings_map[lvl]

                    headings_map[level] = heading_text

                    if level <= 2:
                        current_chapter = None
                        current_chapter_num = None
                        current_section = None
                        current_article = None
                        current_article_num = None
                        current_clause = None
                        current_clause_num = None
                    elif level <= 3:
                        current_section = None
                        current_article = None
                        current_article_num = None
                        current_clause = None
                        current_clause_num = None
                    elif level <= 4:
                        current_article = None
                        current_article_num = None
                        current_clause = None
                        current_clause_num = None

                    current_slug = slugify(heading_text, max_length=25)

                continue

            buffer.append(raw_line)

        flush()
        return sections
