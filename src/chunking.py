from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap

        chunks: list[str] = []

        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)

            if start + self.chunk_size >= len(text):
                break

        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection:
        - ". "
        - "! "
        - "? "
        - ".\\n"

    Extra whitespace is stripped from every sentence/chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(
            1,
            max_sentences_per_chunk,
        )

    def chunk(self, text: str) -> list[str]:
        # Empty input
        if not text or not text.strip():
            return []

        # Split AFTER sentence-ending punctuation.
        #
        # Example:
        # "Hello world. How are you? Fine!"
        #
        # becomes:
        # [
        #     "Hello world.",
        #     "How are you?",
        #     "Fine!"
        # ]
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        # Remove empty sentences and normalize whitespace
        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        chunks: list[str] = []

        # Group N sentences into one chunk
        for start in range(
            0,
            len(sentences),
            self.max_sentences_per_chunk,
        ):
            group = sentences[
                start : start + self.max_sentences_per_chunk
            ]

            chunk = " ".join(group).strip()

            if chunk:
                chunks.append(chunk)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\\n\\n", "\\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]

    def __init__(
        self,
        separators: list[str] | None = None,
        chunk_size: int = 500,
    ) -> None:
        self.separators = (
            self.DEFAULT_SEPARATORS
            if separators is None
            else list(separators)
        )

        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        text = text.strip()

        if len(text) <= self.chunk_size:
            return [text]

        if not self.separators:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
                if text[i : i + self.chunk_size].strip()
            ]

        return self._split(
            current_text=text,
            remaining_separators=list(self.separators),
        )

    def _split(
        self,
        current_text: str,
        remaining_separators: list[str],
    ) -> list[str]:
        current_text = current_text.strip()

        if not current_text:
            return []

        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
                if current_text[i : i + self.chunk_size].strip()
            ]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
                if current_text[i : i + self.chunk_size].strip()
            ]

        if separator not in current_text:
            return self._split(
                current_text=current_text,
                remaining_separators=next_separators,
            )

        parts = current_text.split(separator)
        chunks: list[str] = []
        buffer = ""

        for part in parts:
            part = part.strip()

            if not part:
                continue

            candidate = (
                part
                if not buffer
                else buffer + separator + part
            )

            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer.strip())
                buffer = ""

            if len(part) > self.chunk_size:
                chunks.extend(
                    self._split(
                        current_text=part,
                        remaining_separators=next_separators,
                    )
                )
            else:
                buffer = part

        if buffer:
            chunks.append(buffer.strip())

        return chunks


def _dot(
    a: list[float],
    b: list[float],
) -> float:
    return sum(
        x * y
        for x, y in zip(a, b)
    )


def compute_similarity(
    vec_a: list[float],
    vec_b: list[float],
) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity =
        dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """

    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class HeadingAwareRecursiveChunker:
    """
    Structure-aware chunker optimized for Vietnamese policy /
    university-service Markdown documents.

    Strategy:
        1. Detect Markdown headings (#, ##, ###, ...).
        2. Detect Vietnamese policy boundaries:
           - CHƯƠNG ...
           - Mục ...
           - Điều ...
        3. Preserve the heading hierarchy as context.
        4. If a section is too large, recursively split only its body.
        5. Prepend the section path to every resulting chunk.

    Example output:

        [SECTION]
        QUY CHẾ ĐÀO TẠO > CHƯƠNG III > Điều 12. Đăng ký học phần

        <content>
    """

    MARKDOWN_HEADING_RE = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$"
    )

    CHAPTER_RE = re.compile(
        r"^\s*(?:\*\*)?\s*"
        r"(CHƯƠNG\s+[IVXLCDM0-9]+.*)"
        r"\s*(?:\*\*)?\s*$",
        re.IGNORECASE,
    )

    SECTION_RE = re.compile(
        r"^\s*(?:\*\*)?\s*"
        r"(Mục\s+\d+.*)"
        r"\s*(?:\*\*)?\s*$",
        re.IGNORECASE,
    )

    ARTICLE_RE = re.compile(
        r"^\s*(?:\*\*)?\s*"
        r"(Điều\s+\d+[a-zA-Z]?[\.\s:].*)"
        r"\s*(?:\*\*)?\s*$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        chunk_size: int = 1000,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = max(200, chunk_size)

        self.separators = (
            ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
            if separators is None
            else list(separators)
        )

    @staticmethod
    def _clean_heading(text: str) -> str:
        """
        Remove lightweight Markdown formatting from headings.
        """

        text = text.strip()

        # Remove Markdown bold/italic markers
        text = text.replace("**", "")
        text = text.replace("__", "")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def _build_path(headings: dict[int, str]) -> list[str]:
        """
        Build ordered hierarchy from heading levels.
        """

        result: list[str] = []

        for level in sorted(headings):
            heading = headings[level].strip()

            if not heading:
                continue

            # Avoid exact duplicate headings
            if result and result[-1].casefold() == heading.casefold():
                continue

            result.append(heading)

        return result

    @staticmethod
    def _make_prefix(path: list[str]) -> str:
        """
        Convert heading hierarchy into retrieval context.
        """

        if not path:
            return ""

        return "[SECTION]\n" + " > ".join(path)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections = self._extract_sections(text)

        chunks: list[str] = []

        for path, body in sections:
            body = body.strip()

            if not body:
                continue

            prefix = self._make_prefix(path)

            if prefix:
                complete = f"{prefix}\n\n{body}"
            else:
                complete = body

            # Section already fits
            if len(complete) <= self.chunk_size:
                chunks.append(complete)
                continue

            # -----------------------------------------
            # Oversized section
            # → recursive split only the BODY
            # -----------------------------------------

            prefix_cost = len(prefix) + 2 if prefix else 0

            available_size = max(
                200,
                self.chunk_size - prefix_cost,
            )

            recursive = RecursiveChunker(
                separators=self.separators,
                chunk_size=available_size,
            )

            sub_chunks = recursive.chunk(body)

            for sub_chunk in sub_chunks:
                sub_chunk = sub_chunk.strip()

                if not sub_chunk:
                    continue

                if prefix:
                    final_chunk = (
                        f"{prefix}\n\n{sub_chunk}"
                    )
                else:
                    final_chunk = sub_chunk

                chunks.append(final_chunk)

        return chunks

    def _extract_sections(
        self,
        text: str,
    ) -> list[tuple[list[str], str]]:
        """
        Parse Markdown/policy structure into:

            [
                (heading_path, section_body),
                ...
            ]
        """

        lines = text.splitlines()

        headings: dict[int, str] = {}

        sections: list[tuple[list[str], str]] = []

        current_path: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer

            body = "\n".join(buffer).strip()

            if body:
                sections.append(
                    (
                        list(current_path),
                        body,
                    )
                )

            buffer = []

        for raw_line in lines:
            line = raw_line.strip()

            # -----------------------------------------
            # Markdown heading
            # # / ## / ### / #### ...
            # -----------------------------------------

            markdown_match = self.MARKDOWN_HEADING_RE.match(
                line
            )

            if markdown_match:
                flush()

                level = len(
                    markdown_match.group(1)
                )

                heading_text = self._clean_heading(
                    markdown_match.group(2)
                )

                # Remove this level and all children
                for existing_level in list(headings):
                    if existing_level >= level:
                        del headings[existing_level]

                headings[level] = heading_text

                current_path = self._build_path(
                    headings
                )

                continue

            # -----------------------------------------
            # CHƯƠNG ...
            # -----------------------------------------

            chapter_match = self.CHAPTER_RE.match(line)

            if chapter_match:
                flush()

                # Synthetic hierarchy level
                for existing_level in list(headings):
                    if existing_level >= 20:
                        del headings[existing_level]

                headings[20] = self._clean_heading(
                    chapter_match.group(1)
                )

                current_path = self._build_path(
                    headings
                )

                continue

            # -----------------------------------------
            # Mục ...
            # -----------------------------------------

            section_match = self.SECTION_RE.match(line)

            if section_match:
                flush()

                for existing_level in list(headings):
                    if existing_level >= 30:
                        del headings[existing_level]

                headings[30] = self._clean_heading(
                    section_match.group(1)
                )

                current_path = self._build_path(
                    headings
                )

                continue

            # -----------------------------------------
            # Điều ...
            # -----------------------------------------

            article_match = self.ARTICLE_RE.match(line)

            if article_match:
                flush()

                # New Điều replaces previous Điều
                headings.pop(40, None)

                headings[40] = self._clean_heading(
                    article_match.group(1)
                )

                current_path = self._build_path(
                    headings
                )

                continue

            buffer.append(raw_line)

        flush()

        return sections

class ChunkingStrategyComparator:
    """
    Run all built-in chunking strategies
    and compare their results.
    """

    @staticmethod
    def _stats(chunks: list[str]) -> dict:
        count = len(chunks)
        avg_length = (
            sum(len(chunk) for chunk in chunks) / count
            if count
            else 0.0
        )

        return {
            "count": count,
            "avg_length": avg_length,
            "chunks": chunks,
        }

    def compare(
        self,
        text: str,
        chunk_size: int = 200,
    ) -> dict:
        fixed_chunks = FixedSizeChunker(
            chunk_size=chunk_size,
            overlap=0,
        ).chunk(text)

        sentence_chunks = SentenceChunker(
            max_sentences_per_chunk=3,
        ).chunk(text)

        recursive_chunks = RecursiveChunker(
            chunk_size=chunk_size,
        ).chunk(text)

        return {
            "fixed_size": self._stats(fixed_chunks),
            "by_sentences": self._stats(sentence_chunks),
            "recursive": self._stats(recursive_chunks),
        }
