"""
Utility helpers for advanced chunking of Vietnamese university policy documents.
"""

from __future__ import annotations

import re
from typing import Any

# Mapping for Vietnamese characters with diacritics to ASCII base equivalents
VIETNAMESE_ACCENT_MAP: dict[str, str] = {
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "đ": "d",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
}

ROMAN_NUMERALS: dict[str, int] = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def remove_vietnamese_accents(text: str) -> str:
    """
    Replace Vietnamese accented characters with their ASCII counterparts.

    Args:
        text: Input string potentially containing Vietnamese diacritics.

    Returns:
        String with accented characters replaced by unaccented letters.
    """
    res: list[str] = []
    for ch in text.lower():
        res.append(VIETNAMESE_ACCENT_MAP.get(ch, ch))
    return "".join(res)


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert arbitrary text into a clean snake_case identifier.

    Args:
        text: Input text string.
        max_length: Maximum character length of the generated slug.

    Returns:
        Alphanumeric snake_case slug.
    """
    unaccented = remove_vietnamese_accents(text)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", unaccented).lower().strip("_")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug or "sec"


def roman_to_int(roman_str: str) -> int | None:
    """
    Convert a Roman numeral string (e.g. 'I', 'II', 'VI', 'XII') to an integer.

    Args:
        roman_str: Roman numeral string.

    Returns:
        Integer value or None if the input is not a valid Roman numeral.
    """
    if not roman_str:
        return None

    cleaned = roman_str.strip().upper()
    if not cleaned or not all(c in ROMAN_NUMERALS for c in cleaned):
        return None

    total = 0
    prev_val = 0
    for char in reversed(cleaned):
        val = ROMAN_NUMERALS[char]
        if val < prev_val:
            total -= val
        else:
            total += val
            prev_val = val

    return total


def parse_number_or_roman(val_str: str) -> str:
    """
    Normalize a chapter/article/clause number into a string integer or clean label.

    Handles:
        - Arabic numerals ('1', '6') -> '1', '6'
        - Roman numerals ('II', 'IV') -> '2', '4'
        - Alphanumeric codes ('12a') -> '12a'
        - General text fallback -> slugified string

    Args:
        val_str: Extracted number/numeral string.

    Returns:
        Normalized string identifier.
    """
    cleaned = val_str.strip()
    if cleaned.isdigit():
        return cleaned

    roman_val = roman_to_int(cleaned)
    if roman_val is not None:
        return str(roman_val)

    match = re.match(r"^(\d+[a-zA-Z]?)", cleaned)
    if match:
        return match.group(1).lower()

    return slugify(cleaned, max_length=20)


def clean_heading(text: str) -> str:
    """
    Remove lightweight Markdown bold, italic, link syntax, and collapse whitespaces.

    Args:
        text: Raw heading line from document.

    Returns:
        Clean, human-readable heading string.
    """
    text = text.strip()
    # Strip leading markdown heading hashes
    text = re.sub(r"^#+\s*", "", text)
    # Strip markdown link format: [anchor](url) -> anchor
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Strip bold/italic markers
    text = text.replace("**", "").replace("__", "")
    text = text.replace("*", "").replace("_", "")
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    Extract YAML frontmatter from Markdown text if present.

    Args:
        text: Raw markdown text.

    Returns:
        Tuple of (metadata_dict, body_content_without_frontmatter).
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) >= 3:
        raw_yaml = parts[1].strip()
        body = parts[2].strip()
        metadata: dict[str, Any] = {}
        try:
            import yaml
            parsed = yaml.safe_load(raw_yaml)
            if isinstance(parsed, dict):
                metadata = parsed
        except Exception:
            # Fallback line-by-line parser if PyYAML is unavailable or malformed
            for line in raw_yaml.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key:
                        metadata[key] = val
        return metadata, body

    return {}, text


def build_context_prefix(title: str, section_path: list[str]) -> str:
    """
    Build a structured context prefix for retrieval content.

    Format:
        [SECTION]
        Document: {title} > CHƯƠNG II > Điều 6

    Args:
        title: Document title.
        section_path: List of hierarchical heading strings.

    Returns:
        Context prefix string starting with '[SECTION]\n...'.
    """
    path_items = [p for p in section_path if p.strip()]

    # Avoid duplicating document title at the beginning of path
    if path_items and title and path_items[0].casefold() == title.casefold():
        path_items = path_items[1:]

    if title and path_items:
        hierarchy = f"Document: {title} > " + " > ".join(path_items)
    elif title:
        hierarchy = f"Document: {title}"
    elif path_items:
        hierarchy = " > ".join(path_items)
    else:
        hierarchy = "Document"

    return f"[SECTION]\n{hierarchy}"


def recursive_split_text(
    text: str,
    max_chunk_size: int,
    separators: list[str] | None = None,
    overlap: int = 0,
) -> list[str]:
    """
    Recursively split text using priority separators until all pieces fit max_chunk_size.

    Args:
        text: Input text to split.
        max_chunk_size: Maximum allowed character length per chunk.
        separators: Priority list of string separators.
        overlap: Character overlap between consecutive chunks if applicable.

    Returns:
        List of trimmed text chunks.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [text]

    default_separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    seps = default_separators if separators is None else list(separators)

    return _split_recursive_helper(
        current_text=text,
        remaining_separators=seps,
        max_chunk_size=max_chunk_size,
        overlap=overlap,
    )


def _split_recursive_helper(
    current_text: str,
    remaining_separators: list[str],
    max_chunk_size: int,
    overlap: int,
) -> list[str]:
    """Internal recursive split helper."""
    current_text = current_text.strip()
    if not current_text:
        return []

    if len(current_text) <= max_chunk_size:
        return [current_text]

    if not remaining_separators:
        # Fallback to hard character slicing
        step = max_chunk_size - overlap if max_chunk_size > overlap else max_chunk_size
        chunks: list[str] = []
        for start in range(0, len(current_text), step):
            piece = current_text[start : start + max_chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + max_chunk_size >= len(current_text):
                break
        return chunks

    separator = remaining_separators[0]
    next_separators = remaining_separators[1:]

    if separator == "":
        step = max_chunk_size - overlap if max_chunk_size > overlap else max_chunk_size
        chunks = []
        for start in range(0, len(current_text), step):
            piece = current_text[start : start + max_chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + max_chunk_size >= len(current_text):
                break
        return chunks

    if separator not in current_text:
        return _split_recursive_helper(
            current_text=current_text,
            remaining_separators=next_separators,
            max_chunk_size=max_chunk_size,
            overlap=overlap,
        )

    parts = current_text.split(separator)
    chunks: list[str] = []
    buffer = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        candidate = part if not buffer else buffer + separator + part

        if len(candidate) <= max_chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer.strip())
            buffer = ""

        if len(part) > max_chunk_size:
            sub_splits = _split_recursive_helper(
                current_text=part,
                remaining_separators=next_separators,
                max_chunk_size=max_chunk_size,
                overlap=overlap,
            )
            chunks.extend(sub_splits)
        else:
            buffer = part

    if buffer:
        chunks.append(buffer.strip())

    return chunks
