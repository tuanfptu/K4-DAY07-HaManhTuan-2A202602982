"""Corpus and document quality validation for Advanced RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parser import parse_frontmatter

# Standard required metadata fields in corpus documents
REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "title",
    "source_url",
    "retrieved_at",
    "document_version",
    "audience",
)

# Standard recognized audience values
DEFAULT_VALID_AUDIENCES: set[str] = {
    "student",
    "faculty",
    "staff",
    "prospective_student",
    "alumni",
    "all",
    "public",
    "general",
    "parent",
    "enterprise",
}

# Minimum character length threshold for non-trivial policy documents
MIN_BODY_CHARACTERS: int = 100


def validate_document(
    filepath: Path | str,
    valid_audiences: set[str] | None = None,
) -> list[str]:
    """Validate a single document file for metadata completeness, encoding, and content quality.

    Checks performed:
        - File existence and UTF-8 encoding validity (no decode errors or replacement chars).
        - Presence of all required metadata fields (title, source_url, retrieved_at, document_version, audience).
        - Valid audience value.
        - Non-empty document body.
        - Non-suspicious body length (at least 100 characters).

    Args:
        filepath: Path to the Markdown document.
        valid_audiences: Optional custom set of valid audience strings.

    Returns:
        List of issue description strings. Empty list if document is completely valid.
    """
    path = Path(filepath)
    issues: list[str] = []

    if not path.is_file():
        return [f"File not found or not a regular file: {filepath}"]

    # Check UTF-8 encoding
    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"Encoding error: File is not valid UTF-8 ({exc})"]

    if "\ufffd" in text:
        issues.append("Encoding issue: File contains Unicode replacement character (\\ufffd)")

    # Parse YAML frontmatter
    metadata, body = parse_frontmatter(text)

    # Validate required metadata fields
    for req_field in REQUIRED_METADATA_FIELDS:
        if req_field not in metadata or metadata[req_field] is None:
            issues.append(f"Missing required metadata field: '{req_field}'")
        elif str(metadata[req_field]).strip() == "":
            issues.append(f"Empty required metadata field: '{req_field}'")

    # Validate audience
    if "audience" in metadata and metadata["audience"] is not None:
        aud = str(metadata["audience"]).strip().lower()
        allowed = valid_audiences if valid_audiences is not None else DEFAULT_VALID_AUDIENCES
        if aud and aud not in allowed:
            issues.append(
                f"Invalid audience value: '{metadata['audience']}' (expected one of: {sorted(allowed)})"
            )

    # Validate body content
    stripped_body = body.strip()
    if not stripped_body:
        issues.append("Empty content: Document body contains no text")
    elif len(stripped_body) < MIN_BODY_CHARACTERS:
        issues.append(
            f"Suspiciously short document: Body has {len(stripped_body)} characters (minimum: {MIN_BODY_CHARACTERS})"
        )

    return issues


def validate_corpus(
    corpus_dir: Path | str,
    valid_audiences: set[str] | None = None,
) -> dict[str, list[str]]:
    """Validate all Markdown (.md) documents within a corpus directory.

    Args:
        corpus_dir: Directory containing Markdown documents.
        valid_audiences: Optional custom set of valid audience strings.

    Returns:
        Dictionary mapping filename to list of discovered issues: {filename: [issues]}.
    """
    dir_path = Path(corpus_dir)
    if not dir_path.is_dir():
        return {str(corpus_dir): [f"Corpus directory not found: {corpus_dir}"]}

    md_files = sorted(dir_path.glob("*.md"), key=lambda p: p.name.lower())
    if not md_files:
        return {"": [f"No Markdown (.md) documents found in corpus directory: {corpus_dir}"]}

    results: dict[str, list[str]] = {}
    for filepath in md_files:
        results[filepath.name] = validate_document(
            filepath,
            valid_audiences=valid_audiences,
        )

    return results
