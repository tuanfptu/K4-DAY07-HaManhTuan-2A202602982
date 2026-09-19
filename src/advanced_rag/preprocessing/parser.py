"""Markdown document and YAML frontmatter parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Graceful import of PyYAML
try:
    import yaml

    _HAS_YAML = True
except ImportError:
    yaml = None
    _HAS_YAML = False


def _fallback_parse_yaml(raw_text: str) -> dict[str, Any]:
    """Pure-Python fallback parser for basic YAML key-value pairs."""
    data: dict[str, Any] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Handle quotes
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            parsed_val: Any = val[1:-1]
        elif val.lower() == "true":
            parsed_val = True
        elif val.lower() == "false":
            parsed_val = False
        elif val.lower() in ("null", "none", "~", ""):
            parsed_val = None
        else:
            try:
                parsed_val = int(val)
            except ValueError:
                try:
                    parsed_val = float(val)
                except ValueError:
                    parsed_val = val

        data[key] = parsed_val
    return data


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from Markdown body text.

    Handles missing frontmatter, empty content, and YAML syntax issues gracefully.

    Args:
        text: Raw Markdown document text.

    Returns:
        A tuple of (metadata_dict, body_text).
    """
    if not text or not isinstance(text, str):
        return {}, ""

    # Strip UTF-8 BOM if present
    cleaned_text = text.lstrip("\ufeff")

    # Frontmatter regex: begins with --- on first line, ends with --- on its own line
    frontmatter_pattern = re.compile(
        r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)(.*)$",
        re.DOTALL,
    )
    match = frontmatter_pattern.match(cleaned_text)

    if not match:
        return {}, cleaned_text

    raw_frontmatter = match.group(1)
    body_text = match.group(2)

    metadata: dict[str, Any] = {}
    if _HAS_YAML and yaml is not None:
        try:
            parsed = yaml.safe_load(raw_frontmatter)
            if isinstance(parsed, dict):
                metadata = parsed
            elif parsed is None:
                metadata = {}
            else:
                metadata = {"_raw": parsed}
        except Exception:
            metadata = _fallback_parse_yaml(raw_frontmatter)
    else:
        metadata = _fallback_parse_yaml(raw_frontmatter)

    return metadata, body_text


def parse_markdown_file(filepath: Path | str) -> tuple[dict[str, Any], str]:
    """Read a Markdown file, parse its YAML frontmatter, and return (metadata, body).

    Args:
        filepath: Path to the Markdown file.

    Returns:
        Tuple of (metadata_dict, body_text).

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {filepath}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback with error replacement for resilient parsing
        raw_text = path.read_text(encoding="utf-8-sig", errors="replace")

    metadata, body = parse_frontmatter(raw_text)

    # If doc_id is not specified in frontmatter, assign from file stem
    if "doc_id" not in metadata:
        metadata["doc_id"] = path.stem

    return metadata, body


def load_corpus(corpus_dir: str | Path) -> list[tuple[dict[str, Any], str]]:
    """Load all Markdown (.md) documents from a directory.

    Args:
        corpus_dir: Directory containing .md files.

    Returns:
        List of (metadata_dict, body_text) tuples sorted deterministically by filename.
    """
    dir_path = Path(corpus_dir)
    if not dir_path.is_dir():
        return []

    # Deterministic alphabetical ordering by filename
    md_files = sorted(dir_path.glob("*.md"), key=lambda p: p.name.lower())

    corpus: list[tuple[dict[str, Any], str]] = []
    for file_path in md_files:
        try:
            meta, body = parse_markdown_file(file_path)
            corpus.append((meta, body))
        except Exception:
            continue

    return corpus
