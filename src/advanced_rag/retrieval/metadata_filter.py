"""Metadata-based pre-filtering for chunks in Advanced RAG."""

from __future__ import annotations

from typing import Any


def filter_chunks(chunks: list[dict], filters: dict | None) -> list[dict]:
    """Filter chunk dictionaries based on metadata key-value criteria.

    If filters is None or empty, all chunks are returned preserving order.
    Otherwise, keeps only chunks where all filter key-value pairs match in
    chunk_metadata (or fallback metadata dict).

    Args:
        chunks: List of chunk dictionaries.
        filters: Dictionary of metadata key-value conditions to match.

    Returns:
        Filtered list of chunks preserving original order.
    """
    if filters is None or not filters:
        return list(chunks)

    filtered: list[dict] = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        # Check chunk_metadata first, fallback to metadata, fallback to chunk itself
        meta = chunk.get("chunk_metadata")
        if meta is None and "metadata" in chunk:
            meta = chunk.get("metadata")
        if meta is None or not isinstance(meta, dict):
            meta = chunk

        matches = True
        for key, expected_val in filters.items():
            actual_val = meta.get(key)

            if actual_val == expected_val:
                continue

            # Support list-membership matching (e.g. audience in ['student', 'all'])
            if isinstance(actual_val, (list, tuple, set)) and expected_val in actual_val:
                continue
            if isinstance(expected_val, (list, tuple, set)) and actual_val in expected_val:
                continue

            matches = False
            break

        if matches:
            filtered.append(chunk)

    return filtered


def filter_by_audience(chunks: list[dict], audience: str) -> list[dict]:
    """Convenience function for L3A requirement to filter chunks by audience.

    Pre-filters chunks to retain only those targeted at the specified audience
    (e.g., 'student', 'faculty', 'staff', or 'all').

    Args:
        chunks: List of chunk dictionaries.
        audience: Target audience string to filter for.

    Returns:
        Filtered list of chunk dictionaries.
    """
    if not audience:
        return list(chunks)

    return filter_chunks(chunks, {"audience": audience})
