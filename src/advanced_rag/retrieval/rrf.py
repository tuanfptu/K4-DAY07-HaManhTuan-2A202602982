"""Reciprocal Rank Fusion (RRF) for combining multiple retrieval rankings."""

from __future__ import annotations

from typing import List, Tuple


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked retrieval lists using Reciprocal Rank Fusion (RRF).

    For each document d across all provided rankings, its fused score is:
        RRF_score(d) = sum(1.0 / (k + rank(d)))
    where rank(d) is the 1-based rank position of d in each ranking list.

    Args:
        rankings: List of ranked lists, where each list contains (chunk_index, score) tuples.
        k: Smoothing constant parameter (default is 60 as per standard RRF literature).

    Returns:
        Fused list of (chunk_index, rrf_score) tuples sorted in descending order of RRF score.
    """
    if not rankings:
        return []

    effective_k = k if k > 0 else 60
    rrf_scores: dict[int, float] = {}

    for ranking in rankings:
        for pos, item in enumerate(ranking):
            if not isinstance(item, (tuple, list)) or len(item) < 1:
                continue
            chunk_idx = int(item[0])
            rank = pos + 1  # 1-based rank position
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + (1.0 / (effective_k + rank))

    if not rrf_scores:
        return []

    # Sort documents by fused RRF score descending; break ties by chunk index ascending
    fused_ranking = sorted(
        rrf_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [(idx, float(score)) for idx, score in fused_ranking]
