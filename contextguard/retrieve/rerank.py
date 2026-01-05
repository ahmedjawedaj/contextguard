"""
Deduplication and reranking utilities.

Design:
- Stateless helpers to deduplicate chunks and rerank by score or custom key.
- Override-friendly: pass key functions for dedup and sorting.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

from ..core.specs import Chunk


def dedup_chunks(
    chunks: Iterable[Chunk],
    key_fn: Callable[[Chunk], str] = lambda c: c.provenance.source_id if c.provenance else "",
    keep_best_score: bool = True,
) -> List[Chunk]:
    """
    Deduplicate chunks by a key (default: source_id).
    If keep_best_score is True, keep the highest-score chunk per key.
    """
    best: Dict[str, Tuple[float, Chunk]] = {}
    for ch in chunks:
        key = key_fn(ch)
        score = ch.score or 0.0
        if key not in best or (keep_best_score and score > best[key][0]):
            best[key] = (score, ch)
    return [pair[1] for pair in best.values()]


def rerank_by_score(
    chunks: Iterable[Chunk],
    reverse: bool = True,
) -> List[Chunk]:
    """
    Simple rerank by chunk.score (None treated as 0).
    """
    return sorted(chunks, key=lambda c: c.score or 0.0, reverse=reverse)


def rerank_custom(
    chunks: Iterable[Chunk],
    sort_fn: Callable[[Chunk], float],
    reverse: bool = True,
) -> List[Chunk]:
    """Rerank by a custom numeric key."""
    return sorted(chunks, key=sort_fn, reverse=reverse)

