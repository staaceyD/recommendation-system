"""Top-K ranking metrics with binary relevance.

Every function takes a ranked list of movie ids (best first) and the set of ids
that were actually relevant for that user, and scores only the first `k`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the top-k that the user really liked."""
    if k <= 0:
        return 0.0
    return _hits(ranked, relevant, k) / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the user's liked movies that made it into the top-k."""
    if not relevant:
        return 0.0
    return _hits(ranked, relevant, k) / len(relevant)


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """1.0 if the top-k contains at least one liked movie -- "was the list useful at all"."""
    return 1.0 if _hits(ranked, relevant, k) else 0.0


def average_precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Precision averaged over the positions that hit -- rewards hits near the top."""
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    total = 0.0
    for position, movie_id in enumerate(ranked[:k], start=1):
        if movie_id in relevant:
            hits += 1
            total += hits / position
    return total / min(k, len(relevant))


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Discounted gain over the ideal ordering: 1.0 means every hit sits as high as it could."""
    if not relevant or k <= 0:
        return 0.0
    gain = sum(_discount(i) for i, movie_id in enumerate(ranked[:k]) if movie_id in relevant)
    ideal = sum(_discount(i) for i in range(min(k, len(relevant))))
    return gain / ideal if ideal else 0.0


def coverage(recommended: set[int], catalogue_size: int) -> float:
    """Share of the catalogue the recommender ever suggests -- 0.02 means it reuses 2% of it."""
    if catalogue_size <= 0:
        return 0.0
    return len(recommended) / catalogue_size


def novelty(recommended: Sequence[int], popularity: dict[int, int], total: int) -> float:
    """Mean self-information of the recommendations, in bits.

    Low means blockbusters everyone has already seen; high means the long tail.
    Read it next to precision -- a recommender can buy either one with the other.
    """
    if not recommended or total <= 0:
        return 0.0
    return sum(
        -math.log2(max(popularity.get(movie_id, 0), 1) / total) for movie_id in recommended
    ) / len(recommended)


def _hits(ranked: Sequence[int], relevant: set[int], k: int) -> int:
    return sum(1 for movie_id in ranked[:k] if movie_id in relevant)


def _discount(position: int) -> float:
    return 1.0 / math.log2(position + 2)
