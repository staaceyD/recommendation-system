"""Cold-start recommendations: popular, well-rated movies.

Used when the trained model has no embedding for the user (new user, or nobody
in the training data). Ranks movies by a Bayesian-adjusted mean rating so a
handful of 5-star ratings can't beat a genuinely popular title, and -- if the
user has rated anything highly -- prefers movies sharing those genres.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from app.extensions import db
from app.models import Rating, movie_genres

DEFAULT_LIMIT = 20

# Movies below this many ratings are ignored; the prior pulls a movie's score
# toward the overall mean with a weight of PRIOR_STRENGTH ratings.
MIN_RATINGS = 50
PRIOR_STRENGTH = 50

# Ratings at or above this count as "liked" for the genre preference.
LIKED_RATING = 3.5

# The ranking aggregates the whole ratings table, so it can't be rebuilt per
# request -- cache it, but expire it so new ratings do eventually show up.
CACHE_TTL_SECONDS = 900

_ranking: list[tuple[int, frozenset[int]]] | None = None
_built_at = 0.0
_lock = threading.Lock()


def recommend(user_id: int, limit: int = DEFAULT_LIMIT) -> list[int]:
    ranking = _get_ranking()

    rated = db.session.execute(
        db.select(Rating.movie_id, Rating.rating).where(Rating.user_id == user_id)
    ).all()
    seen = {movie_id for movie_id, _ in rated}
    liked_genres = _genres_of({mid for mid, rating in rated if rating >= LIKED_RATING})

    picks = _take(ranking, limit, exclude=seen, genres=liked_genres)
    if len(picks) < limit:  # genre filter too tight -- backfill on pure popularity
        picks += _take(ranking, limit - len(picks), exclude=seen | set(picks), genres=None)
    return picks


def _take(ranking, limit, *, exclude, genres):
    out = []
    if limit <= 0:
        return out
    for movie_id, movie_genre_ids in ranking:
        if movie_id in exclude:
            continue
        if genres and movie_genre_ids.isdisjoint(genres):
            continue
        out.append(movie_id)
        if len(out) == limit:
            break
    return out


def _genres_of(movie_ids: set[int]) -> set[int]:
    if not movie_ids:
        return set()
    return set(
        db.session.execute(
            db.select(movie_genres.c.genre_id).where(movie_genres.c.movie_id.in_(movie_ids))
        ).scalars()
    )


def _get_ranking() -> list[tuple[int, frozenset[int]]]:
    global _ranking, _built_at
    if _stale():
        with _lock:
            if _stale():  # another thread may have built it while we waited
                _ranking = _build_ranking()
                _built_at = time.monotonic()
    return _ranking


def _stale() -> bool:
    return _ranking is None or time.monotonic() - _built_at >= CACHE_TTL_SECONDS


def _build_ranking() -> list[tuple[int, frozenset[int]]]:
    stats = db.session.execute(
        db.select(Rating.movie_id, db.func.count(), db.func.avg(Rating.rating))
        .group_by(Rating.movie_id)
        .having(db.func.count() >= MIN_RATINGS)
    ).all()
    total = sum(count for _, count, _ in stats)
    overall_mean = sum(count * avg for _, count, avg in stats) / total if total else 0.0

    genres_by_movie = defaultdict(set)
    for movie_id, genre_id in db.session.execute(
        db.select(movie_genres.c.movie_id, movie_genres.c.genre_id)
    ):
        genres_by_movie[movie_id].add(genre_id)

    scored = sorted(
        (
            (bayesian_score(count, avg, overall_mean, PRIOR_STRENGTH), movie_id)
            for movie_id, count, avg in stats
        ),
        reverse=True,
    )
    return [(movie_id, frozenset(genres_by_movie.get(movie_id, ()))) for _, movie_id in scored]


def bayesian_score(count: int, avg: float, overall_mean: float, prior: int) -> float:
    """A movie's mean rating pulled toward `overall_mean` by `prior` imaginary average ratings."""
    return (count * avg + prior * overall_mean) / (count + prior)


def reset() -> None:
    """Drop the cached ranking (tests, or after a bulk ratings change)."""
    global _ranking, _built_at
    _ranking, _built_at = None, 0.0
