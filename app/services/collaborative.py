"""Cold-start recommendations: popular, well-rated movies.

Used when the trained model has no embedding for the user (new user, or nobody
in the training data). Ranks movies by a Bayesian-adjusted mean rating so a
handful of 5-star ratings can't beat a genuinely popular title, and -- if the
user has rated anything at all -- prefers movies sharing those genres.
"""

from __future__ import annotations

from collections import defaultdict

from app.extensions import db
from app.models import Rating, movie_genres

DEFAULT_LIMIT = 20

# Movies below this many ratings are ignored; the prior pulls a movie's score
# toward the overall mean with a weight of PRIOR_STRENGTH ratings.
MIN_RATINGS = 50
PRIOR_STRENGTH = 50

_ranking: dict[tuple[int, int], list[tuple[int, frozenset[int]]]] = {}


def recommend(user_id: int, limit: int = DEFAULT_LIMIT) -> list[int]:
    ranking = _get_ranking()

    seen = set(
        db.session.execute(db.select(Rating.movie_id).where(Rating.user_id == user_id)).scalars()
    )
    liked_genres = _genres_of(seen)

    picks = _take(ranking, limit, exclude=seen, genres=liked_genres)
    if len(picks) < limit:  # genre filter too tight -- backfill on pure popularity
        picks += _take(ranking, limit - len(picks), exclude=seen | set(picks), genres=None)
    return picks


def _take(ranking, limit, *, exclude, genres):
    out = []
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
    key = (MIN_RATINGS, PRIOR_STRENGTH)
    if key not in _ranking:
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
                (_bayesian(count, avg, overall_mean, PRIOR_STRENGTH), movie_id)
                for movie_id, count, avg in stats
            ),
            reverse=True,
        )
        _ranking[key] = [
            (movie_id, frozenset(genres_by_movie.get(movie_id, ()))) for _, movie_id in scored
        ]
    return _ranking[key]


def _bayesian(count: int, avg: float, overall_mean: float, prior: int) -> float:
    return (count * avg + prior * overall_mean) / (count + prior)


def reset() -> None:
    _ranking.clear()
