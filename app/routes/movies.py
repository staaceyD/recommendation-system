from datetime import UTC, datetime

from flask import Blueprint, jsonify, request
from sqlalchemy.orm import selectinload

from app.extensions import db
from app.models import Genre, Movie, Rating, User

bp = Blueprint("movies", __name__)

DEFAULT_PER_PAGE = 20
MAX_PER_PAGE = 100

MIN_RATING = 0.5
MAX_RATING = 5.0


@bp.get("/movies")
def list_movies():
    title_search = request.args.get("q", type=str)
    genre = request.args.get("genre", type=str)
    page = request.args.get("page", 1, type=int) or 1
    per_page = request.args.get("per_page", DEFAULT_PER_PAGE, type=int) or DEFAULT_PER_PAGE
    per_page = max(1, min(per_page, MAX_PER_PAGE))

    query = db.select(Movie).options(selectinload(Movie.genres)).order_by(Movie.id)
    if title_search:
        query = query.where(Movie.title.ilike(f"%{title_search}%"))
    if genre:
        query = query.join(Movie.genres).where(Genre.name == genre)

    pagination = db.paginate(query, page=page, per_page=per_page, error_out=False)
    return jsonify(
        movies=[m.to_dict() for m in pagination.items],
        page=pagination.page,
        per_page=per_page,
        total=pagination.total,
    )


@bp.post("/movies/<int:movie_id>/rate")
def rate_movie(movie_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="request body must be a JSON object"), 400

    user_id = payload.get("user_id")
    rating = payload.get("rating")
    if not _is_int(user_id):
        return jsonify(error="user_id is required and must be an integer"), 400
    if not _is_number(rating) or not _valid_rating(rating):
        return jsonify(
            error=f"rating must be between {MIN_RATING} and {MAX_RATING} in 0.5 steps"
        ), 400

    if db.session.get(Movie, movie_id) is None:
        return jsonify(error=f"movie {movie_id} not found"), 404
    if db.session.get(User, user_id) is None:
        return jsonify(error=f"user {user_id} not found"), 404

    existing = db.session.execute(
        db.select(Rating).where(Rating.user_id == user_id, Rating.movie_id == movie_id)
    ).scalar_one_or_none()

    if existing is None:
        row = Rating(
            user_id=user_id, movie_id=movie_id, rating=float(rating), rated_at=datetime.now(UTC)
        )
        db.session.add(row)
        db.session.commit()
        return jsonify(row.to_dict()), 201

    existing.rating = float(rating)
    existing.rated_at = datetime.now(UTC)
    db.session.commit()
    return jsonify(existing.to_dict()), 200


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_rating(value):
    return MIN_RATING <= value <= MAX_RATING and (value * 2) % 1 == 0
