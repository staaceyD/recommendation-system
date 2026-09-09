from flask import Blueprint, jsonify, request
from sqlalchemy.orm import selectinload

from app.extensions import db
from app.models import Genre, Movie

bp = Blueprint("movies", __name__)

DEFAULT_PER_PAGE = 20
MAX_PER_PAGE = 100


@bp.get("/movies")
def list_movies():
    q = request.args.get("q", type=str)
    genre = request.args.get("genre", type=str)
    page = request.args.get("page", 1, type=int) or 1
    per_page = request.args.get("per_page", DEFAULT_PER_PAGE, type=int) or DEFAULT_PER_PAGE
    per_page = max(1, min(per_page, MAX_PER_PAGE))

    query = db.select(Movie).options(selectinload(Movie.genres)).order_by(Movie.id)
    if q:
        query = query.where(Movie.title.ilike(f"%{q}%"))
    if genre:
        query = query.join(Movie.genres).where(Genre.name == genre)

    pagination = db.paginate(query, page=page, per_page=per_page, error_out=False)
    return jsonify(
        movies=[m.to_dict() for m in pagination.items],
        page=pagination.page,
        per_page=per_page,
        total=pagination.total,
    )
