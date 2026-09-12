from flask import Blueprint, jsonify, request

from app.extensions import db
from app.models import Movie
from app.services import collaborative, rating_based

bp = Blueprint("recommendations", __name__)

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


@bp.get("/recommendations")
def recommendations():
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify(error="user_id is required and must be an integer"), 400

    limit = request.args.get("limit", DEFAULT_LIMIT, type=int) or DEFAULT_LIMIT
    limit = max(1, min(limit, MAX_LIMIT))

    movie_ids = rating_based.recommend(user_id, limit)
    strategy = "rating_based"
    if not movie_ids:
        movie_ids = collaborative.recommend(user_id, limit)
        strategy = "collaborative"

    movies = {
        m.id: m
        for m in db.session.execute(db.select(Movie).where(Movie.id.in_(movie_ids))).scalars()
    }
    return jsonify(
        user_id=user_id,
        strategy=strategy,
        recommendations=[movies[mid].to_dict() for mid in movie_ids if mid in movies],
    )
