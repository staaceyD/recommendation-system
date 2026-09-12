from app.models.genre import Genre
from app.models.movie import Movie, movie_genres
from app.models.rating import Rating
from app.models.user import User

__all__ = ["User", "Genre", "Movie", "Rating", "movie_genres"]
