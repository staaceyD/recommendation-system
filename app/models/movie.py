from app.extensions import db

movie_genres = db.Table(
    "movie_genres",
    db.Column("movie_id", db.Integer, db.ForeignKey("movies.id"), primary_key=True),
    db.Column("genre_id", db.Integer, db.ForeignKey("genres.id"), primary_key=True),
)


class Movie(db.Model):
    __tablename__ = "movies"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)

    genres = db.relationship("Genre", secondary=movie_genres, backref="movies")
    ratings = db.relationship("Rating", back_populates="movie", cascade="all, delete-orphan")

    def to_dict(self):
        return {"id": self.id, "title": self.title, "genres": sorted(g.name for g in self.genres)}

    def __repr__(self):
        return f"<Movie {self.id} {self.title!r}>"
