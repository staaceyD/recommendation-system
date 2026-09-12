from app.extensions import db


class Rating(db.Model):
    __tablename__ = "ratings"
    __table_args__ = (
        db.UniqueConstraint("user_id", "movie_id", name="uq_rating_user_movie"),
        # Covering index for the item side: GROUP BY movie_id with AVG(rating)
        # reads only the index. Also serves plain movie_id lookups (leftmost col).
        db.Index("ix_ratings_movie_id_rating", "movie_id", "rating"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey("movies.id"), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    rated_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User", back_populates="ratings")
    movie = db.relationship("Movie", back_populates="ratings")

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "movie_id": self.movie_id,
            "rating": self.rating,
            "rated_at": self.rated_at.isoformat() if self.rated_at else None,
        }

    def __repr__(self):
        return f"<Rating user={self.user_id} movie={self.movie_id} rating={self.rating}>"
