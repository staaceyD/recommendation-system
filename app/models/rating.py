from app.extensions import db


class Rating(db.Model):
    __tablename__ = "ratings"
    __table_args__ = (db.UniqueConstraint("user_id", "movie_id", name="uq_rating_user_movie"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey("movies.id"), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    rated_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User", back_populates="ratings")
    movie = db.relationship("Movie", back_populates="ratings")

    def __repr__(self):
        return f"<Rating user={self.user_id} movie={self.movie_id} rating={self.rating}>"
