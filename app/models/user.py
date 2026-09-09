from datetime import UTC, datetime

from app.extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))

    ratings = db.relationship("Rating", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.id}>"
