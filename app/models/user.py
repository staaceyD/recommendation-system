from datetime import datetime, timezone

from app.extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    ratings = db.relationship("Rating", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.id} {self.username!r}>"
