import csv

import pytest

from app import create_app
from app.extensions import db
from app.models import Genre, Movie, Rating, User
from app.services import collaborative, rating_based
from data import seed


@pytest.fixture(scope="session")
def _app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app


@pytest.fixture(autouse=True)
def _clean_db(_app):
    db.session.remove()
    for table in reversed(db.metadata.sorted_tables):
        db.session.execute(table.delete())
    db.session.commit()
    rating_based.reset()
    collaborative.reset()
    yield


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    path = tmp_path / "mf"
    monkeypatch.setenv("MODEL_DIR", str(path))
    return path


@pytest.fixture
def train_model(model_dir):
    def _train(**overrides):
        from app.ml.train import train

        opts = {"dim": 8, "epochs": 40, "batch_size": 256, "validation_split": 0.0, "verbose": 0}
        opts.update(overrides)
        history = train(model_dir, **opts)
        rating_based.reset()
        return history

    return _train


@pytest.fixture
def client(_app):
    return _app.test_client()


@pytest.fixture
def make_movie(_app):
    def _make(title, genres=()):
        movie = Movie(title=title)
        for name in genres:
            existing = db.session.execute(
                db.select(Genre).where(Genre.name == name)
            ).scalar_one_or_none()
            movie.genres.append(existing or Genre(name=name))
        db.session.add(movie)
        db.session.commit()
        return movie

    return _make


@pytest.fixture
def make_user(_app):
    def _make(user_id=None):
        user = User(id=user_id)
        db.session.add(user)
        db.session.commit()
        return user

    return _make


@pytest.fixture
def make_rating(_app):
    def _make(user, movie, rating):
        row = Rating(
            user_id=getattr(user, "id", user),
            movie_id=getattr(movie, "id", movie),
            rating=rating,
        )
        db.session.add(row)
        db.session.commit()
        return row

    return _make


@pytest.fixture
def preferences(make_movie, make_user, make_rating):
    """Two clean taste groups: action fans and romance fans.

    ``target`` is an action fan who has only rated the first four movies of each
    genre, so the last two of each are unseen-but-learnable candidates.
    """
    action = [make_movie(f"Action {i}", ["Action"]) for i in range(6)]
    romance = [make_movie(f"Romance {i}", ["Romance"]) for i in range(6)]
    action_fans = [make_user() for _ in range(6)]
    romance_fans = [make_user() for _ in range(6)]
    target = action_fans[0]

    for fan in action_fans:
        seen_action = action[:4] if fan is target else action
        seen_romance = romance[:4] if fan is target else romance
        for movie in seen_action:
            make_rating(fan, movie, 5.0)
        for movie in seen_romance:
            make_rating(fan, movie, 1.5)
    for fan in romance_fans:
        for movie in romance:
            make_rating(fan, movie, 5.0)
        for movie in action:
            make_rating(fan, movie, 1.5)

    return {
        "action": action,
        "romance": romance,
        "target": target,
        "all_users": action_fans + romance_fans,
    }


@pytest.fixture
def db_conn(_app):
    raw = db.engine.raw_connection()
    cur = raw.cursor()
    yield raw, cur
    raw.rollback()
    raw.close()


@pytest.fixture
def csv_file(tmp_path):
    def _make(name, header, rows):
        path = tmp_path / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    return _make


@pytest.fixture
def seed_csvs(monkeypatch, csv_file):
    def _set(movies_rows, ratings_rows):
        movies = csv_file("movies.csv", ["movieId", "title", "genre"], movies_rows)
        ratings = csv_file("ratings.csv", ["userId", "movieId", "rating"], ratings_rows)
        monkeypatch.setattr(seed, "MOVIES_CSV", movies)
        monkeypatch.setattr(seed, "RATINGS_CSV", ratings)

    return _set
