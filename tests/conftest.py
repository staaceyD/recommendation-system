import csv

import pytest

from app import create_app
from app.extensions import db
from app.models import Genre, Movie
from data import seed


@pytest.fixture(scope="session")
def _app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app


@pytest.fixture(autouse=True)
def _clean_db(_app):
    for table in reversed(db.metadata.sorted_tables):
        db.session.execute(table.delete())
    db.session.commit()
    yield


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
