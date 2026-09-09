import csv

import pytest

from app import create_app
from app.extensions import db
from data import seed


@pytest.fixture(scope="session")
def _app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app


@pytest.fixture
def db_conn(_app):
    raw = db.engine.raw_connection()
    cur = raw.cursor()
    for table in seed.TABLES:
        cur.execute(f"DELETE FROM {table}")
    raw.commit()
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
