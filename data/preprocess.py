"""Turn the raw MovieLens ``ml-32m`` CSVs into the shape ``seed.py`` expects."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
SRC_DIR = RAW_DIR / "ml-32m"

SRC_MOVIES = SRC_DIR / "movies.csv"
SRC_RATINGS = SRC_DIR / "ratings.csv"
OUT_MOVIES = RAW_DIR / "movies.csv"
OUT_RATINGS = RAW_DIR / "ratings.csv"

NO_GENRES = "(no genres listed)"


def preprocess_movies(src: Path, out: Path) -> tuple[int, int]:
    """Explode ``movieId,title,genres`` into one ``movieId,title,genre`` row per pair."""
    movies = 0
    rows_out = 0
    with (
        src.open(newline="", encoding="utf-8") as fh_in,
        out.open("w", newline="", encoding="utf-8") as fh_out,
    ):
        reader = csv.reader(fh_in)
        writer = csv.writer(fh_out)
        header = next(reader)
        if header[:3] != ["movieId", "title", "genres"]:
            sys.exit(f"unexpected header in {src}: {header!r}")
        writer.writerow(["movieId", "title", "genre"])
        for movie_id, title, genres in reader:
            movies += 1
            parts = [g for g in genres.split("|") if g] or [NO_GENRES]
            for genre in parts:
                writer.writerow([movie_id, title, genre])
                rows_out += 1
    return movies, rows_out


def preprocess_ratings(src: Path, out: Path) -> int:
    """Drop the timestamp column: ``userId,movieId,rating,timestamp`` -> ``...,rating``."""
    count = 0
    with (
        src.open(newline="", encoding="utf-8") as fh_in,
        out.open("w", newline="", encoding="utf-8") as fh_out,
    ):
        reader = csv.reader(fh_in)
        writer = csv.writer(fh_out)
        header = next(reader)
        if header[:3] != ["userId", "movieId", "rating"]:
            sys.exit(f"unexpected header in {src}: {header!r}")
        writer.writerow(["userId", "movieId", "rating"])
        for row in reader:
            writer.writerow(row[:3])
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    for path in (SRC_MOVIES, SRC_RATINGS):
        if not path.exists():
            parser.error(
                f"missing input file: {path}\n"
                "Download ml-32m.zip from https://grouplens.org/datasets/movielens/ "
                "and unzip it into data/raw/ (creating data/raw/ml-32m/)."
            )

    print(f"movies:  {SRC_MOVIES} -> {OUT_MOVIES}")
    movies, movie_rows = preprocess_movies(SRC_MOVIES, OUT_MOVIES)
    print(f"  {movies:,} movies -> {movie_rows:,} movie/genre rows")

    print(f"ratings: {SRC_RATINGS} -> {OUT_RATINGS}")
    ratings = preprocess_ratings(SRC_RATINGS, OUT_RATINGS)
    print(f"  {ratings:,} ratings")

    print("Done. Now run: uv run python data/seed.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
