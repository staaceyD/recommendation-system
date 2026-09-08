"""Seed the database from the MovieLens CSVs in ``data/raw/``.

Usage:

    uv run python data/seed.py                 # full seed (~32M ratings)
    uv run python data/seed.py --reset         # wipe the tables first, then seed
    uv run python data/seed.py --ratings-limit 500000   # partial seed for dev/testing

Respects ``FLASK_ENV`` for which database to target (via the app config), so it
never writes to the test database unless explicitly asked to.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent / "raw"
MOVIES_CSV = RAW_DIR / "movies.csv"
RATINGS_CSV = RAW_DIR / "ratings.csv"

NO_GENRES = "(no genres listed)"

# Child-to-parent order, safe for TRUNCATE / delete.
TABLES = ("ratings", "movie_genres", "users", "movies", "genres")

# Row count of the full ratings file (minus header); only used for progress %.
RATINGS_TOTAL_ESTIMATE = 32_000_204

INSERT_CHUNK = 10_000
RATINGS_BATCH = 50_000


def chunked_executemany(cursor, sql: str, rows, chunk: int = INSERT_CHUNK) -> int:
    """``executemany`` in bounded batches so we never blow past ``max_allowed_packet``."""
    total = 0
    batch: list = []
    for row in rows:
        batch.append(row)
        if len(batch) >= chunk:
            cursor.executemany(sql, batch)
            total += len(batch)
            batch.clear()
    if batch:
        cursor.executemany(sql, batch)
        total += len(batch)
    return total


def missing_tables(cursor) -> list[str]:
    """Return the seedable tables that don't exist yet."""
    cursor.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"
    )
    present = {row[0].lower() for row in cursor.fetchall()}
    return [t for t in TABLES if t not in present]


def table_counts(cursor) -> dict[str, int]:
    """Return the current row count of each seedable table, keyed by table name.

    Used before seeding to detect a non-empty database (so we can refuse to run
    without ``--reset``) and after seeding to print a summary of what actually
    landed in the database.
    """
    counts = {}
    for table in TABLES:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = cursor.fetchone()[0]
    return counts


def reset_tables(cursor) -> None:
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    for table in TABLES:
        cursor.execute(f"TRUNCATE TABLE {table}")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")


def load_movies_and_genres(cursor) -> set[int]:
    """Populate ``genres``, ``movies``, ``movie_genres``. Return the set of movie ids."""
    titles: dict[int, str] = {}
    genres_by_movie: dict[int, set[str]] = defaultdict(set)

    with MOVIES_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            movie_id = int(row[0])
            if movie_id not in titles:
                titles[movie_id] = row[1]
            genre = row[2].strip()
            if genre and genre != NO_GENRES:
                genres_by_movie[movie_id].add(genre)

    all_genres = sorted({g for gs in genres_by_movie.values() for g in gs})
    chunked_executemany(cursor, "INSERT INTO genres (name) VALUES (%s)", ((g,) for g in all_genres))
    cursor.execute("SELECT id, name FROM genres")
    genre_id = {name: gid for gid, name in cursor.fetchall()}

    chunked_executemany(
        cursor,
        "INSERT INTO movies (id, title) VALUES (%s, %s)",
        titles.items(),
    )
    chunked_executemany(
        cursor,
        "INSERT INTO movie_genres (movie_id, genre_id) VALUES (%s, %s)",
        ((mid, genre_id[g]) for mid, gs in genres_by_movie.items() for g in gs),
    )

    print(
        f"  genres:       {len(all_genres):>10,}\n"
        f"  movies:       {len(titles):>10,}\n"
        f"  movie_genres: {sum(len(gs) for gs in genres_by_movie.values()):>10,}"
    )
    return set(titles)


def load_users_and_ratings(
    connection, cursor, valid_movie_ids: set[int], ratings_limit: int | None
) -> None:
    created_at = datetime.now(UTC).replace(tzinfo=None, microsecond=0)

    seen_users: set[int] = set()
    pending_users: list[tuple[int, datetime]] = []
    pending_ratings: list[tuple[int, int, float]] = []

    read = 0
    inserted_ratings = 0
    skipped_missing_movie = 0
    start = time.monotonic()

    def flush() -> None:
        nonlocal inserted_ratings
        if pending_users:
            chunked_executemany(
                cursor,
                "INSERT IGNORE INTO users (id, created_at) VALUES (%s, %s)",
                pending_users,
            )
            pending_users.clear()
        if pending_ratings:
            chunked_executemany(
                cursor,
                "INSERT IGNORE INTO ratings (user_id, movie_id, rating) VALUES (%s, %s, %s)",
                pending_ratings,
            )
            inserted_ratings += len(pending_ratings)
            pending_ratings.clear()
        connection.commit()

    with RATINGS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if ratings_limit is not None and read >= ratings_limit:
                break
            read += 1
            user_id = int(row[0])
            movie_id = int(row[1])

            if movie_id not in valid_movie_ids:
                skipped_missing_movie += 1
                continue

            if user_id not in seen_users:
                seen_users.add(user_id)
                pending_users.append((user_id, created_at))
            pending_ratings.append((user_id, movie_id, float(row[2])))

            if len(pending_ratings) >= RATINGS_BATCH:
                flush()
                elapsed = time.monotonic() - start
                rate = read / elapsed if elapsed else 0
                pct = (
                    100 * read / ratings_limit
                    if ratings_limit
                    else 100 * read / RATINGS_TOTAL_ESTIMATE
                )
                print(
                    f"  ratings: {read:>12,} read ({pct:5.1f}%)  "
                    f"{inserted_ratings:>12,} inserted  "
                    f"{rate:>8,.0f} rows/s",
                    end="\r",
                    flush=True,
                )

    flush()
    print()
    print(
        f"  users:        {len(seen_users):>10,}\n"
        f"  ratings:      {inserted_ratings:>10,} inserted "
        f"({skipped_missing_movie:,} skipped - movie not in movies.csv)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="TRUNCATE the seedable tables before loading.",
    )
    parser.add_argument(
        "--ratings-limit",
        type=int,
        default=None,
        metavar="N",
        help="Only read the first N rows of ratings.csv (for a quick partial seed).",
    )
    args = parser.parse_args(argv)

    for path in (MOVIES_CSV, RATINGS_CSV):
        if not path.exists():
            parser.error(f"missing input file: {path}")

    app = create_app()
    with app.app_context():
        engine = db.engine
        target = engine.url.render_as_string(hide_password=True)
        print(f"Seeding {target}")

        raw = engine.raw_connection()
        try:
            cursor = raw.cursor()

            missing = missing_tables(cursor)
            if missing:
                parser.error(
                    f"schema not initialised (missing tables: {', '.join(missing)}); "
                    "run `uv run flask db upgrade` first"
                )

            existing = table_counts(cursor)
            if any(existing.values()):
                if not args.reset:
                    parser.error(
                        "tables are not empty "
                        f"({', '.join(f'{k}={v:,}' for k, v in existing.items() if v)}); "
                        "pass --reset to wipe and re-seed"
                    )
                print("Resetting tables (--reset)...")
                reset_tables(cursor)
                raw.commit()

            # Speed up the bulk load; our own filtering keeps referential integrity.
            # NB: unique_checks is left on -- INSERT IGNORE relies on the unique-index
            # check to drop duplicate (user_id, movie_id) rows, and disabling it would
            # let duplicates corrupt uq_rating_user_movie.
            cursor.execute("SET SESSION foreign_key_checks = 0")

            print("Loading movies and genres...")
            valid_movie_ids = load_movies_and_genres(cursor)
            raw.commit()

            print("Loading users and ratings...")
            load_users_and_ratings(raw, cursor, valid_movie_ids, args.ratings_limit)

            cursor.execute("SET SESSION foreign_key_checks = 1")
            raw.commit()

            print("Final row counts:")
            for table, count in table_counts(cursor).items():
                print(f"  {table:<13} {count:>12,}")
        finally:
            raw.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
