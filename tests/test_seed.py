from data import seed


class RecordingCursor:
    def __init__(self):
        self.batches = []

    def executemany(self, sql, rows):
        self.batches.append(list(rows))


def test_chunked_executemany_splits_into_batches():
    cur = RecordingCursor()
    total = seed.chunked_executemany(cur, "SQL", range(25), chunk=10)

    assert total == 25
    assert [len(b) for b in cur.batches] == [10, 10, 5]


def test_chunked_executemany_exact_multiple_has_no_trailing_batch():
    cur = RecordingCursor()
    total = seed.chunked_executemany(cur, "SQL", range(20), chunk=10)

    assert total == 20
    assert [len(b) for b in cur.batches] == [10, 10]


def test_chunked_executemany_empty_input():
    cur = RecordingCursor()

    assert seed.chunked_executemany(cur, "SQL", [], chunk=10) == 0
    assert cur.batches == []


def test_load_movies_and_genres(db_conn, seed_csvs):
    conn, cur = db_conn
    seed_csvs(
        [
            [1, "Toy Story (1995)", "Adventure"],
            [1, "Toy Story (1995)", "Animation"],
            [2, "Heat (1995)", "Action"],
            [3, "Mystery (1999)", "(no genres listed)"],
        ],
        [],
    )

    movie_ids = seed.load_movies_and_genres(cur)
    conn.commit()

    assert movie_ids == {1, 2, 3}
    counts = seed.table_counts(cur)
    assert (counts["genres"], counts["movies"], counts["movie_genres"]) == (3, 3, 3)
    cur.execute("SELECT title FROM movies WHERE id = 2")
    assert cur.fetchone()[0] == "Heat (1995)"


def test_load_users_and_ratings_skips_unknown_movies(db_conn, seed_csvs):
    conn, cur = db_conn
    seed_csvs(
        [[1, "Toy Story (1995)", "Adventure"], [2, "Heat (1995)", "Action"]],
        [[10, 1, 4.0], [10, 2, 3.5], [11, 1, 5.0], [11, 99, 2.0]],
    )
    movie_ids = seed.load_movies_and_genres(cur)
    conn.commit()

    seed.load_users_and_ratings(conn, cur, movie_ids, None)

    counts = seed.table_counts(cur)
    assert (counts["users"], counts["ratings"]) == (2, 3)
    cur.execute("SELECT rating FROM ratings WHERE user_id = 10 AND movie_id = 1")
    assert cur.fetchone()[0] == 4.0


def test_load_users_and_ratings_respects_limit(db_conn, seed_csvs):
    conn, cur = db_conn
    seed_csvs(
        [[1, "A", "Action"], [2, "B", "Action"], [3, "C", "Action"]],
        [[10, 1, 4.0], [11, 2, 3.0], [12, 3, 2.0]],
    )
    movie_ids = seed.load_movies_and_genres(cur)
    conn.commit()

    seed.load_users_and_ratings(conn, cur, movie_ids, 2)

    counts = seed.table_counts(cur)
    assert (counts["users"], counts["ratings"]) == (2, 2)


def test_load_users_and_ratings_ignores_duplicate_rating(db_conn, seed_csvs):
    conn, cur = db_conn
    seed_csvs([[1, "A", "Action"]], [[10, 1, 4.0], [10, 1, 2.0]])
    movie_ids = seed.load_movies_and_genres(cur)
    conn.commit()

    seed.load_users_and_ratings(conn, cur, movie_ids, None)

    assert seed.table_counts(cur)["ratings"] == 1


def test_missing_tables_none_when_schema_present(db_conn):
    _, cur = db_conn
    assert seed.missing_tables(cur) == []


def test_missing_tables_reports_absent_table(db_conn, monkeypatch):
    _, cur = db_conn
    monkeypatch.setattr(seed, "TABLES", seed.TABLES + ("does_not_exist",))
    assert seed.missing_tables(cur) == ["does_not_exist"]


def test_reset_tables_empties_all(db_conn, seed_csvs):
    conn, cur = db_conn
    seed_csvs([[1, "A", "Action"]], [[10, 1, 4.0]])
    seed.load_movies_and_genres(cur)
    conn.commit()
    seed.load_users_and_ratings(conn, cur, {1}, None)
    assert any(seed.table_counts(cur).values())

    seed.reset_tables(cur)
    conn.commit()

    assert not any(seed.table_counts(cur).values())
