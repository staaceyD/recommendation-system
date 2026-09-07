import pytest

from data import preprocess


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_movies_explodes_pipe_genres(csv_file, tmp_path):
    src = csv_file(
        "movies.csv",
        ["movieId", "title", "genres"],
        [[1, "Toy Story (1995)", "Adventure|Animation"], [2, "Heat (1995)", "Action"]],
    )
    out = tmp_path / "out.csv"

    assert preprocess.preprocess_movies(src, out) == (2, 3)
    assert lines(out) == [
        "movieId,title,genre",
        "1,Toy Story (1995),Adventure",
        "1,Toy Story (1995),Animation",
        "2,Heat (1995),Action",
    ]


def test_movies_without_genres_get_one_placeholder_row(csv_file, tmp_path):
    src = csv_file(
        "movies.csv",
        ["movieId", "title", "genres"],
        [[3, "Mystery (1999)", "(no genres listed)"], [4, "Blank (2000)", ""]],
    )
    out = tmp_path / "out.csv"

    assert preprocess.preprocess_movies(src, out) == (2, 2)
    assert lines(out)[1:] == [
        "3,Mystery (1999),(no genres listed)",
        "4,Blank (2000),(no genres listed)",
    ]


def test_movies_quotes_title_containing_comma(csv_file, tmp_path):
    src = csv_file(
        "movies.csv",
        ["movieId", "title", "genres"],
        [[11, "American President, The (1995)", "Comedy|Romance"]],
    )
    out = tmp_path / "out.csv"
    preprocess.preprocess_movies(src, out)

    assert lines(out)[1] == '11,"American President, The (1995)",Comedy'


def test_movies_rejects_unexpected_header(csv_file, tmp_path):
    src = csv_file("movies.csv", ["movieId", "title", "genre"], [[1, "X", "Action"]])
    with pytest.raises(SystemExit):
        preprocess.preprocess_movies(src, tmp_path / "out.csv")


def test_ratings_drops_timestamp_column(csv_file, tmp_path):
    src = csv_file(
        "ratings.csv",
        ["userId", "movieId", "rating", "timestamp"],
        [[1, 17, 4.0, 964982703], [1, 25, 1.5, 964981247]],
    )
    out = tmp_path / "out.csv"

    assert preprocess.preprocess_ratings(src, out) == 2
    assert lines(out) == ["userId,movieId,rating", "1,17,4.0", "1,25,1.5"]


def test_ratings_rejects_unexpected_header(csv_file, tmp_path):
    src = csv_file("ratings.csv", ["user", "movie", "rating", "ts"], [[1, 2, 3.0, 4]])
    with pytest.raises(SystemExit):
        preprocess.preprocess_ratings(src, tmp_path / "out.csv")
