import pytest

from app.services import collaborative


@pytest.fixture(autouse=True)
def _small_thresholds(monkeypatch):
    monkeypatch.setattr(collaborative, "MIN_RATINGS", 1)
    monkeypatch.setattr(collaborative, "PRIOR_STRENGTH", 5)


@pytest.fixture
def popular(make_movie, make_user, make_rating):
    movies = {
        "hit": make_movie("Hit", ["Action"]),
        "mid": make_movie("Mid", ["Action"]),
        "flop": make_movie("Flop", ["Drama"]),
        "niche": make_movie("Niche", ["Drama"]),
    }
    raters = [make_user() for _ in range(10)]
    for rater in raters:
        make_rating(rater, movies["hit"], 5.0)
        make_rating(rater, movies["mid"], 3.5)
    for rater in raters[:4]:
        make_rating(rater, movies["flop"], 2.0)
    make_rating(raters[0], movies["niche"], 5.0)
    return movies


def test_ranks_loved_and_widely_rated_first(popular):
    recs = collaborative.recommend(999, limit=10)

    assert recs[0] == popular["hit"].id
    assert recs[-1] == popular["flop"].id


def test_excludes_movies_the_user_already_rated(popular, make_user, make_rating):
    user = make_user()
    make_rating(user, popular["hit"], 4.0)

    assert popular["hit"].id not in collaborative.recommend(user.id)


def test_prefers_genres_the_user_has_rated(popular, make_user, make_rating):
    user = make_user()
    make_rating(user, popular["niche"], 4.0)  # Drama

    recs = collaborative.recommend(user.id, limit=2)

    assert recs[0] == popular["flop"].id  # the only other Drama title


def test_backfills_when_the_genre_pool_is_too_small(popular, make_user, make_rating):
    user = make_user()
    make_rating(user, popular["flop"], 4.0)  # Drama; only "niche" left in Drama

    recs = collaborative.recommend(user.id, limit=3)

    assert len(recs) == 3
    assert popular["flop"].id not in recs
