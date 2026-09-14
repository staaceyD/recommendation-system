import numpy as np
import pandas as pd
import pytest

from app.ml import evaluate
from app.ml.evaluate import (
    PopularityRecommender,
    RandomRecommender,
    build_cases,
    split_ratings,
)

ACTION = list(range(1, 41))
ROMANCE = list(range(101, 141))


@pytest.fixture(scope="module")
def taste_frame():
    """Two taste groups rating a random half of the catalogue -- separable, but sparse."""
    rng = np.random.default_rng(7)
    rows = []
    for group_index, group in enumerate((ACTION, ROMANCE)):
        favourites = set(group)
        for offset in range(60):
            user_id = 1000 * (group_index + 1) + offset
            for movie_id in ACTION + ROMANCE:
                if rng.random() < 0.5:
                    rows.append((user_id, movie_id, 5.0 if movie_id in favourites else 1.5))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def frame_of(rows):
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def test_split_holds_out_a_share_of_every_users_ratings(taste_frame):
    train, test = split_ratings(taste_frame, holdout=0.2)

    assert len(train) + len(test) == len(taste_frame)
    for user_id, group in test.groupby("user_id"):
        total = (taste_frame["user_id"] == user_id).sum()
        assert len(group) == round(total * 0.2)


def test_split_leaves_every_user_some_training_history(taste_frame):
    train, test = split_ratings(taste_frame, holdout=0.9)

    assert set(test["user_id"]) <= set(train["user_id"])


def test_split_never_puts_the_same_rating_on_both_sides(taste_frame):
    train, test = split_ratings(taste_frame, holdout=0.2)

    pairs = {"user_id", "movie_id"}
    overlap = train[list(pairs)].merge(test[list(pairs)], on=list(pairs))
    assert overlap.empty


def test_split_keeps_users_with_too_few_ratings_whole():
    frame = frame_of(
        [(1, m, 4.0) for m in range(3)] + [(user, m, 4.0) for user in (2, 3) for m in range(10)]
    )

    train, test = split_ratings(frame, holdout=0.2, min_ratings=5)

    assert set(test["user_id"]) == {2, 3}
    assert (train["user_id"] == 1).sum() == 3


def test_split_drops_holdout_movies_the_model_could_never_rank():
    # movie 99 is rated once, by one user -- if that rating lands in the holdout the
    # movie is absent from training, so no recommender can return it
    frame = frame_of([(1, m, 4.0) for m in range(10)] + [(1, 99, 5.0), (2, 99, 5.0)])
    frame = pd.concat([frame, frame_of([(2, m, 4.0) for m in range(10)])])

    train, test = split_ratings(frame, holdout=1.0, min_ratings=2)

    assert set(test["movie_id"]) <= set(train["movie_id"])


def test_cases_only_count_movies_the_user_actually_liked():
    train = frame_of([(1, 10, 5.0), (1, 11, 5.0)])
    test = frame_of([(1, 20, 5.0), (1, 21, 1.0)])

    ((user_id, seen, relevant),) = build_cases(train, test, num_users=10)

    assert user_id == 1
    assert seen == {10, 11}
    assert relevant == {20}


def test_cases_skip_users_whose_holdout_has_nothing_they_liked():
    train = frame_of([(1, 10, 5.0), (2, 10, 5.0)])
    test = frame_of([(1, 20, 5.0), (2, 21, 0.5)])

    assert [case[0] for case in build_cases(train, test, num_users=10)] == [1]


def test_random_recommender_never_returns_a_movie_the_user_has_seen():
    recommender = RandomRecommender(list(range(50)))

    picks = recommender.recommend(1, seen=set(range(40)), k=5)

    assert len(picks) == 5
    assert not set(picks) & set(range(40))


def test_popularity_prefers_the_better_rated_of_two_equally_popular_movies():
    rows = [(user, 1, 5.0) for user in range(60)] + [(user, 2, 2.0) for user in range(60)]

    recommender = PopularityRecommender(frame_of(rows))

    assert recommender.recommend(999, seen=set(), k=2) == [1, 2]


def test_popularity_ignores_movies_almost_nobody_rated():
    rows = [(user, 1, 4.0) for user in range(60)] + [(0, 2, 5.0), (1, 2, 5.0)]

    recommender = PopularityRecommender(frame_of(rows))

    assert recommender.recommend(999, seen=set(), k=5) == [1]


@pytest.fixture(scope="module")
def result(taste_frame):
    return evaluate.run(
        frame=taste_frame, k=10, num_users=200, dim=8, epochs=30, batch_size=256, verbose=0
    )


def test_the_model_beats_random_on_separable_tastes(result):
    scores = {row["strategy"]: row for row in result["strategies"]}
    model, chance = scores["matrix-factorization"], scores["random"]

    for metric in ("precision", "recall", "ndcg", "map"):
        assert model[metric] > 2 * chance[metric]
    assert model["hit_rate"] > chance["hit_rate"]


def test_every_strategy_is_scored_on_the_same_users(result):
    # a handful of the 120 users happen to hold out nothing they liked, and are dropped
    assert 100 <= result["users_scored"] <= 120
    assert {row["strategy"] for row in result["strategies"]} == {
        "matrix-factorization",
        "popularity",
        "random",
    }


def test_the_model_predicts_ratings_better_than_the_global_mean(result):
    accuracy = result["rating_accuracy"]

    assert accuracy["rmse"] < accuracy["baseline_rmse"]
    assert accuracy["rated"] == result["test_ratings"]


def test_report_reads_as_a_table(result):
    text = evaluate.format_report(result)

    assert "prec@10" in text
    assert "matrix-factorization" in text
    assert "random" in text


def test_run_refuses_an_empty_ratings_table():
    with pytest.raises(SystemExit):
        evaluate.run(frame=frame_of([]))
