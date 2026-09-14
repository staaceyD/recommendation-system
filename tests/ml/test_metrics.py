import math

import pytest

from app.ml import metrics

# Hits at positions 1 and 3 of a five-long list; the user liked four movies overall.
RANKED = [10, 20, 30, 40, 50]
RELEVANT = {10, 30, 60, 70}


def test_precision_counts_hits_against_the_list_length():
    assert metrics.precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert metrics.precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 2)


def test_recall_counts_hits_against_everything_the_user_liked():
    assert metrics.recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 4)


def test_hit_rate_only_asks_whether_anything_landed():
    assert metrics.hit_rate_at_k(RANKED, RELEVANT, 5) == 1.0
    assert metrics.hit_rate_at_k([20, 40], RELEVANT, 2) == 0.0


def test_average_precision_rewards_hits_near_the_top():
    # positions 1 and 3 -> (1/1 + 2/3) / min(5, 4)
    assert metrics.average_precision_at_k(RANKED, RELEVANT, 5) == pytest.approx((1 + 2 / 3) / 4)
    assert metrics.average_precision_at_k(
        [30, 10, 20], RELEVANT, 3
    ) > metrics.average_precision_at_k([20, 30, 10], RELEVANT, 3)


def test_ndcg_is_one_when_every_hit_is_as_high_as_it_could_be():
    assert metrics.ndcg_at_k([10, 30, 60, 70], RELEVANT, 4) == pytest.approx(1.0)


def test_ndcg_discounts_by_position():
    expected = (1 / math.log2(2) + 1 / math.log2(4)) / (
        1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)
    )
    assert metrics.ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric",
    [
        metrics.precision_at_k,
        metrics.recall_at_k,
        metrics.ndcg_at_k,
        metrics.average_precision_at_k,
        metrics.hit_rate_at_k,
    ],
)
def test_metrics_are_zero_without_hits(metric):
    assert metric([1, 2, 3], {4, 5}, 3) == 0.0


@pytest.mark.parametrize(
    "metric",
    [metrics.recall_at_k, metrics.ndcg_at_k, metrics.average_precision_at_k],
)
def test_metrics_stay_zero_when_the_user_liked_nothing(metric):
    assert metric([1, 2, 3], set(), 3) == 0.0


def test_coverage_is_the_share_of_the_catalogue_used():
    assert metrics.coverage({1, 2}, 10) == pytest.approx(0.2)
    assert metrics.coverage(set(), 0) == 0.0


def test_novelty_is_higher_for_the_long_tail():
    popularity = {1: 500, 2: 1}  # a blockbuster and a movie nobody watched
    total = 1000
    assert metrics.novelty([1], popularity, total) == pytest.approx(1.0)
    assert metrics.novelty([2], popularity, total) > metrics.novelty([1], popularity, total)
    assert metrics.novelty([], popularity, total) == 0.0
