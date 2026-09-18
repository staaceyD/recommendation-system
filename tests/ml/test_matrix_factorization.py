import json

import numpy as np
import pytest

from app.ml.artifact import META_FILE, MODEL_FILE, VOCAB_FILE, MFArtifact
from app.ml.model import build_model, scaled_l2
from app.ml.train import _save
from app.ml.vocab import Vocab


def test_build_model_scores_one_value_per_pair():
    model = build_model(num_users=4, num_movies=6, dim=3)
    out = model(
        {"user": np.array([0, 1, 2], dtype="int64"), "movie": np.array([5, 4, 3], dtype="int64")},
        training=False,
    )
    assert tuple(out.shape) == (3, 1)


def test_scaled_l2_holds_the_strength_per_rating_constant():
    """`l2 * N` is the invariant -- a fixed `l2` regularizes harder the more data there is."""
    products = [scaled_l2(n) * n for n in (1_600_000, 6_000_000, 32_000_000)]
    assert products == pytest.approx([products[0]] * len(products))


def test_training_scales_l2_to_the_dataset_when_it_is_not_given(
    preferences, train_model, model_dir
):
    train_model(epochs=1, l2=None)  # the fixture pins l2; this exercises the default path

    meta = json.loads((model_dir / META_FILE).read_text())
    assert meta["l2"] == pytest.approx(scaled_l2(meta["num_ratings"]))


def test_training_writes_a_loadable_artifact(preferences, train_model, model_dir):
    history = train_model(epochs=10)

    assert (model_dir / MODEL_FILE).exists()
    assert (model_dir / VOCAB_FILE).exists()
    meta = json.loads((model_dir / META_FILE).read_text())
    assert meta["num_users"] == len(preferences["all_users"])
    assert history.history["rmse"][-1] < 2.0

    artifact = MFArtifact.load(model_dir)
    assert artifact is not None
    assert artifact.knows_user(preferences["target"].id)
    assert not artifact.knows_user(999_999)


def test_load_returns_none_without_a_trained_model(model_dir):
    assert MFArtifact.load(model_dir) is None


def test_load_returns_none_for_a_torn_save(preferences, train_model, model_dir):
    train_model(epochs=1)
    (model_dir / VOCAB_FILE).unlink()  # simulates a save killed mid-write

    assert MFArtifact.load(model_dir) is None


def test_a_failed_save_leaves_the_previous_artifact_in_place(preferences, train_model, model_dir):
    train_model(epochs=1)
    before = (model_dir / VOCAB_FILE).read_text()

    class Exploding:
        def save(self, _path):
            raise RuntimeError("disk full")

    class FakeHistory:
        history = {"rmse": [0.5]}

    with pytest.raises(RuntimeError):
        _save(model_dir, Exploding(), Vocab([1]), Vocab([2]), [1], 3.5, 8, 1e-5, FakeHistory(), 1)

    assert (model_dir / VOCAB_FILE).read_text() == before
    assert MFArtifact.load(model_dir) is not None
    assert [p.name for p in model_dir.parent.iterdir()] == [model_dir.name]


def test_load_returns_none_for_incomplete_metadata(preferences, train_model, model_dir):
    train_model(epochs=1)
    (model_dir / META_FILE).write_text(json.dumps({"dim": 8}))  # global_mean never written

    assert MFArtifact.load(model_dir) is None


def test_load_rejects_a_vocab_from_another_run(preferences, train_model, model_dir):
    train_model(epochs=1)
    vocab = json.loads((model_dir / VOCAB_FILE).read_text())
    vocab["movie_ids"].append(999_999)  # stale vocab left behind by an older run
    (model_dir / VOCAB_FILE).write_text(json.dumps(vocab))

    assert MFArtifact.load(model_dir) is None


def test_rank_unseen_ignores_a_non_positive_limit(preferences, train_model, model_dir):
    train_model(epochs=1)
    artifact = MFArtifact.load(model_dir)

    assert artifact.rank_unseen(preferences["target"].id, seen=set(), limit=0) == []
    assert artifact.rank_unseen(preferences["target"].id, seen=set(), limit=-5) == []


def test_scoring_from_the_weights_matches_the_model_graph(preferences, train_model, model_dir):
    train_model(epochs=5)
    artifact = MFArtifact.load(model_dir)
    target = preferences["target"].id

    movie_ids = artifact.movies.ids
    graph = (
        np.asarray(
            artifact.model(
                {
                    "user": np.full(len(movie_ids), artifact.users.to_index(target), dtype="int64"),
                    "movie": np.arange(len(movie_ids), dtype="int64"),
                },
                training=False,
            )
        ).reshape(-1)
        + artifact.global_mean
    )

    assert artifact.scores(target) == pytest.approx(graph, abs=1e-4)
    assert artifact.predict_pairs([target] * len(movie_ids), movie_ids) == pytest.approx(
        graph, abs=1e-4
    )


def test_scoring_an_unknown_id(preferences, train_model, model_dir):
    train_model(epochs=1)
    artifact = MFArtifact.load(model_dir)
    known = preferences["action"][0].id

    assert artifact.scores(999_999) is None
    predicted = artifact.predict_pairs([999_999, preferences["target"].id], [known, 999_999])
    assert np.isnan(predicted).all()


def test_the_support_filter_holds_back_thinly_rated_movies(preferences, train_model, model_dir):
    train_model(epochs=5)
    artifact = MFArtifact.load(model_dir)
    target = preferences["target"].id
    # every fixture movie has 12 ratings, so 13 excludes the catalogue entirely
    well_supported = artifact.movie_counts >= 13

    assert not well_supported.any()
    assert artifact.rank_unseen(target, seen=set(), limit=3, min_support=13) != []
    assert artifact.rank_unseen(target, seen=set(), limit=3, min_support=1) == artifact.rank_unseen(
        target, seen=set(), limit=3, min_support=13
    )


def test_the_support_filter_ranks_well_supported_movies_first(
    preferences, make_movie, make_user, make_rating, train_model, model_dir
):
    # one movie the target's own crowd adores, rated by only two of them
    obscure = make_movie("Obscure Action", ["Action"])
    for fan in preferences["all_users"][:2]:
        make_rating(fan, obscure, 5.0)
    train_model(epochs=40)
    artifact = MFArtifact.load(model_dir)
    target = preferences["target"].id

    assert artifact.movie_counts[artifact.movies.index[obscure.id]] == 2

    ranked = artifact.rank_unseen(target, seen=set(), limit=13, min_support=5)
    well_supported = [
        movie_id
        for movie_id in ranked
        if artifact.movie_counts[artifact.movies.index[movie_id]] >= 5
    ]
    assert ranked[: len(well_supported)] == well_supported  # backfill lands at the end
    assert ranked.index(obscure.id) >= len(well_supported)


def test_load_rejects_an_artifact_without_rating_counts(preferences, train_model, model_dir):
    train_model(epochs=1)
    vocab = json.loads((model_dir / VOCAB_FILE).read_text())
    del vocab["movie_counts"]  # written by a version from before the support filter
    (model_dir / VOCAB_FILE).write_text(json.dumps(vocab))

    assert MFArtifact.load(model_dir) is None


def test_model_learns_the_rating_gap(preferences, train_model, model_dir):
    train_model(epochs=40)
    artifact = MFArtifact.load(model_dir)
    target = preferences["target"].id

    liked = preferences["action"][5].id  # unseen by target, loved by other action fans
    disliked = preferences["romance"][5].id

    ranked = artifact.rank_unseen(target, seen=set(), limit=12)
    assert ranked.index(liked) < ranked.index(disliked)
