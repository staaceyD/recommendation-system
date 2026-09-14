import json

import numpy as np
import pytest

from app.ml.artifact import META_FILE, MODEL_FILE, VOCAB_FILE, MFArtifact
from app.ml.model import build_model
from app.ml.train import _save
from app.ml.vocab import Vocab


def test_build_model_scores_one_value_per_pair():
    model = build_model(num_users=4, num_movies=6, dim=3)
    out = model(
        {"user": np.array([0, 1, 2], dtype="int64"), "movie": np.array([5, 4, 3], dtype="int64")},
        training=False,
    )
    assert tuple(out.shape) == (3, 1)


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
        _save(model_dir, Exploding(), Vocab([1]), Vocab([2]), 3.5, 8, FakeHistory(), 1)

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


def test_model_learns_the_rating_gap(preferences, train_model, model_dir):
    train_model(epochs=40)
    artifact = MFArtifact.load(model_dir)
    target = preferences["target"].id

    liked = preferences["action"][5].id  # unseen by target, loved by other action fans
    disliked = preferences["romance"][5].id

    ranked = artifact.rank_unseen(target, seen=set(), limit=12)
    assert ranked.index(liked) < ranked.index(disliked)
