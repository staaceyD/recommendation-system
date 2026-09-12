import json

import numpy as np

from app.ml.artifact import META_FILE, MODEL_FILE, VOCAB_FILE, MFArtifact
from app.ml.model import build_model


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


def test_model_learns_the_rating_gap(preferences, train_model, model_dir):
    train_model(epochs=40)
    artifact = MFArtifact.load(model_dir)
    target = preferences["target"].id

    liked = preferences["action"][5].id  # unseen by target, loved by other action fans
    disliked = preferences["romance"][5].id

    ranked = artifact.rank_unseen(target, seen=set(), limit=12)
    assert ranked.index(liked) < ranked.index(disliked)
