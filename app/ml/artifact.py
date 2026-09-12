from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.ml.vocab import Vocab

MODEL_FILE = "model.keras"
VOCAB_FILE = "vocab.json"
META_FILE = "meta.json"


class MFArtifact:
    """A trained matrix-factorization model plus the id<->index vocabularies."""

    def __init__(self, model, users: Vocab, movies: Vocab, global_mean: float):
        self.model = model
        self.users = users
        self.movies = movies
        self.global_mean = global_mean

    @classmethod
    def load(cls, model_dir: str | Path) -> MFArtifact | None:
        path = Path(model_dir)
        if not all((path / f).exists() for f in (MODEL_FILE, VOCAB_FILE, META_FILE)):
            return None

        import keras

        try:
            model = keras.models.load_model(path / MODEL_FILE)
            vocab = json.loads((path / VOCAB_FILE).read_text())
            meta = json.loads((path / META_FILE).read_text())
            users = Vocab(vocab["user_ids"])
            movies = Vocab(vocab["movie_ids"])
            global_mean = float(meta["global_mean"])
        except (OSError, ValueError, KeyError, TypeError):
            # A save interrupted mid-write (killed process, full disk) leaves a torn
            # artifact -- treat it as absent rather than crashing the request.
            return None

        if not _shapes_agree(model, users, movies):
            return None
        return cls(model, users, movies, global_mean)

    def knows_user(self, user_id: int) -> bool:
        return user_id in self.users

    def rank_unseen(self, user_id: int, seen: set[int], limit: int) -> list[int]:
        """Movie ids with the highest predicted rating for the user, excluding `seen`."""
        if limit <= 0:
            return []

        user_index = self.users.to_index(user_id)
        if user_index is None:
            return []

        movie_indices = np.array(
            [i for i, movie_id in enumerate(self.movies.ids) if movie_id not in seen],
            dtype="int64",
        )
        if movie_indices.size == 0:
            return []

        users = np.full(movie_indices.shape, user_index, dtype="int64")
        residual = np.asarray(
            self.model({"user": users, "movie": movie_indices}, training=False)
        ).reshape(-1)
        predicted = residual + self.global_mean

        limit = min(limit, predicted.size)
        top = np.argpartition(-predicted, limit - 1)[:limit]
        top = top[np.argsort(-predicted[top])]
        return [self.movies.ids[movie_indices[i]] for i in top]


def _shapes_agree(model, users: Vocab, movies: Vocab) -> bool:
    """Reject an artifact whose model and vocabularies come from different runs."""
    try:
        num_users = model.get_layer("user_embedding").input_dim
        num_movies = model.get_layer("movie_embedding").input_dim
    except ValueError:
        return False
    return num_users == len(users) and num_movies == len(movies)
