from __future__ import annotations

import json
from collections.abc import Sequence
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
        self._factors = None

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

        predicted_all = self.scores(user_id)
        if predicted_all is None:
            return []

        movie_indices = np.array(
            [i for i, movie_id in enumerate(self.movies.ids) if movie_id not in seen],
            dtype="int64",
        )
        if movie_indices.size == 0:
            return []

        predicted = predicted_all[movie_indices]
        limit = min(limit, predicted.size)
        top = np.argpartition(-predicted, limit - 1)[:limit]
        top = top[np.argsort(-predicted[top])]
        return [self.movies.ids[movie_indices[i]] for i in top]

    def scores(self, user_id: int) -> np.ndarray | None:
        """Predicted rating for every movie, positionally aligned with `self.movies.ids`."""
        user_index = self.users.to_index(user_id)
        if user_index is None:
            return None
        user_vec, movie_vecs, user_bias, movie_bias = self._get_factors()
        return (
            movie_vecs @ user_vec[user_index]
            + user_bias[user_index]
            + movie_bias
            + self.global_mean
        )

    def predict_pairs(self, user_ids: Sequence[int], movie_ids: Sequence[int]) -> np.ndarray:
        """Predicted ratings for aligned (user_id, movie_id) pairs; NaN where an id is unknown."""
        users = _indices(self.users, user_ids)
        movies = _indices(self.movies, movie_ids)
        user_vec, movie_vecs, user_bias, movie_bias = self._get_factors()

        out = np.full(users.shape, np.nan)
        known = (users >= 0) & (movies >= 0)
        u, m = users[known], movies[known]
        out[known] = (
            (user_vec[u] * movie_vecs[m]).sum(axis=1) + user_bias[u] + movie_bias[m]
        ) + self.global_mean
        return out

    def _get_factors(self):
        """The trained weights as plain arrays -- the same arithmetic the model graph does.

        Scoring the whole catalogue through `model()` one user at a time is a forward
        pass per user; as four matrices it is a single matmul, which is what makes
        full-catalogue evaluation tractable.
        """
        if self._factors is None:
            layer = self.model.get_layer
            self._factors = (
                np.asarray(layer("user_embedding").get_weights()[0], dtype="float64"),
                np.asarray(layer("movie_embedding").get_weights()[0], dtype="float64"),
                np.asarray(layer("user_bias").get_weights()[0], dtype="float64").reshape(-1),
                np.asarray(layer("movie_bias").get_weights()[0], dtype="float64").reshape(-1),
            )
        return self._factors


def _indices(vocab: Vocab, raw_ids: Sequence[int]) -> np.ndarray:
    """Embedding indices for `raw_ids`, with -1 standing in for an unknown id."""
    ids = np.asarray(raw_ids)
    return np.fromiter(
        (vocab.index.get(int(raw), -1) for raw in ids), dtype="int64", count=ids.size
    )


def _shapes_agree(model, users: Vocab, movies: Vocab) -> bool:
    """Reject an artifact whose model and vocabularies come from different runs."""
    try:
        num_users = model.get_layer("user_embedding").input_dim
        num_movies = model.get_layer("movie_embedding").input_dim
    except ValueError:
        return False
    return num_users == len(users) and num_movies == len(movies)
