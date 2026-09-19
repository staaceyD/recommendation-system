from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.ml.vocab import Vocab

MODEL_FILE = "model.keras"
VOCAB_FILE = "vocab.json"
META_FILE = "meta.json"

# Don't recommend a movie whose embedding was fitted from fewer ratings than this.
# A movie costs `dim + 1` parameters (33 by default), so a handful of ratings leaves
# them badly underdetermined -- and an overfitted embedding predicts extreme ratings,
# which is exactly what floats a movie to the top of a full-catalogue ranking.
# This filter used to be load-bearing: on a 6M slice it was precision@10 0.085 filtered
# against 0.062 unfiltered. Most of that gap was it covering for unregularized biases --
# once those carry L2 too (see app/ml/model.py) the same slice is 0.100 against 0.098, so
# the filter is now worth a rounding error rather than a third of the score. Keep it: an
# embedding still needs more evidence than an average does, which is why the cold-start
# ranker refuses movies under 50 ratings. But it is no longer propping anything up, and
# the value of 100 has never been derived -- it was picked against a model with flattened
# embeddings and has only ever been re-checked since.
MIN_SUPPORT = 100


class MFArtifact:
    """A trained matrix-factorization model plus the id<->index vocabularies."""

    def __init__(
        self,
        model,
        users: Vocab,
        movies: Vocab,
        movie_counts: np.ndarray,
        global_mean: float,
    ):
        self.model = model
        self.users = users
        self.movies = movies
        self.movie_counts = movie_counts
        self.global_mean = global_mean
        self._factors = None
        self._eligible_cache: tuple[int, np.ndarray] | None = None

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
            # Absent in artifacts written before the support filter -- retrain.
            movie_counts = np.asarray(vocab["movie_counts"], dtype="int64")
            global_mean = float(meta["global_mean"])
        except (OSError, ValueError, KeyError, TypeError):
            # A save interrupted mid-write (killed process, full disk) leaves a torn
            # artifact -- treat it as absent rather than crashing the request.
            return None

        if not _shapes_agree(model, users, movies) or movie_counts.shape != (len(movies),):
            return None
        return cls(model, users, movies, movie_counts, global_mean)

    def knows_user(self, user_id: int) -> bool:
        return user_id in self.users

    def rank_unseen(
        self, user_id: int, seen: set[int], limit: int, min_support: int = MIN_SUPPORT
    ) -> list[int]:
        """Movie ids with the highest predicted rating for the user, excluding `seen`.

        Movies rated fewer than `min_support` times in training are held back, and
        only used to top the list up if too few well-supported ones are left --
        the same shape as the cold-start ranker backfilling past its genre filter.
        """
        if limit <= 0:
            return []

        predicted = self.scores(user_id)
        if predicted is None:
            return []

        unseen = np.ones(len(self.movies), dtype=bool)
        for movie_id in seen:
            index = self.movies.index.get(movie_id)
            if index is not None:
                unseen[index] = False

        eligible = unseen & self._eligible(min_support)
        picks = self._top(predicted, eligible, limit)
        if len(picks) < limit:
            picks += self._top(predicted, unseen & ~eligible, limit - len(picks))
        return picks

    def _top(self, predicted: np.ndarray, mask: np.ndarray, limit: int) -> list[int]:
        candidates = np.flatnonzero(mask)
        if candidates.size == 0 or limit <= 0:
            return []

        scores = predicted[candidates]
        limit = min(limit, scores.size)
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [self.movies.ids[candidates[i]] for i in top]

    def _eligible(self, min_support: int) -> np.ndarray:
        """Which movies have enough ratings behind their embedding to be trusted."""
        if self._eligible_cache is None or self._eligible_cache[0] != min_support:
            self._eligible_cache = (min_support, self.movie_counts >= min_support)
        return self._eligible_cache[1]

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
