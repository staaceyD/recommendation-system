"""The recommender model: classic matrix factorization in Keras.

Each user and each movie gets a learned vector (embedding) plus a scalar bias.
The predicted deviation from the global mean rating is

    user_vec · movie_vec + user_bias + movie_bias

Trained with MSE against `rating - global_mean`, so at inference the estimated
rating is `global_mean + model(user, movie)`.
"""

from __future__ import annotations

import keras
from keras import layers

EMBEDDING_DIM = 32

# Neither constant is free to change on taste -- both were measured, and both have a
# failure mode that RMSE cannot see. See docs/model-tuning.md before touching them.
#
# `l2` is quoted against a dataset size because `embeddings_regularizer` penalises the
# whole matrix every step: a fixed `l2` regularizes harder the more data you train on.
L2_REFERENCE = 1e-5
L2_REFERENCE_RATINGS = 1_600_000

# The biases need a strength of their own -- fitted against a mean rather than a 32-dim
# direction, the factors' penalty barely moves them. Deliberately not the precision
# argmax (that is 100x); past 30x the bias stops reaching the ranking at all.
BIAS_L2_MULTIPLIER = 30.0


def scaled_l2(num_ratings: int) -> float:
    """The `L2_REFERENCE` strength carried over to a dataset of `num_ratings`.

    A `--limit` slice needs this too: reusing the full-size `l2` on a smaller `N`
    over-regularizes it just as badly in reverse.
    """
    if num_ratings <= 0:
        return L2_REFERENCE
    return L2_REFERENCE * L2_REFERENCE_RATINGS / num_ratings


def scaled_bias_l2(num_ratings: int) -> float:
    return scaled_l2(num_ratings) * BIAS_L2_MULTIPLIER


def build_model(
    num_users: int,
    num_movies: int,
    dim: int = EMBEDDING_DIM,
    l2: float = L2_REFERENCE,
    bias_l2: float | None = None,
):
    if bias_l2 is None:
        bias_l2 = l2 * BIAS_L2_MULTIPLIER
    reg = keras.regularizers.l2(l2)
    bias_reg = keras.regularizers.l2(bias_l2)
    user_in = keras.Input(shape=(), dtype="int64", name="user")
    movie_in = keras.Input(shape=(), dtype="int64", name="movie")

    user_vec = layers.Embedding(num_users, dim, embeddings_regularizer=reg, name="user_embedding")(
        user_in
    )
    movie_vec = layers.Embedding(
        num_movies, dim, embeddings_regularizer=reg, name="movie_embedding"
    )(movie_in)
    user_bias = layers.Embedding(num_users, 1, embeddings_regularizer=bias_reg, name="user_bias")(
        user_in
    )
    movie_bias = layers.Embedding(
        num_movies, 1, embeddings_regularizer=bias_reg, name="movie_bias"
    )(movie_in)

    dot = layers.Dot(axes=1)([user_vec, movie_vec])
    residual = layers.Add(name="residual")([dot, user_bias, movie_bias])

    return keras.Model(inputs=[user_in, movie_in], outputs=residual, name="matrix_factorization")
