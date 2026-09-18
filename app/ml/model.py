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

# `embeddings_regularizer` penalises the whole embedding matrix on every step, while
# a row's data gradient arrives only on its own ratings. Per epoch that makes the
# effective strength on a row with `c` ratings scale as `l2 * N / c` for a dataset of
# `N`: the batch size cancels, but the strength grows linearly with the dataset. So a
# fixed `l2` does not mean a fixed amount of regularization -- it means more of it the
# more data you train on, until the embeddings are squeezed flat and the biases are
# left explaining everything. That is a bias-only popularity model wearing embeddings,
# and it hands every user the same list. Measured as the bias-to-personalisation ratio
# at a fixed l2=1e-5: 2.6x at 2M ratings -> 7.8x at 6M -> 14.6x at 12M -> 36.4x at 24M.
#
# So `l2` has to be quoted against a dataset size and scaled from there, which is what
# `scaled_l2` does. Holding `l2 * N` fixed holds the ratio at ~2.5x across all four.
#
# Keep the whole-matrix penalty rather than a per-observation one (`activity_regularizer`).
# It shrinks every row by the same amount per step, so a row with 3 ratings is shrunk far
# harder relative to its evidence than one with 3,000 -- thin rows stay near zero instead
# of predicting extremes and floating to the top of a full-catalogue ranking. A per-sample
# penalty is scale-invariant for free but gives up that support-proportional shrinkage,
# and measured at 6M it fell below the popularity baseline (prec@10 0.038 vs 0.054, and
# 0.003 unfiltered -- barely above random).
L2_REFERENCE = 1e-5
L2_REFERENCE_RATINGS = 1_600_000


def scaled_l2(num_ratings: int) -> float:
    """The `L2_REFERENCE` strength carried over to a dataset of `num_ratings`.

    Training on a `--limit` slice needs this too: the slice is a different `N`, so
    reusing the full-size `l2` over-regularizes it just as badly in reverse.
    """
    if num_ratings <= 0:
        return L2_REFERENCE
    return L2_REFERENCE * L2_REFERENCE_RATINGS / num_ratings


def build_model(
    num_users: int, num_movies: int, dim: int = EMBEDDING_DIM, l2: float = L2_REFERENCE
):
    reg = keras.regularizers.l2(l2)
    user_in = keras.Input(shape=(), dtype="int64", name="user")
    movie_in = keras.Input(shape=(), dtype="int64", name="movie")

    user_vec = layers.Embedding(num_users, dim, embeddings_regularizer=reg, name="user_embedding")(
        user_in
    )
    movie_vec = layers.Embedding(
        num_movies, dim, embeddings_regularizer=reg, name="movie_embedding"
    )(movie_in)
    user_bias = layers.Embedding(num_users, 1, name="user_bias")(user_in)
    movie_bias = layers.Embedding(num_movies, 1, name="movie_bias")(movie_in)

    dot = layers.Dot(axes=1)([user_vec, movie_vec])
    residual = layers.Add(name="residual")([dot, user_bias, movie_bias])

    return keras.Model(inputs=[user_in, movie_in], outputs=residual, name="matrix_factorization")
