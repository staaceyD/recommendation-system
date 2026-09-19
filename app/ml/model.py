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

# The biases need `scaled_l2` *and* a strength of their own: they are fitted against a
# mean, not a 32-dim direction, so the same penalty that suits the factors barely moves
# them. Swept at full scale (32M, one split, the same 1000 users):
#
#                    x1    x3    x10   x30   x100  x300  x1000
#   prec@10          .0637 .0712 .0815 .0876 .0932 .0916 .0905
#   documentary-heavy 49.7% 39.7% 14.7%  6.3%  6.7%  9.7%  9.3%
#   recs below 3.0/5    --    --   0.8%  1.9%  4.0%    --    --
#
# NOT the precision argmax, deliberately -- that is x100, and taking it would be the exact
# mistake MIN_SUPPORT warns about below. Two things it cannot see: the documentary symptom
# this was all for bottoms out at x30 and x100 does not improve it, and past x30 the bias
# is shrunk far enough that item quality stops reaching the ranking -- x100 recommends
# movies the catalogue rates under 3.0 at 2.1x the rate of x30 (4.0% against 1.9%) and
# predicts their ratings worse (calibration MAE .604 against .563). prec@10 rewards that,
# because a taste-matched bad film is still one the user rated.
#
# The optimum is interior either way, so this is not "delete the bias": by x1000 the term
# is gone (spread 0.008) and every metric is worse. x30 keeps enough of it to stay honest.
#
# Coverage rises across the whole sweep (0.0121 -> 0.0193), so none of this is precision
# bought by narrowing the catalogue -- that failure would show up here as coverage falling.
#
# Calibrated at 32M. The multiplier is a ratio, so `scaled_l2` should carry it across
# dataset sizes -- but that is an assumption, not a measurement; only 32M was swept.
BIAS_L2_MULTIPLIER = 30.0


def scaled_l2(num_ratings: int) -> float:
    """The `L2_REFERENCE` strength carried over to a dataset of `num_ratings`.

    Training on a `--limit` slice needs this too: the slice is a different `N`, so
    reusing the full-size `l2` over-regularizes it just as badly in reverse.
    """
    if num_ratings <= 0:
        return L2_REFERENCE
    return L2_REFERENCE * L2_REFERENCE_RATINGS / num_ratings


def scaled_bias_l2(num_ratings: int) -> float:
    """The bias strength for a dataset of `num_ratings` -- `BIAS_L2_MULTIPLIER` x the factors'."""
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
    user_in = keras.Input(shape=(), dtype="int64", name="user")
    movie_in = keras.Input(shape=(), dtype="int64", name="movie")

    user_vec = layers.Embedding(num_users, dim, embeddings_regularizer=reg, name="user_embedding")(
        user_in
    )
    movie_vec = layers.Embedding(
        num_movies, dim, embeddings_regularizer=reg, name="movie_embedding"
    )(movie_in)
    # The biases carry the same `l2` as the factors, for the same reason the factors carry
    # it at all. Left unregularized they are the one part of the model that never shrinks
    # toward zero, so a movie's bias reaches its raw mean no matter how little evidence is
    # behind it -- and MovieLens documentaries average as high as Crime on a seventeenth of
    # the ratings (47 per movie against 801). Nothing pulled those biases back, so they
    # outranked every well-supported title for any user whose own vector was small, which is
    # every user with a short rating history. `MIN_SUPPORT` could not catch it either: a
    # 322-rating documentary clears a 100-rating bar easily.
    user_bias = layers.Embedding(
        num_users, 1, embeddings_regularizer=keras.regularizers.l2(bias_l2), name="user_bias"
    )(user_in)
    movie_bias = layers.Embedding(
        num_movies, 1, embeddings_regularizer=keras.regularizers.l2(bias_l2), name="movie_bias"
    )(movie_in)

    dot = layers.Dot(axes=1)([user_vec, movie_vec])
    residual = layers.Add(name="residual")([dot, user_bias, movie_bias])

    return keras.Model(inputs=[user_in, movie_in], outputs=residual, name="matrix_factorization")
