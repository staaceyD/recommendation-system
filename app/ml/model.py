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
L2 = 1e-6


def build_model(num_users: int, num_movies: int, dim: int = EMBEDDING_DIM, l2: float = L2):
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
