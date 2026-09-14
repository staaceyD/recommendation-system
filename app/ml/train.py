"""Train the matrix-factorization recommender on the ratings table.

    uv run python -m app.ml.train                 # full training, saves to instance/mf/
    uv run python -m app.ml.train --limit 2000000  # quick partial run
    uv run python -m app.ml.train --epochs 3 --dim 16

Training is offline: the app never trains on a request. The saved artifact
(`model.keras` + `vocab.json` + `meta.json`) is what `rating_based` loads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.artifact import META_FILE, MODEL_FILE, VOCAB_FILE
from app.ml.data import load_ratings
from app.ml.model import EMBEDDING_DIM, L2, build_model
from app.ml.paths import model_dir
from app.ml.vocab import Vocab

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8192
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_VALIDATION_SPLIT = 0.05
SEED = 42


def train(
    out_dir: str | Path,
    *,
    frame: pd.DataFrame | None = None,
    dim: int = EMBEDDING_DIM,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    l2: float = L2,
    limit: int | None = None,
    verbose: int = 1,
):
    """Fit on `frame`, or on the whole ratings table when it is omitted."""
    import keras

    keras.utils.set_random_seed(SEED)

    if frame is None:
        frame = load_ratings(limit)
    if frame.empty:
        raise SystemExit("no ratings to train on -- seed the database first")

    users = Vocab(sorted(frame["user_id"].unique().tolist()))
    movies = Vocab(sorted(frame["movie_id"].unique().tolist()))

    user_idx = frame["user_id"].map(users.index).to_numpy()
    movie_idx = frame["movie_id"].map(movies.index).to_numpy()
    ratings = frame["rating"].to_numpy(dtype="float32")
    global_mean = float(ratings.mean())

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(ratings))
    user_idx, movie_idx = user_idx[order], movie_idx[order]
    target = ratings[order] - global_mean

    # Saved with the artifact so inference can skip movies with too little evidence
    # behind their embedding -- see MIN_SUPPORT in app/ml/artifact.py.
    counts = frame["movie_id"].value_counts()
    movie_counts = [int(counts[movie_id]) for movie_id in movies.ids]

    model = build_model(len(users), len(movies), dim=dim, l2=l2)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    history = model.fit(
        {"user": user_idx, "movie": movie_idx},
        target,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose,
    )

    _save(out_dir, model, users, movies, movie_counts, global_mean, dim, l2, history, len(ratings))
    return history


def _save(out_dir, model, users, movies, movie_counts, global_mean, dim, l2, history, num_ratings):
    """Write the three artifact files into a staging dir, then swap it into place.

    Writing them straight into `out_dir` would let an interrupted run leave a new
    `model.keras` beside the previous run's `vocab.json` -- an artifact that loads
    but predicts against the wrong ids.
    """
    path = Path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        val_rmse = history.history.get("val_rmse")
        model.save(staging / MODEL_FILE)
        (staging / VOCAB_FILE).write_text(
            json.dumps(
                {
                    "user_ids": users.ids,
                    "movie_ids": movies.ids,
                    "movie_counts": movie_counts,
                }
            )
        )
        (staging / META_FILE).write_text(
            json.dumps(
                {
                    "dim": dim,
                    "l2": l2,
                    "global_mean": global_mean,
                    "num_users": len(users),
                    "num_movies": len(movies),
                    "num_ratings": num_ratings,
                    "final_rmse": float(history.history["rmse"][-1]),
                    "final_val_rmse": float(val_rmse[-1]) if val_rmse else None,
                    "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _swap(staging, path)


def _swap(staging: Path, path: Path) -> None:
    """Move the staged artifact onto `path`, keeping the old one until it lands."""
    backup = path.parent / f".{path.name}-previous"
    shutil.rmtree(backup, ignore_errors=True)
    if path.exists():
        os.replace(path, backup)
    try:
        os.replace(staging, path)
    except OSError:
        if backup.exists():
            os.replace(backup, path)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output dir (default: instance/mf/)")
    parser.add_argument("--dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--l2", type=float, default=L2, help="embedding regularization strength")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    args = parser.parse_args(argv)

    from app import create_app

    app = create_app()
    with app.app_context():
        out_dir = args.out or model_dir()
        print(f"Training -> {out_dir}")
        history = train(
            out_dir,
            dim=args.dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2=args.l2,
            limit=args.limit,
        )
    print(f"Done. final RMSE {history.history['rmse'][-1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
