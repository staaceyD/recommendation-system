"""Offline evaluation: how good are the recommendations, really?

    uv run python -m app.ml.evaluate                  # evaluate on the whole ratings table
    uv run python -m app.ml.evaluate --limit 2000000  # quick run on a slice
    uv run python -m app.ml.evaluate --k 20 --users 2000 --json report.json

Training RMSE says how well the model predicts a rating it was fitted on. It does
not say whether the *list* the user sees is any good, so this splits the ratings
per user (most of a user's ratings train, the rest are held out and never seen by
the model), asks each recommender for a top-K list, and checks how much of the
holdout it recovered.

Scores like "precision@10 = 0.11" mean nothing on their own, so the same users and
the same holdout are scored for two baselines as well: `popularity`, which is the
cold-start ranking the app already falls back to, and `random`. A model that cannot
beat `popularity` is not earning the embeddings it costs.

This always trains its own model on the training split. It deliberately cannot
evaluate `instance/mf/`: that artifact was fitted on every rating in the table, so
every test row would already be memorised and every score would come out flattering.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml import metrics
from app.ml.artifact import MFArtifact
from app.ml.data import load_ratings
from app.ml.model import EMBEDDING_DIM
from app.ml.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    SEED,
    train,
)
from app.services.collaborative import LIKED_RATING, MIN_RATINGS, PRIOR_STRENGTH, bayesian_score

DEFAULT_K = 10
DEFAULT_HOLDOUT = 0.2
DEFAULT_EVAL_USERS = 1000

# A user needs enough ratings that holding some back still leaves a taste to learn.
MIN_USER_RATINGS = 5


def split_ratings(
    frame: pd.DataFrame,
    *,
    holdout: float = DEFAULT_HOLDOUT,
    min_ratings: int = MIN_USER_RATINGS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a random `holdout` share of each user's ratings.

    Per user, not per row: a global split would leave most test users with no
    training history at all, which measures cold-start rather than the model.
    Users below `min_ratings` keep all of their ratings in the training half.

    MovieLens timestamps are dropped in preprocessing, so this is a random split
    rather than "predict what they watched next" -- it slightly flatters any
    recommender, since some holdout ratings are older than the ones it trained on.
    """
    shuffled = frame.sample(frac=1.0, random_state=seed).sort_values("user_id", kind="stable")
    sizes = shuffled.groupby("user_id")["rating"].transform("size").to_numpy()
    position = shuffled.groupby("user_id").cumcount().to_numpy()

    held = np.where(sizes >= min_ratings, np.maximum(1, np.rint(sizes * holdout)), 0)
    held = np.minimum(held, sizes - 1)  # never empty a user's training history
    is_test = position < held

    train_frame = shuffled[~is_test].reset_index(drop=True)
    test_frame = shuffled[is_test]
    # A movie seen only in the holdout has no embedding, so no recommender could
    # ever return it -- scoring against it would just be noise in every column.
    test_frame = test_frame[test_frame["movie_id"].isin(train_frame["movie_id"].unique())]
    return train_frame, test_frame.reset_index(drop=True)


class ModelRecommender:
    """The served path: `rating_based` ranking every unseen movie by predicted rating."""

    name = "matrix-factorization"

    def __init__(self, artifact: MFArtifact):
        self.artifact = artifact

    def recommend(self, user_id: int, seen: set[int], k: int) -> list[int]:
        return self.artifact.rank_unseen(user_id, seen, k)


class PopularityRecommender:
    """The cold-start fallback's ranking, rebuilt from the training split only.

    `collaborative.recommend` also prefers genres the user already likes; this drops
    that, so treat it as a floor for the fallback rather than a measurement of it.
    """

    name = "popularity"

    def __init__(self, train_frame: pd.DataFrame):
        stats = train_frame.groupby("movie_id")["rating"].agg(["count", "mean"])
        stats = stats[stats["count"] >= MIN_RATINGS]
        overall_mean = (
            (stats["count"] * stats["mean"]).sum() / stats["count"].sum() if len(stats) else 0.0
        )
        scores = bayesian_score(stats["count"], stats["mean"], overall_mean, PRIOR_STRENGTH)
        self.ranked = scores.sort_values(ascending=False).index.tolist()

    def recommend(self, user_id: int, seen: set[int], k: int) -> list[int]:
        out = []
        for movie_id in self.ranked:
            if movie_id not in seen:
                out.append(movie_id)
                if len(out) == k:
                    break
        return out


class RandomRecommender:
    """The sanity floor. Anything that cannot beat this is broken, not merely weak."""

    name = "random"

    def __init__(self, catalogue: list[int], seed: int = SEED):
        self.catalogue = np.asarray(catalogue)
        self.rng = np.random.default_rng(seed)

    def recommend(self, user_id: int, seen: set[int], k: int) -> list[int]:
        size = min(len(self.catalogue), k + len(seen))
        picks = self.rng.choice(self.catalogue, size=size, replace=False)
        return [int(movie_id) for movie_id in picks if movie_id not in seen][:k]


@dataclass
class Report:
    strategy: str
    precision: float
    recall: float
    ndcg: float
    map: float
    hit_rate: float
    coverage: float
    novelty: float


def score_recommender(
    recommender,
    cases: list[tuple[int, set[int], set[int]]],
    *,
    k: int,
    popularity: dict[int, int],
    total_ratings: int,
    catalogue_size: int,
) -> Report:
    """Average the top-K metrics over `cases` of (user_id, seen, relevant)."""
    totals = dict.fromkeys(("precision", "recall", "ndcg", "map", "hit_rate"), 0.0)
    recommended_anywhere: set[int] = set()
    novelties = []

    for user_id, seen, relevant in cases:
        ranked = recommender.recommend(user_id, seen, k)
        totals["precision"] += metrics.precision_at_k(ranked, relevant, k)
        totals["recall"] += metrics.recall_at_k(ranked, relevant, k)
        totals["ndcg"] += metrics.ndcg_at_k(ranked, relevant, k)
        totals["map"] += metrics.average_precision_at_k(ranked, relevant, k)
        totals["hit_rate"] += metrics.hit_rate_at_k(ranked, relevant, k)
        recommended_anywhere.update(ranked)
        novelties.append(metrics.novelty(ranked, popularity, total_ratings))

    n = len(cases)
    return Report(
        strategy=recommender.name,
        **{name: value / n for name, value in totals.items()},
        coverage=metrics.coverage(recommended_anywhere, catalogue_size),
        novelty=float(np.mean(novelties)) if novelties else 0.0,
    )


def build_cases(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    num_users: int,
    seed: int = SEED,
) -> list[tuple[int, set[int], set[int]]]:
    """Sample users and collect, for each, what they had seen and what they liked in the holdout.

    Every strategy is scored over this one list, so the comparison between them is
    not clouded by them having been asked about different users.
    """
    liked = test_frame[test_frame["rating"] >= LIKED_RATING]
    candidates = liked["user_id"].unique()
    if len(candidates) > num_users:
        candidates = np.random.default_rng(seed).choice(candidates, num_users, replace=False)

    chosen = set(candidates.tolist())
    relevant_by_user = liked[liked["user_id"].isin(chosen)].groupby("user_id")["movie_id"].agg(set)
    seen_slice = train_frame[train_frame["user_id"].isin(chosen)]
    seen_by_user = seen_slice.groupby("user_id")["movie_id"].agg(set)

    return [
        (int(user_id), seen_by_user.get(user_id, set()), relevant)
        for user_id, relevant in relevant_by_user.items()
    ]


def rating_accuracy(artifact: MFArtifact, test_frame: pd.DataFrame) -> dict:
    """Holdout RMSE, next to the RMSE of just predicting the global mean every time."""
    predicted = artifact.predict_pairs(test_frame["user_id"], test_frame["movie_id"])
    known = ~np.isnan(predicted)
    actual = test_frame["rating"].to_numpy()[known]
    return {
        "rmse": _rmse(predicted[known] - actual),
        "baseline_rmse": _rmse(artifact.global_mean - actual),
        "rated": int(known.sum()),
    }


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors)))) if errors.size else float("nan")


def run(
    *,
    k: int = DEFAULT_K,
    num_users: int = DEFAULT_EVAL_USERS,
    holdout: float = DEFAULT_HOLDOUT,
    limit: int | None = None,
    frame: pd.DataFrame | None = None,
    seed: int = SEED,
    verbose: int = 1,
    **train_kwargs,
) -> dict:
    """Split, train on the training half, and score every strategy on the holdout."""
    if frame is None:
        frame = load_ratings(limit)
    if frame.empty:
        raise SystemExit("no ratings to evaluate -- seed the database first")

    train_frame, test_frame = split_ratings(frame, holdout=holdout, seed=seed)
    if test_frame.empty:
        raise SystemExit(
            f"nothing held out -- every user has fewer than {MIN_USER_RATINGS} ratings"
        )

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "mf"
        train(
            out_dir,
            frame=train_frame,
            validation_split=0.0,
            verbose=verbose,
            **train_kwargs,
        )
        artifact = MFArtifact.load(out_dir)
        if artifact is None:
            raise SystemExit("training produced no loadable artifact")

        cases = build_cases(train_frame, test_frame, num_users=num_users, seed=seed)
        if not cases:
            raise SystemExit(f"no held-out ratings at or above {LIKED_RATING} to score against")

        counts = train_frame["movie_id"].value_counts()
        popularity = counts.to_dict()
        shared = {
            "k": k,
            "popularity": popularity,
            "total_ratings": int(counts.sum()),
            "catalogue_size": len(artifact.movies),
        }
        reports = [
            score_recommender(ModelRecommender(artifact), cases, **shared),
            score_recommender(PopularityRecommender(train_frame), cases, **shared),
            score_recommender(RandomRecommender(artifact.movies.ids, seed), cases, **shared),
        ]
        accuracy = rating_accuracy(artifact, test_frame)

    return {
        "k": k,
        "users_scored": len(cases),
        "holdout": holdout,
        "liked_threshold": LIKED_RATING,
        "train_ratings": len(train_frame),
        "test_ratings": len(test_frame),
        "catalogue_size": len(artifact.movies),
        "rating_accuracy": accuracy,
        "strategies": [asdict(report) for report in reports],
    }


COLUMNS = [
    ("precision", "prec@K"),
    ("recall", "recall@K"),
    ("ndcg", "ndcg@K"),
    ("map", "map@K"),
    ("hit_rate", "hit@K"),
    ("coverage", "coverage"),
    ("novelty", "novelty"),
]


def format_report(result: dict) -> str:
    k = result["k"]
    width = max(len(row["strategy"]) for row in result["strategies"])
    header = f"{'strategy':<{width}}" + "".join(
        f"  {label.replace('K', str(k)):>10}" for _, label in COLUMNS
    )
    lines = [
        f"Ranking quality @{k} -- {result['users_scored']:,} users, "
        f"{result['holdout']:.0%} of each user's ratings held out, "
        f"relevant = rated >= {result['liked_threshold']}",
        "",
        header,
        "-" * len(header),
    ]
    for row in result["strategies"]:
        cells = "".join(f"  {row[key]:>10.4f}" for key, _ in COLUMNS)
        lines.append(f"{row['strategy']:<{width}}{cells}")

    accuracy = result["rating_accuracy"]
    lines += [
        "",
        f"Rating prediction on the holdout: RMSE {accuracy['rmse']:.4f} "
        f"(predicting the global mean every time: {accuracy['baseline_rmse']:.4f})",
        f"Trained on {result['train_ratings']:,} ratings, "
        f"scored against {result['test_ratings']:,} held out, "
        f"catalogue of {result['catalogue_size']:,} movies.",
        "",
        "prec/recall/ndcg/map/hit: higher is better. coverage: share of the catalogue ever",
        "recommended. novelty: bits of surprise, where a low number means blockbusters.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="list length to score")
    parser.add_argument("--users", type=int, default=DEFAULT_EVAL_USERS, metavar="N")
    parser.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT, metavar="FRACTION")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--json", default=None, metavar="PATH", help="also write the raw numbers")
    args = parser.parse_args(argv)

    from app import create_app

    app = create_app()
    with app.app_context():
        result = run(
            k=args.k,
            num_users=args.users,
            holdout=args.holdout,
            limit=args.limit,
            epochs=args.epochs,
            dim=args.dim,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    print(format_report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
