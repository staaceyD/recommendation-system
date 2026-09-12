"""Recommendations for users the trained model has learned an embedding for.

Loads the matrix-factorization artifact once per process and ranks unseen movies
by predicted rating. Returns `[]` when there is no model or the user is unknown
to it -- the dispatcher then falls back to `collaborative`.
"""

from __future__ import annotations

import threading

from app.extensions import db
from app.models import Rating

DEFAULT_LIMIT = 20

_artifact = None
_loaded = False
_lock = threading.Lock()


def recommend(user_id: int, limit: int = DEFAULT_LIMIT) -> list[int]:
    artifact = _get_artifact()
    if artifact is None or not artifact.knows_user(user_id):
        return []

    seen = set(
        db.session.execute(db.select(Rating.movie_id).where(Rating.user_id == user_id)).scalars()
    )
    return artifact.rank_unseen(user_id, seen, limit)


def _get_artifact():
    global _artifact, _loaded
    if not _loaded:
        with _lock:
            if not _loaded:  # another thread may have loaded it while we waited
                from app.ml.artifact import MFArtifact
                from app.ml.paths import model_dir

                _artifact = MFArtifact.load(model_dir())
                _loaded = True
    return _artifact


def reset() -> None:
    """Drop the cached artifact (tests, or after retraining)."""
    global _artifact, _loaded
    _artifact, _loaded = None, False
