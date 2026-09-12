from __future__ import annotations

import pandas as pd

from app.extensions import db


def load_ratings(limit: int | None = None) -> pd.DataFrame:
    """All ratings as a DataFrame with columns ``user_id``, ``movie_id``, ``rating``."""
    sql = "SELECT user_id, movie_id, rating FROM ratings"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql_query(sql, db.engine)
