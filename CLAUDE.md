# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Movie recommender web app: Flask backend, SQLite database (a single file, `instance/recsys.db` —
no server), seeded from the MovieLens dataset.
Recommendation strategy is hybrid:
- User is in the trained model → recommend from their learned embedding (TensorFlow
  matrix-factorization model, trained offline by `app/ml/train.py`).
- User is unknown to the model (new / not in training) → `collaborative` cold-start fallback
  (Bayesian-adjusted popularity, genre-aware).

Full plan and roadmap live in `README.md` — treat it as the source of truth for scope and
build order, and keep its checklist up to date as work lands.

**Status:** working vertical slice. App factory + config + SQLAlchemy models (`User`, `Movie`,
`Genre`, `Rating`, `movie_genres`) with Alembic migrations against `instance/recsys.db`; tests run
against in-memory SQLite. Endpoints: `GET /movies`, `POST /movies/<id>/rate`, `GET /recommendations`.
`app/ml/` is the Keras matrix-factorization model + offline training script; the trained artifact
lives in `instance/mf/` (gitignored). `app/services/{rating_based,collaborative}.py` are the two
recommenders, dispatched thinly from the `/recommendations` route. MovieLens `movieId`/`userId` are
reused as primary keys; users are placeholder rows from `ratings.csv`. No auth.

## Commands

This project uses **uv** exclusively — no bare `pip`/`venv`/`python -m venv`.

```bash
uv sync                        # install/update deps into .venv from pyproject.toml + uv.lock
uv run flask run               # run the dev server
uv run flask db upgrade        # apply migrations
uv run python data/seed.py     # seed the DB
uv run python -m app.ml.train   # train the recommender model -> instance/mf/
uv run python -m app.ml.evaluate  # score the recommenders on a held-out split
uv run pytest                  # run tests
uv run pytest --cov=app        # run tests with coverage
uv run ruff check .            # lint (CI gate)
uv run ruff format --check .   # formatting check (CI gate)
uv add <package>                # add a runtime dependency
uv add --dev <package>          # add a dev-only dependency
```

`uv add`/`uv add --dev` update both `pyproject.toml` and `uv.lock` — always commit both together,
never hand-edit `uv.lock`.

The database is a local SQLite file, configured through `DATABASE_URL` in `.env` (defaults to
`sqlite:///recsys.db`, which Flask resolves to `instance/recsys.db`). No server, no Docker.

## Conventions

- Config via environment variables / `.env` (python-dotenv), never hardcoded credentials.
- Two independent recommendation services: `rating_based` (trained-model inference) and
  `collaborative` (cold-start). Dispatched thinly from the `/recommendations` route — try the
  model, fall back to cold-start. Keep that branching in the one place, not scattered across routes.
- Training is offline only. The app loads a saved artifact from `instance/mf/`; it never trains
  on a request. `MODEL_DIR` env var overrides the artifact location (tests point it at a tmp dir).
- Evaluation (`app/ml/evaluate.py`) trains its own model on a held-out split and never scores
  `instance/mf/` — that artifact saw every rating, so any holdout is already memorised. Report
  ranking metrics against the popularity and random baselines, never on their own.
- Don't tune `MIN_SUPPORT` or `L2` by maximising precision@K: offline top-N rewards narrowing the
  catalogue, and the limit of that is recommending only blockbusters — i.e. becoming the baseline.
  Both are set on statistical grounds and justified in the README; check `coverage` alongside.
- Tests run against a separate test DB/config, never the dev database. Tests that seed must load
  only a handful of rows (tiny fixture CSVs), never the full dataset. Model tests train a tiny
  model on fixture data — keep them small (few epochs, `dim` ~8).
- Keep comments and docstrings minimal — only what isn't obvious from the code. One-line docstrings
  where a docstring earns its place; skip them on self-explanatory helpers and tests.
