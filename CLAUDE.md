# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Movie recommender web app: Flask backend, SQLite database (a single file, `instance/recsys.db` —
no server), seeded from the MovieLens dataset.
Recommendation strategy is hybrid:
- User has existing ratings → recommend from that rating history.
- User has no ratings (cold start) → fall back to collaborative filtering over the full ratings
  matrix.

Full plan and roadmap live in `README.md` — treat it as the source of truth for scope and
build order, and keep its checklist up to date as work lands.

**Status:** DB scaffolding exists: `app/` (app factory, config, extensions) and SQLAlchemy models
(`User`, `Movie`, `Genre`, `Rating`, plus the `movie_genres` join table) with the first Alembic
migration applied to `instance/recsys.db`. Tests run against an in-memory SQLite DB. The MovieLens
`ml-32m` dataset is downloaded locally into `data/raw/` (gitignored, never committed);
`data/preprocess.py`
converts the raw CSVs into the shape `data/seed.py` loads, and the seed script parses those and
loads genres/movies/users/ratings — MovieLens `movieId`/`userId` are reused as primary keys, users
are placeholder rows built from the ids in `ratings.csv`. No routes, auth, or tests have been
written yet — don't assume any other planned structure in the README exists until you check.

## Commands

This project uses **uv** exclusively — no bare `pip`/`venv`/`python -m venv`.

```bash
uv sync                        # install/update deps into .venv from pyproject.toml + uv.lock
uv run flask run               # run the dev server
uv run flask db upgrade        # apply migrations
uv run python data/seed.py     # seed the DB
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
- Two independent recommendation services are expected: a rating-history-based one and a
  collaborative-filtering one, dispatched from a single `/recommendations` endpoint based on
  whether the requesting user has ratings. Keep that dispatch logic thin — the branching belongs
  in one place, not scattered across routes.
- Tests run against a separate test DB/config, never the dev database. Tests that seed must load
  only a handful of rows (tiny fixture CSVs), never the full dataset.
- Keep comments and docstrings minimal — only what isn't obvious from the code. One-line docstrings
  where a docstring earns its place; skip them on self-explanatory helpers and tests.
