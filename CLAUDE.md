# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Movie recommender web app: Flask backend, MySQL database, seeded from the MovieLens dataset.
Recommendation strategy is hybrid:
- User has existing ratings → recommend from that rating history.
- User has no ratings (cold start) → fall back to collaborative filtering over the full ratings
  matrix.

Full plan and roadmap live in `README.md` — treat it as the source of truth for scope and
build order, and keep its checklist up to date as work lands.

**Status:** DB scaffolding exists: `app/` (app factory, config, extensions) and SQLAlchemy models
(`User`, `Movie`, `Genre`, `Rating`, plus the `movie_genres` join table) with a first Alembic
migration applied to local MySQL (`recsys` and `recsys_test` databases, empty tables). No routes,
auth, seed script, or tests have been written yet — don't assume any other planned structure in the
README exists until you check.

## Commands

This project uses **uv** exclusively — no bare `pip`/`venv`/`python -m venv`.

```bash
uv sync                        # install/update deps into .venv from pyproject.toml + uv.lock
uv run flask run               # run the dev server
uv run flask db upgrade        # apply migrations
uv run python data/seed.py     # seed the DB
uv run pytest                  # run tests
uv run pytest --cov=app        # run tests with coverage
uv add <package>                # add a runtime dependency
uv add --dev <package>          # add a dev-only dependency
```

`uv add`/`uv add --dev` update both `pyproject.toml` and `uv.lock` — always commit both together,
never hand-edit `uv.lock`.

MySQL runs locally via Docker Compose (once `docker-compose.yml` exists) or an existing local
instance, configured through `DATABASE_URL` in `.env` (see `README.md`).

## Conventions

- Config via environment variables / `.env` (python-dotenv), never hardcoded credentials.
- Two independent recommendation services are expected: a rating-history-based one and a
  collaborative-filtering one, dispatched from a single `/recommendations` endpoint based on
  whether the requesting user has ratings. Keep that dispatch logic thin — the branching belongs
  in one place, not scattered across routes.
- Tests run against a separate test DB/config, never the dev database.
