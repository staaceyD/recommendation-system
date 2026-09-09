# Recommendation System

A movie recommender web app built with **Flask** and **SQLite**. The database is seeded with a
public movie ratings dataset ([MovieLens](https://grouplens.org/datasets/movielens/)). For any
given user:

- If the user has **rated movies already**, recommendations are generated from that rating
  history (similar movies / similar users based on what they've rated).
- If the user is **new and has no ratings yet**, recommendations fall back to **collaborative
  filtering** over the full ratings matrix (e.g. popularity among similar user cohorts) to give a
  reasonable cold-start experience.

Every endpoint receives a `user_id` from the caller — this app does not manage authentication or
create users. Users are the placeholder rows loaded by the seed script from the MovieLens dataset.

## Tech stack

- **Backend:** Python 3.11+, Flask
- **Database:** SQLite (a single file, `instance/recsys.db` — no server to run)
- **ORM / migrations:** Flask-SQLAlchemy, Flask-Migrate (Alembic)
- **ML / data:** pandas, numpy, scikit-learn (and/or `implicit` / `surprise` for matrix
  factorization CF)
- **Testing:** pytest, pytest-cov
- **Config:** python-dotenv
- **Env / package management:** [uv](https://docs.astral.sh/uv/)

## Project structure (planned)

```
recommendation-system/
├── app/
│   ├── __init__.py            # app factory
│   ├── config.py              # config classes (Dev/Test/Prod)
│   ├── extensions.py          # db, migrate instances
│   ├── models/                # SQLAlchemy models: User, Movie, Rating
│   ├── routes/                # Flask blueprints (movies, ratings, recommendations)
│   └── services/
│       ├── rating_based.py    # recs from a user's own rating history
│       └── collaborative.py   # collaborative filtering for cold-start users
├── data/
│   ├── raw/                   # MovieLens CSVs: ml-32m/ source + preprocessed output (gitignored)
│   ├── preprocess.py          # raw ml-32m CSVs -> the shape seed.py loads
│   └── seed.py                # loads raw/ into SQLite
├── instance/                   # recsys.db lives here (gitignored)
├── migrations/                 # Alembic migrations
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_routes.py
│   └── test_recommenders.py
├── .env.example
├── pyproject.toml              # deps + uv config
├── uv.lock                     # uv lockfile (committed)
├── run.py                      # entrypoint
└── README.md
```

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (manages the Python version, the
  virtual environment, and dependencies — no manual `venv`/`pip` needed)

That's it — SQLite ships with Python, so there's no database server to install or run.

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd recommendation-system
uv sync
```

`uv sync` creates a `.venv/`, installs the correct Python version if needed, and installs all
dependencies (including dev/test tools) from `pyproject.toml` / `uv.lock`. You don't need to
activate the venv — prefix commands with `uv run` (as shown below), or run
`source .venv/bin/activate` if you prefer.

### 2. Configure environment variables

```bash
cp .env.example .env
```

```
FLASK_APP=run.py
FLASK_ENV=development
DATABASE_URL=sqlite:///recsys.db
```

`sqlite:///recsys.db` is a relative path, so Flask resolves it to `instance/recsys.db`. Use an
absolute path (`sqlite:////abs/path/recsys.db`) to put the file elsewhere.

### 3. Run migrations

```bash
uv run flask db upgrade
```

### 4. Get the dataset

The MovieLens data is **not** committed to the repo (everything under `data/raw/`
is gitignored). Download [`ml-32m.zip`](https://grouplens.org/datasets/movielens/)
and unzip it into `data/raw/` so the files land at `data/raw/ml-32m/`, then
convert them to the shape the seed script loads:

```bash
uv run python data/preprocess.py
```

This writes `data/raw/movies.csv` (`movieId,title,genre`, one row per movie/genre
pair) and `data/raw/ratings.csv` (`userId,movieId,rating`).

### 5. Seed the database

Reads `data/raw/movies.csv` and `data/raw/ratings.csv` and loads
genres/movies/users/ratings into SQLite (the full seed produces a ~1.2 GB `instance/recsys.db`):

```bash
uv run python data/seed.py                    # full seed (~32M ratings, a few minutes)
uv run python data/seed.py --reset            # wipe the tables first, then re-seed
uv run python data/seed.py --ratings-limit N  # only load the first N ratings (quick partial seed)
```

Users are synthesised from the ids in `ratings.csv` as placeholder rows (no
credentials). The script targets whichever database the current `FLASK_ENV`
resolves to, so it won't touch the test database unless asked.

### 6. Run the app

```bash
uv run flask run
```

The API will be available at `http://localhost:5000`.

### Managing dependencies

Add a new runtime dependency:

```bash
uv add <package>
```

Add a dev-only dependency (e.g. a test tool):

```bash
uv add --dev <package>
```

This updates `pyproject.toml` and `uv.lock` — commit both.

## Running tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=app
```

Tests run against a separate config (`FLASK_ENV=testing`) backed by an in-memory SQLite database
(`TEST_DATABASE_URL=sqlite://`), so they never touch `instance/recsys.db`.

## Linting

```bash
uv run ruff check .        # lint
uv run ruff format .       # auto-format
uv run ruff format --check .   # verify formatting (what CI runs)
```

## CI

`.github/workflows/ci.yml` runs on every push and pull request against `main`, in two jobs:

- **lint** — `ruff check` + `ruff format --check`
- **test** — `pytest --cov=app` against an in-memory SQLite database

Merges into `main` should be gated on both jobs passing. That gate is repository configuration,
not code: in GitHub go to **Settings → Branches → Add branch ruleset** (or **Branch protection
rules**) for `main`, enable **Require status checks to pass before merging**, and select the
`lint` and `test` checks (they appear in the list after the workflow has run once).

## API endpoints (planned)

| Method | Endpoint                        | Description                                          |
|--------|---------------------------------|-----------------------------------------------------|
| GET    | `/movies`                       | List / search movies                                |
| POST   | `/movies/<id>/rate`             | Rate a movie (body includes `user_id`)              |
| GET    | `/recommendations?user_id=<id>` | Get recommendations for the given user               |

Every request carries the `user_id` of the acting user (query param or request body) — there is no
registration, login, or session handling in this service.

`GET /recommendations` is the core endpoint: it checks whether the given user has existing ratings
and dispatches to `rating_based` or `collaborative` service accordingly.

## Roadmap

- [ ] Project scaffolding (Flask app factory, config, `pyproject.toml`/`uv.lock`, `.gitignore`)
- [x] DB schema: `User`, `Movie`, `Genre`, `Rating` models + first Alembic migration
- [x] Seed script for MovieLens dataset (`data/preprocess.py` + `data/seed.py`)
- [ ] Movie listing/search endpoints
- [ ] Rating endpoint
- [ ] Rating-based recommendation service (for users with ratings)
- [ ] Collaborative filtering recommendation service (cold-start users)
- [ ] `/recommendations` endpoint wiring both strategies together
- [ ] Test suite (models, routes, both recommenders)
- [x] CI (lint + tests on push / PR — `.github/workflows/ci.yml`)

## Contributing

This is a personal/pet project — no formal contribution process yet. Open an issue or PR if you
spot something.
