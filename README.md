# Recommendation System

A movie recommender web app built with **Flask** and **SQLite**. The database is seeded with a
public movie ratings dataset ([MovieLens](https://grouplens.org/datasets/movielens/)). For any
given user:

- If the user was in the **trained model** (a TensorFlow matrix-factorization model fit offline
  on the ratings table), recommendations come from their learned embedding — the movies with the
  highest predicted rating that they haven't seen.
- If the user is **unknown to the model** (new, or nobody in the training data), recommendations
  fall back to a **collaborative** cold-start ranking: the most popular, well-rated movies
  (Bayesian-adjusted so a few 5-star ratings can't beat a genuinely popular title), preferring
  genres the user has already rated.

Every endpoint receives a `user_id` from the caller — this app does not manage authentication or
create users. Users are the placeholder rows loaded by the seed script from the MovieLens dataset.

## Tech stack

- **Backend:** Python 3.11+, Flask
- **Database:** SQLite (a single file, `instance/recsys.db` — no server to run)
- **ORM / migrations:** Flask-SQLAlchemy, Flask-Migrate (Alembic)
- **ML:** TensorFlow / Keras (matrix-factorization model), pandas, numpy
- **Testing:** pytest, pytest-cov
- **Config:** python-dotenv
- **Env / package management:** [uv](https://docs.astral.sh/uv/)

## Project structure

```
recommendation-system/
├── app/
│   ├── __init__.py            # app factory
│   ├── config.py              # config classes (Dev/Test/Prod)
│   ├── extensions.py          # db, migrate instances
│   ├── models/                # SQLAlchemy models: User, Movie, Genre, Rating
│   ├── routes/                # Flask blueprints (movies, recommendations)
│   ├── services/
│   │   ├── rating_based.py    # recs from the trained model's user embeddings
│   │   └── collaborative.py   # popularity-based cold-start fallback
│   └── ml/
│       ├── model.py           # the Keras matrix-factorization model
│       ├── train.py           # offline training script (python -m app.ml.train)
│       ├── artifact.py        # load a trained model for inference
│       ├── data.py            # pull ratings out of the DB
│       └── vocab.py           # user/movie id <-> embedding index
├── data/
│   ├── raw/                   # MovieLens CSVs: ml-32m/ source + preprocessed output (gitignored)
│   ├── preprocess.py          # raw ml-32m CSVs -> the shape seed.py loads
│   └── seed.py                # loads raw/ into SQLite
├── docs/
│   └── recsys.postman_collection.json  # Postman/Insomnia collection for every endpoint
├── instance/                   # recsys.db + mf/ (trained model) live here (gitignored)
├── migrations/                 # Alembic migrations
├── tests/
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

### 6. Train the recommender

```bash
uv run python -m app.ml.train                     # full training -> instance/mf/
uv run python -m app.ml.train --limit 2000000     # quick partial run
uv run python -m app.ml.train --epochs 3 --dim 16 # smaller / faster
```

Training is **offline** — the app never trains on a request. It reads the `ratings`
table, fits the matrix-factorization model, and writes `instance/mf/`
(`model.keras` + `vocab.json` + `meta.json`). `GET /recommendations` loads that
artifact on first use; retrain and restart to pick up a new one. Without a trained
model every user falls through to the collaborative cold-start path.

### 7. Run the app

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

## API endpoints

| Method | Endpoint                        | Description                                          |
|--------|---------------------------------|-----------------------------------------------------|
| GET    | `/movies`                       | List / search movies (`q`, `genre`, `page`, `per_page`) |
| POST   | `/movies/<id>/rate`             | Rate a movie (JSON body: `user_id`, `rating` 0.5–5.0) |
| GET    | `/recommendations?user_id=<id>` | Recommendations for the user (`limit` optional)      |

Every request carries the `user_id` of the acting user (query param or request body) — there is no
registration, login, or session handling in this service.

`GET /recommendations` is the core endpoint. It tries `rating_based` (the trained model) first and
falls back to `collaborative` (cold-start) when that returns nothing; the response's `strategy`
field says which ran.

### API client collection

`docs/recsys.postman_collection.json` is a ready-to-run collection covering every endpoint, with
query-param and JSON body templates plus saved response examples (including the 400/404 cases).
Import it into **Postman** (*Import → File*) or **Insomnia** (*Import → From File* — it reads the
Postman v2.1 format), then set the collection variables: `baseUrl` (default
`http://127.0.0.1:5000`), `userId`, `newUserId` (a user the trained model has never seen, to
exercise the cold-start path), and `movieId`.

## How the model works

`app/ml/` is a textbook **matrix factorization** recommender in Keras. Every user and every movie
gets a learned vector (an embedding) plus a scalar bias; the predicted rating is

```
global_mean + user_vec · movie_vec + user_bias + movie_bias
```

`app/ml/train.py` fits it against `rating - global_mean` with MSE loss (so RMSE is the quality
metric), shuffles, holds out a validation split, and saves the model + id vocabularies to
`instance/mf/`. At request time `rating_based` loads that once per process and, for a known user,
scores every unseen movie through the model and returns the top `limit`.

`tests/ml/` trains a tiny model on synthetic separable data each run — no committed model blob,
nothing mocked — so the training and inference paths are actually exercised in CI.

## Roadmap

- [ ] Project scaffolding (Flask app factory, config, `pyproject.toml`/`uv.lock`, `.gitignore`)
- [x] DB schema: `User`, `Movie`, `Genre`, `Rating` models + first Alembic migration
- [x] Seed script for MovieLens dataset (`data/preprocess.py` + `data/seed.py`)
- [x] Movie listing/search endpoints (`GET /movies` — title search, genre filter, pagination)
- [x] Rating endpoint (`POST /movies/<id>/rate` — upsert, 404 on unknown movie/user, no user creation)
- [x] Rating-based recommendation service — TensorFlow matrix-factorization model (`app/ml/`, `app/services/rating_based.py`)
- [x] Collaborative cold-start service (`app/services/collaborative.py` — Bayesian-adjusted popularity)
- [x] `/recommendations` endpoint dispatching model → cold-start fallback
- [ ] Test suite (models, routes, both recommenders) — routes + recommenders done; model tests in `tests/ml/`
- [x] CI (lint + tests on push / PR — `.github/workflows/ci.yml`)

## Contributing

This is a personal/pet project — no formal contribution process yet. Open an issue or PR if you
spot something.
