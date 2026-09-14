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
│       ├── evaluate.py        # offline evaluation (python -m app.ml.evaluate)
│       ├── metrics.py         # top-K ranking metrics
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
uv run python -m app.ml.train --l2 1e-4           # stronger regularization
```

Training is **offline** — the app never trains on a request. It reads the `ratings`
table, fits the matrix-factorization model, and writes `instance/mf/`
(`model.keras` + `vocab.json` + `meta.json`). `GET /recommendations` loads that
artifact on first use; retrain and restart to pick up a new one. Without a trained
model every user falls through to the collaborative cold-start path.

Raising `--epochs` past the default of 5 makes the *ranking* worse even though RMSE barely
moves — see [Evaluation](#evaluation) before turning it up.

### 7. Check whether the recommendations are any good

```bash
uv run python -m app.ml.evaluate                       # evaluate on the whole ratings table
uv run python -m app.ml.evaluate --limit 2000000       # quick run on a slice
uv run python -m app.ml.evaluate --k 20 --users 2000   # longer lists, more users
uv run python -m app.ml.evaluate --json report.json    # also write the raw numbers
```

See [Evaluation](#evaluation) for what the numbers mean.

### 8. Run the app

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
scores every unseen movie through the model and returns the top `limit` — skipping movies with
fewer than `MIN_SUPPORT` ratings behind their embedding, for the reasons in
[Evaluation](#evaluation).

`tests/ml/` trains a tiny model on synthetic separable data each run — no committed model blob,
nothing mocked — so the training and inference paths are actually exercised in CI.

## Evaluation

`uv run python -m app.ml.evaluate` answers the question training RMSE can't: is the *list* the
user actually sees any good?

**How it works.** Each user's ratings are split — most train, a random 20% held out. A model is
trained on the training half only, then asked for a top-K list per user, and scored on how much of
that user's holdout it recovered (relevant = the user rated it 3.5 or higher). The same users and
the same holdout are scored for two baselines: `popularity` (the Bayesian-adjusted ranking the
cold-start fallback uses) and `random`. Absolute scores like "precision@10 = 0.08" mean nothing in
isolation — the baselines are what make them readable.

It always trains its own model and deliberately *cannot* score `instance/mf/`: that artifact was
fitted on every rating in the table, so every held-out row would already be memorised.

| Metric | Reads as |
|--------|----------|
| `prec@K` | Share of the K recommendations the user really liked |
| `recall@K` | Share of everything they liked that the list caught |
| `ndcg@K` | Position-weighted: 1.0 means every hit sits as high as it could |
| `map@K` | Precision averaged over the hits — rewards hits near the top |
| `hit@K` | Share of users whose list contained *anything* they liked |
| `coverage` | Share of the catalogue ever recommended (0.001 = the same few hundred titles) |
| `novelty` | Bits of surprise; low means blockbusters, high means the long tail |

**What it currently says.** On a 1M-rating slice, 500 users, K=10:

```
strategy                   prec@10   recall@10     ndcg@10      map@10      hit@10    coverage     novelty
model (support >= 100)      0.1126      0.0868      0.1491      0.0802      0.5360      0.0148      9.9621
model (unfiltered)          0.0974      0.0713      0.1294      0.0678      0.4960      0.0156     10.9100
popularity                  0.0786      0.0604      0.1034      0.0513      0.4460      0.0015      9.6283
random                      0.0010      0.0002      0.0008      0.0002      0.0100      0.1854     17.5163

Rating prediction on the holdout: RMSE 0.8075 (predicting the global mean every time: 1.0516)
```

The model beats popularity on every ranking metric — ~43% more precision@10 — while recommending
10x more of the catalogue (coverage 0.0148 against 0.0015), so it is genuinely personalising
rather than imitating the baseline. It holds on a 4M-rating slice too, by a narrower margin
(prec@10 0.0640 against 0.0508).

That was not true when the evaluation was first written. It reported `prec@10 = 0.064` against
popularity's `0.079` — the model *lost*. What fixed it:

- **`L2 = 1e-6` → `1e-5`** (`app/ml/model.py`). At 1e-6, thinly-rated movies picked up overfitted
  embeddings predicting extreme ratings, and floated to the top of a full-catalogue ranking. At
  1e-4 and above the embeddings collapse to zero and the model degenerates into a bias-only
  popularity clone (1e-4 and 1e-3 score identically). This alone took prec@10 to 0.097.
- **A minimum-support filter at inference** (`MIN_SUPPORT = 100` in `app/ml/artifact.py`). A movie
  costs `dim + 1` = 33 parameters, so an embedding fitted from a handful of ratings is
  underdetermined. Movies under the threshold are held back and only used to top the list up if
  too few well-supported ones remain — the same shape as the cold-start ranker backfilling past
  its genre filter. Training stores the per-movie counts in `vocab.json`, so serving never has to
  ask the database.

Two honest caveats:

- **Offline top-N flatters popularity**, because unrated ≠ disliked: a blockbuster scores as a
  "hit" partly because it is the kind of film a user had the chance to rate at all. This cuts the
  other way too — tuning `MIN_SUPPORT` upward keeps "improving" the metric (800 scores better than
  100 on the 1M slice) purely by narrowing the catalogue until the model *is* the popularity
  baseline, at 0.6% coverage. 100 is set on statistical grounds, not by chasing the number, and
  the report always carries an unfiltered row so the cost of the filter stays visible.
- **Training longer still hurts**, just no longer catastrophically: at 15 epochs prec@10 is 0.097
  rather than 0.113 (before these fixes it collapsed to 0.026, far below the baseline). RMSE
  barely moves either way, which is exactly why RMSE alone was never going to catch this.
  `DEFAULT_EPOCHS = 5` stands.

Worth trying from here: early stopping on a ranking metric rather than on loss, and a ranking loss
(BPR / implicit feedback) instead of MSE on explicit ratings — MSE optimises rating accuracy, and
ranking is what the app actually serves.

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
- [x] Offline evaluation (`app/ml/evaluate.py` — per-user holdout, top-K metrics vs popularity/random baselines)
- [x] Close the gap the evaluation found — stronger L2 + a minimum-support filter put the model ahead of popularity (see [Evaluation](#evaluation))
- [ ] Rank-aware training: early stopping on NDCG, and a ranking loss (BPR) instead of MSE

## Contributing

This is a personal/pet project — no formal contribution process yet. Open an issue or PR if you
spot something.
