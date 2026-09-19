# Recommendation System

A movie recommender web app built with **Flask** and **SQLite**, seeded from
[MovieLens](https://grouplens.org/datasets/movielens/). For any given user:

- If the user was in the **trained model** (a TensorFlow matrix-factorization model fit offline on
  the ratings table), recommendations come from their learned embedding — the movies with the
  highest predicted rating that they haven't seen.
- If the user is **unknown to the model** (new, or nobody in the training data), recommendations
  fall back to a **collaborative** cold-start ranking: the most popular, well-rated movies
  (Bayesian-adjusted so a few 5-star ratings can't beat a genuinely popular title), preferring
  genres the user has already rated.

Every endpoint takes a `user_id` from the caller — there is no registration, login, or session
handling. Users are placeholder rows loaded by the seed script.

## Tech stack

- **Backend:** Python 3.11+, Flask
- **Database:** SQLite (a single file, `instance/recsys.db` — no server to run)
- **ORM / migrations:** Flask-SQLAlchemy, Flask-Migrate (Alembic)
- **ML:** TensorFlow / Keras (matrix factorization), pandas, numpy
- **Testing:** pytest, pytest-cov
- **Env / package management:** [uv](https://docs.astral.sh/uv/)

## Project structure

```
app/
├── models/                # User, Movie, Genre, Rating
├── routes/                # blueprints: movies, recommendations
├── services/
│   ├── rating_based.py    # recs from the trained model's user embeddings
│   └── collaborative.py   # popularity-based cold-start fallback
└── ml/
    ├── model.py           # the Keras matrix-factorization model
    ├── train.py           # offline training      (python -m app.ml.train)
    ├── evaluate.py        # offline evaluation    (python -m app.ml.evaluate)
    ├── metrics.py         # top-K ranking metrics
    ├── artifact.py        # load a trained model for inference
    ├── data.py            # pull ratings out of the DB
    └── vocab.py           # user/movie id <-> embedding index
data/
├── raw/                   # MovieLens CSVs, source + preprocessed (gitignored)
├── preprocess.py          # raw ml-32m CSVs -> the shape seed.py loads
└── seed.py                # loads raw/ into SQLite
docs/
├── recsys.postman_collection.json
└── model-tuning.md        # why the regularization constants are what they are
instance/                  # recsys.db + mf/ (trained model) — gitignored
```

## Getting started

The only prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/), which
manages the Python version, the virtualenv and the dependencies. SQLite ships with Python, so
there is no database server to install.

### 1. Install

```bash
git clone <repo-url>
cd recommendation-system
uv sync
```

Prefix commands with `uv run` rather than activating the venv.

### 2. Configure

```bash
cp .env.example .env    # FLASK_APP, FLASK_ENV, DATABASE_URL
```

`DATABASE_URL=sqlite:///recsys.db` is relative, so Flask resolves it to `instance/recsys.db`; use
`sqlite:////abs/path/recsys.db` to put the file elsewhere.

### 3. Migrate

```bash
uv run flask db upgrade
```

### 4. Get the dataset

`data/raw/` is gitignored. Download [`ml-32m.zip`](https://grouplens.org/datasets/movielens/) and
unzip it so the files land at `data/raw/ml-32m/`, then convert them to the shape the seed script
loads (`movieId,title,genre` one row per pair, and `userId,movieId,rating`):

```bash
uv run python data/preprocess.py
```

### 5. Seed

```bash
uv run python data/seed.py                    # full seed (~32M ratings, a few minutes, ~1.2 GB)
uv run python data/seed.py --reset            # wipe the tables first
uv run python data/seed.py --ratings-limit N  # only the first N ratings
```

Users are synthesised from the ids in `ratings.csv` as placeholder rows. The script targets
whichever database the current `FLASK_ENV` resolves to, so it won't touch the test database.

### 6. Train

```bash
uv run python -m app.ml.train                     # full training -> instance/mf/
uv run python -m app.ml.train --limit 2000000     # quick partial run
uv run python -m app.ml.train --epochs 3 --dim 16 # smaller / faster
uv run python -m app.ml.train --l2 1e-4           # stronger regularization
```

Training is **offline** — the app never trains on a request. It reads the `ratings` table and
writes `instance/mf/` (`model.keras` + `vocab.json` + `meta.json`), which `GET /recommendations`
loads on first use; retrain and restart to pick up a new one. Without a trained model every user
falls through to the cold-start path.

Raising `--epochs` past the default of 5 has made the *ranking* worse even though RMSE barely
moved — see [docs/model-tuning.md](docs/model-tuning.md) before turning it up.

### 7. Check the recommendations are any good

```bash
uv run python -m app.ml.evaluate                       # the whole ratings table
uv run python -m app.ml.evaluate --limit 2000000       # a slice
uv run python -m app.ml.evaluate --k 20 --users 2000   # longer lists, more users
uv run python -m app.ml.evaluate --json report.json    # also write the raw numbers
```

### 8. Run

```bash
uv run flask run    # http://localhost:5000
```

## API endpoints

| Method | Endpoint                        | Description                                          |
|--------|---------------------------------|------------------------------------------------------|
| GET    | `/movies`                       | List / search movies (`q`, `genre`, `page`, `per_page`) |
| POST   | `/movies/<id>/rate`             | Rate a movie (JSON body: `user_id`, `rating` 0.5–5.0) |
| GET    | `/ratings`                      | Rated movies with their rating (`user_id`, `min_rating`, `max_rating`, `page`, `per_page`) |
| GET    | `/recommendations?user_id=<id>` | Recommendations for the user (`limit` optional)      |

`GET /recommendations` is the core endpoint: it tries `rating_based` (the trained model) first and
falls back to `collaborative` (cold-start) when that returns nothing. The response's `strategy`
field says which ran.

`docs/recsys.postman_collection.json` covers every endpoint with body templates and saved response
examples (including the 400/404 cases). Import it into Postman or Insomnia, then set `baseUrl`,
`userId`, `newUserId` (a user the model has never seen, to exercise cold start) and `movieId`.

## Development

```bash
uv run pytest                  # tests
uv run pytest --cov=app        # with coverage
uv run ruff check .            # lint        (CI gate)
uv run ruff format .           # auto-format (CI checks with --check)
uv add <package>               # add a dependency; --dev for a dev-only one
```

`uv add` updates `pyproject.toml` and `uv.lock` — commit both.

Tests run under `FLASK_ENV=testing` against an in-memory SQLite database, so they never touch
`instance/recsys.db`. `.github/workflows/ci.yml` runs lint and tests on every push and PR against
`main`; gating merges on them is repository configuration (**Settings → Branches**), not code.

## How the model works

`app/ml/` is a textbook **matrix factorization** recommender in Keras. Every user and every movie
gets a learned vector (an embedding) plus a scalar bias; the predicted rating is

```
global_mean + user_vec · movie_vec + user_bias + movie_bias
```

`app/ml/train.py` fits it against `rating - global_mean` with MSE loss, and saves the model and id
vocabularies to `instance/mf/`. At request time `rating_based` loads that once per process and,
for a known user, scores every unseen movie and returns the top `limit` — skipping movies with
fewer than `MIN_SUPPORT` ratings behind their embedding.

The regularization is not a pair of magic numbers: `L2` is *derived* from the size of the training
set rather than fixed, and the biases carry a strength of their own on top of that. Both are
measured, and both matter — see [Tuning notes](#tuning-notes).

`tests/ml/` trains a tiny model on synthetic separable data each run — no committed model blob,
nothing mocked — so training and inference are actually exercised in CI.

## Evaluation

`uv run python -m app.ml.evaluate` answers the question training RMSE can't: is the *list* the
user actually sees any good?

Each user's ratings are split 80/20 — **per user**, so test users keep a history to be modelled
from. A model is trained on the training half only, asked for a top-K list per user, and scored on
how much of that user's holdout it recovered (relevant = the user rated it 3.5 or higher). The
same users and the same holdout are scored for `popularity` (the cold-start ranking) and `random`,
because a score like `precision@10 = 0.08` means nothing on its own.

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

`uv run python -m app.ml.evaluate --limit 1000000 --users 500`:

```
strategy                   prec@10   recall@10     ndcg@10      map@10      hit@10    coverage     novelty
model (support >= 100)      0.1392      0.1108      0.1851      0.1050      0.6040      0.0127      9.4576
model (unfiltered)          0.1370      0.1101      0.1832      0.1040      0.5940      0.0143      9.6004
popularity                  0.0786      0.0604      0.1034      0.0513      0.4460      0.0015      9.6283
random                      0.0010      0.0002      0.0008      0.0002      0.0100      0.1854     17.5163

Rating prediction on the holdout: RMSE 0.8382 (predicting the global mean every time: 1.0516)
```

The model beats popularity on every ranking metric while recommending 8x more of the catalogue
(coverage 0.0127 against 0.0015), so it is genuinely personalising rather than imitating the
baseline. The unfiltered row sits just below the filtered one, which is the expected shape now
that the biases are regularized — see [Tuning notes](#tuning-notes).

## Tuning notes

The regularization constants are derived rather than guessed, and two of them were set against
mistakes that `prec@K` alone would have walked straight into. If you are about to change `L2`,
`BIAS_L2_MULTIPLIER`, `MIN_SUPPORT` or `--epochs`, read **[docs/model-tuning.md](docs/model-tuning.md)**
first — it records what each is worth, how it was measured, and which knobs bite back.

The one rule, if you read nothing else: never tune by maximising `prec@K`. Offline top-N rewards
narrowing the catalogue, so check `coverage` alongside it every time.

## Contributing

A personal/pet project — no formal process. Open an issue or PR if you spot something.
