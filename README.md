# Recommendation System

A movie recommender web app built with **Flask** and **MySQL**. The database is seeded with a
public movie ratings dataset ([MovieLens](https://grouplens.org/datasets/movielens/)). For any
given user:

- If the user has **rated movies already**, recommendations are generated from that rating
  history (similar movies / similar users based on what they've rated).
- If the user is **new and has no ratings yet**, recommendations fall back to **collaborative
  filtering** over the full ratings matrix (e.g. popularity among similar user cohorts) to give a
  reasonable cold-start experience.

## Tech stack

- **Backend:** Python 3.11+, Flask
- **Database:** MySQL 8
- **ORM / migrations:** Flask-SQLAlchemy, Flask-Migrate (Alembic)
- **ML / data:** pandas, numpy, scikit-learn (and/or `implicit` / `surprise` for matrix
  factorization CF)
- **Testing:** pytest, pytest-cov
- **Config:** python-dotenv
- **Env / package management:** [uv](https://docs.astral.sh/uv/)
- **Local infra:** Docker Compose (for MySQL)

## Project structure (planned)

```
recommendation-system/
├── app/
│   ├── __init__.py            # app factory
│   ├── config.py              # config classes (Dev/Test/Prod)
│   ├── extensions.py          # db, migrate instances
│   ├── models/                # SQLAlchemy models: User, Movie, Rating
│   ├── routes/                # Flask blueprints (auth, movies, ratings, recommendations)
│   └── services/
│       ├── rating_based.py    # recs from a user's own rating history
│       └── collaborative.py   # collaborative filtering for cold-start users
├── data/
│   ├── raw/                   # downloaded MovieLens CSVs (gitignored)
│   └── seed.py                # loads raw/ into MySQL
├── migrations/                 # Alembic migrations
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_routes.py
│   └── test_recommenders.py
├── docker-compose.yml          # local MySQL
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
- MySQL 8 (locally installed, or via Docker)
- Docker + Docker Compose (optional, recommended for MySQL)

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
SECRET_KEY=change-me
DATABASE_URL=mysql+pymysql://recsys:recsys@localhost:3306/recsys
```

### 3. Start MySQL

Using Docker Compose:

```bash
docker compose up -d
```

Or point `DATABASE_URL` at an existing local MySQL instance.

### 4. Run migrations

```bash
uv run flask db upgrade
```

### 5. Seed the database

Downloads (or reads a local copy of) the MovieLens dataset and loads movies/ratings into MySQL:

```bash
uv run python data/seed.py
```

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

Tests run against a separate test database/config (`FLASK_ENV=testing`) so they never touch dev
data.

## API endpoints (planned)

| Method | Endpoint                        | Description                                   |
|--------|----------------------------------|------------------------------------------------|
| POST   | `/auth/register`                | Create a user                                   |
| POST   | `/auth/login`                   | Log in                                          |
| GET    | `/movies`                       | List / search movies                            |
| POST   | `/movies/<id>/rate`             | Rate a movie                                    |
| GET    | `/recommendations`              | Get recommendations for the current user        |

`GET /recommendations` is the core endpoint: it checks whether the user has existing ratings and
dispatches to `rating_based` or `collaborative` service accordingly.

## Roadmap

- [ ] Project scaffolding (Flask app factory, config, `pyproject.toml`/`uv.lock`, `.gitignore`)
- [ ] Docker Compose for local MySQL
- [ ] DB schema: `User`, `Movie`, `Rating` models + first Alembic migration
- [ ] Seed script for MovieLens dataset
- [ ] Auth (register/login)
- [ ] Movie listing/search endpoints
- [ ] Rating endpoint
- [ ] Rating-based recommendation service (for users with ratings)
- [ ] Collaborative filtering recommendation service (cold-start users)
- [ ] `/recommendations` endpoint wiring both strategies together
- [ ] Test suite (models, routes, both recommenders)
- [ ] CI (lint + tests on push)

## Contributing

This is a personal/pet project — no formal contribution process yet. Open an issue or PR if you
spot something.
