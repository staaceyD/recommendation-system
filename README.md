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
docs/                      # Postman/Insomnia collection
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

Raising `--epochs` past the default of 5 makes the *ranking* worse even though RMSE barely moves —
read [Evaluation](#evaluation) before turning it up.

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
fewer than `MIN_SUPPORT` ratings behind their embedding. `MIN_SUPPORT` was chosen by the
evaluation below; the `L2` strength is *derived* from the size of the training set rather than
fixed, and applies to the biases as well as the factors — both for the reasons in the two sections
below. The code comments record the numbers.

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

On a 1M-rating slice, 500 users, K=10:

```
strategy                   prec@10   recall@10     ndcg@10      map@10      hit@10    coverage     novelty
model (support >= 100)      0.1126      0.0868      0.1491      0.0802      0.5360      0.0148      9.9621
model (unfiltered)          0.0974      0.0713      0.1294      0.0678      0.4960      0.0156     10.9100
popularity                  0.0786      0.0604      0.1034      0.0513      0.4460      0.0015      9.6283
random                      0.0010      0.0002      0.0008      0.0002      0.0100      0.1854     17.5163

Rating prediction on the holdout: RMSE 0.8075 (predicting the global mean every time: 1.0516)
```

The model beats popularity on every ranking metric while recommending 10x more of the catalogue
(coverage 0.0148 against 0.0015), so it is genuinely personalising rather than imitating the
baseline.

### Why `L2` is scaled to the dataset size

That margin used to narrow as the slice grew, and at full size it vanished: the served model
handed *every* user the same 20 titles — the global `movie_bias` ranking, which on MovieLens is
nature documentaries and prestige classics. Personalisation was worth 0.0104 RMSE over a
bias-only model, and user embeddings had ended up *smaller* than their random initialisation.

The cause is that `embeddings_regularizer` penalises the whole embedding matrix on every step,
while a row's data gradient arrives only on its own ratings — so the effective strength on a row
with `c` ratings goes as `l2 * N / c`. Batch size cancels; dataset size does not. A fixed `l2`
therefore means *more* regularization the more data you train on, until the embeddings are flat
and the biases explain everything. Measured as the ratio of `movie_bias` spread to
personalisation spread, at a fixed `l2 = 1e-5`:

| ratings | 2M | 6M | 12M | 24M | 32M (full) |
|---|---|---|---|---|---|
| bias : personalisation | 2.6x | 7.8x | 14.6x | 36.4x | 56.0x |
| distinct titles in 10 users' top-20 | 101 | 49 | 37 | 26 | ~21 |

`scaled_l2` holds `l2 * N` fixed instead, which holds that ratio at ~2.5x across all of them. On a
6M slice, 500 users, K=10 — the same run the old numbers above came from, one scale up:

```
strategy                   prec@10   recall@10     ndcg@10      map@10      hit@10    coverage     novelty
model (support >= 100)      0.0850      0.0498      0.1071      0.0542      0.4360      0.0093     11.7415
model (unfiltered)          0.0620      0.0339      0.0836      0.0424      0.3400      0.0075     15.3309
popularity                  0.0542      0.0372      0.0654      0.0273      0.3540      0.0006     10.9797
random                      0.0004      0.0002      0.0003      0.0001      0.0040      0.0979     19.7253

Rating prediction on the holdout: RMSE 0.7817 (predicting the global mean every time: 1.0611)
```

Against the same slice with the old fixed `l2`, that is prec@10 0.0850 against 0.0450 — which had
been *below* the popularity baseline's 0.0542.

A per-observation penalty (`activity_regularizer`) is scale-invariant without any scaling, and was
tried: it fails, because it gives up the support-proportional shrinkage that keeps thinly-rated
movies from predicting extremes. At 6M it scored prec@10 0.038, below popularity, and 0.003
unfiltered — barely above random.

### Why the biases carry `L2` too

Scaling `L2` fixed the factors but left the *biases* unregularized — they were the only parameters
in the model with no penalty at all. That reopened the same failure through a different door.

An unregularized `movie_bias` never shrinks toward zero, so it reaches a movie's raw mean no matter
how thin the evidence behind it. MovieLens documentaries are exactly the shape that exploits:

| genre | avg rating | ratings per movie |
|---|---|---|
| Crime | 3.692 | 801 |
| **Documentary** | **3.691** | **47** |
| Drama | 3.682 | 421 |

Same average as Crime on a seventeenth of the ratings. Nothing pulled those biases back, so they
outranked well-supported titles for any user whose own vector was small — and `|user_vec|` grows
with rating count (0.51 under 30 ratings, 1.14 over 500), so that is every user with a short
history. On the served 32M artifact, **55% of users had three or more documentaries in their
top-10** and Documentary was the second most recommended genre overall. Decomposing the score shows
it directly — a 141-rating user was fine, a 52-rating user was not:

```
user 1 (141 ratings, |vec| 1.56)  Terminator               4.724 = dot +1.401  movie_bias +0.236
user 2 ( 52 ratings, |vec| 0.92)  Alone in the Wilderness  5.389 = dot +0.358  movie_bias +0.738  (n=410)
```

Giving the biases the same `l2` as the factors — as classic SVD++ does — is the fix. On the 6M
slice, against the run above:

```
strategy                   prec@10   recall@10     ndcg@10      map@10      hit@10    coverage     novelty
model (support >= 100)      0.1000      0.0638      0.1265      0.0668      0.4830      0.0125     10.9457
model (unfiltered)          0.0983      0.0620      0.1248      0.0661      0.4770      0.0124     11.0744
popularity                  0.0483      0.0326      0.0585      0.0240      0.3270      0.0007     10.9880

Rating prediction on the holdout: RMSE 0.7831 (predicting the global mean every time: 1.0611)
```

Every ranking metric improves — prec@10 0.0857 → 0.1000, ndcg 0.1088 → 0.1265, hit rate 0.42 →
0.48 — and *coverage rises* (0.0117 → 0.0125), so this is not the usual precision-by-narrowing
trade. Documentary-heavy lists fall from 42% of users to 17% on the slice, `movie_bias` spread
halves (0.399 → 0.191), and Documentary leaves the top six recommended genres. Holdout RMSE is
unchanged within noise (0.7817 → 0.7831), which is the point: RMSE never saw this bug.

The most telling number is the unfiltered row. It used to sit far below the filtered model (0.0599
vs 0.0857); now the two are nearly equal (0.0983 vs 0.1000). **`MIN_SUPPORT` was largely
compensating for this** — it was a filter over a bug, not just a filter over thin evidence. It is
still worth keeping (an embedding does need more evidence than an average does), but it is now
doing far less work, and that is the honest reading of it.

**This only partly transfers to full scale.** Retrained on all 32M ratings, `movie_bias` spread
falls by more than half (0.514 → 0.218) and individual lists improve a lot — the 52-rating user
above goes from three documentaries to one, with the rest (*It's a Wonderful Life*, *Fiddler on the
Roof*, *The Sound of Music*, *Fly Away Home*) finally matching the Children/Musical/Romance history
they actually rated, and their personalisation term now outweighs the item bias instead of losing
to it. But across a 300-user sample, documentary-heavy lists fall only 55% → 47%, against 42% → 17%
on the slice.

The reason is that the biases were still borrowing the *factors'* strength. Holding `l2 * N` fixed
keeps the aggregate penalty constant, but a movie's rating count `c` also grows with `N` — a
documentary with 47 ratings at 6M has ~250 at 32M — and the effective per-row strength goes as
`l2 * N / c`, so every row is shrunk less at full scale. A bias is also fitted against a single
mean rather than a 32-dim direction, so the penalty that suits the factors barely moves it.

### Calibrating `BIAS_L2_MULTIPLIER`

So the biases get a strength of their own: `scaled_bias_l2 = scaled_l2(N) * BIAS_L2_MULTIPLIER`.
Swept at full scale — 32M ratings, one split, the same 1,000 users for every arm:

| ×factors' `l2` | 1 | 3 | 10 | **30** | 50 | 100 | 150 | 300 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| prec@10 | .0637 | .0712 | .0815 | **.0876** | .0904 | .0932 | .0928 | .0916 | .0905 |
| ndcg@10 | .0825 | .0933 | .1057 | **.1156** | — | .1228 | — | .1231 | .1201 |
| coverage | .0121 | .0134 | .0144 | **.0155** | — | .0174 | — | .0189 | .0193 |
| novelty | 12.43 | 11.81 | 11.09 | **10.71** | — | 10.73 | — | 10.85 | 10.91 |
| holdout RMSE | .7805 | .7846 | .7884 | **.7900** | — | .7912 | — | .7920 | .7923 |
| documentary-heavy users | 49.7% | 39.7% | 14.7% | **6.3%** | — | 6.7% | — | 9.7% | 9.3% |
| `movie_bias` spread | .215 | .161 | .103 | **.061** | — | .032 | — | .017 | .008 |

(popularity baseline on the same split: prec@10 .0274; random: .0001)

The optimum is **interior**, which is the first thing worth knowing: precision rises to ×100 and
then falls, so this is not "the bias term should be deleted". By ×1000 the term is effectively gone
(spread 0.008) and every metric is worse for it. The bias carries real signal; it was simply
carrying far too much.

**The shipped value is ×30, which is not the precision argmax.** That is deliberate, and it is the
same trap `MIN_SUPPORT` is warned about below — here it just wears a different disguise, because
`coverage` *rises* across the whole sweep (0.0121 → 0.0193), so the usual tell is absent. Two
things prec@10 cannot see decide it:

- **The symptom bottoms out at ×30.** Documentary-heavy lists hit their minimum at 6.3% and ×100
  does not improve on it (6.7%). Past ×30, more shrinkage buys no more of the actual fix.
- **Past ×30 the bias stops reaching the ranking at all.** Measured against the catalogue's own
  average ratings — independent of the holdout, so prec@10 has no say in it:

  | | ×10 | **×30** | ×100 |
  |---|---|---|---|
  | true mean rating of recommended movies | 4.070 | **4.014** | 3.959 |
  | share of recommendations the catalogue rates under 3.0 | 0.8% | **1.9%** | 4.0% |
  | calibration MAE (predicted vs true mean) | 0.534 | **0.563** | 0.604 |

  ×100 recommends films the catalogue rates below 3.0 at **2.1× the rate of ×30**, and predicts
  their ratings worse. prec@10 rewards it anyway, because a taste-matched bad film is still a film
  the user rated — offline top-N asks "would they have watched it", not "is it any good".

The ×30 → ×100 precision gain is real, not noise (aggregate prec@10 over 1,000 users has an SE of
±0.0041, but compared pairwise over the same users the gap is +0.0056, 95% CI [+0.0031, +0.0081],
t = 4.44). It is simply not worth what it costs. Holdout RMSE also drifts the wrong way across the
sweep (0.7805 → 0.7900) — expected, since MSE is exactly what an unregularized bias is good at.

Two caveats worth knowing. The multiplier was calibrated **at 32M only**; it is a ratio, so
`scaled_l2` should carry it to other dataset sizes, but that is an assumption rather than a
measurement. And the size of it — an effective `l2 * N` of 1,600 where the textbook bias λ for
MovieLens is nearer 10–25 — is far above what the `l2 * N / c` argument alone predicts. Adam
rescales gradients per parameter, so that argument gives the right *form* and direction but not
the magnitude; the magnitude is empirical.

Two things to know before touching the knobs:

- **Offline top-N flatters popularity**, because unrated ≠ disliked: a blockbuster scores as a
  "hit" partly because it is the kind of film a user had the chance to rate at all. So raising
  `MIN_SUPPORT` keeps "improving" precision — 800 beats 100 on the 1M slice — purely by narrowing
  the catalogue until the model *is* the popularity baseline, at 0.6% coverage. It is set on
  statistical grounds instead, and every report carries an unfiltered row so the filter's cost
  stays visible. Watch `coverage` alongside `prec@K`.
- **Training longer used to hurt the ranking** while RMSE barely moved: 15 epochs gave prec@10
  0.097 against 0.113 at the default of 5. That was measured before `L2` was scaled, when longer
  training meant more steps of a penalty that was already too strong, so it needs re-measuring
  before it is trusted again — the default of 5 epochs is inherited, not re-derived. RMSE alone
  was never going to catch it either way.

Worth trying next: early stopping on a ranking metric rather than on loss, and a ranking loss
(BPR / implicit feedback) instead of MSE — MSE optimises rating accuracy, and ranking is what the
app actually serves.

## Contributing

A personal/pet project — no formal process. Open an issue or PR if you spot something.
