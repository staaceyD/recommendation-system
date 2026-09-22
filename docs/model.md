# How the model works

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
measured, and both matter — see [Tuning notes](model-tuning.md).

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
that the biases are regularized — see [Tuning notes](model-tuning.md).

Constants and how they were chosen: [model-tuning.md](model-tuning.md).
