# Model tuning notes

Why the model's regularization constants are what they are, and which knobs bite back. The short
version lives in the code comments in `app/ml/model.py` and `app/ml/artifact.py`; this is the
working out behind them. See [Evaluation](../README.md#evaluation) for how the numbers are
produced.

Everything here is measured with `uv run python -m app.ml.evaluate`, which trains its own model on
a held-out split — it never scores `instance/mf/`, since that artifact has seen every rating.

## The one rule

**Never tune by maximising `prec@K` alone.** Offline top-N rewards narrowing the catalogue, and
the limit of that is recommending only blockbusters — i.e. becoming the popularity baseline. Read
`coverage` next to `prec@K` every time, and prefer a measurement the holdout cannot see when one
is available. Both mistakes below were caught that way.

## Why `L2` is scaled to the dataset size

On a 1M slice the model beat the popularity baseline comfortably. That margin used to narrow as
the slice grew, and at full size it vanished: the served model handed *every* user the same 20
titles — the global `movie_bias` ranking, which on MovieLens is
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
6M slice, 500 users, K=10 — one scale up from the run in the README:

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

## Why the biases carry `L2` too

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

## Calibrating `BIAS_L2_MULTIPLIER`

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

## Knobs that bite back

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

## Worth trying next

Early stopping on a ranking metric rather than on loss, and a ranking loss (BPR / implicit
feedback) instead of MSE — MSE optimises rating accuracy, and ranking is what the app actually
serves.
