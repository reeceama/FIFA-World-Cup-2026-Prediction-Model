# Parameters

This file lists every number the model runs on, what it does to a simulated game and how it
was set. It also covers why the engine constants and the rating weights were chosen by
different methods.

| Parameter | Value | How it was set | Description |
|---|---|---|---|
| Window | 5 years, frozen 10 June 2026 | judgement | Which matches the model is allowed to see. Anything before June 2021, or after the freeze date, is ignored. |
| Decay rate | 0.00095 | held-out | Older matches count for less, halving in weight roughly every two years, so recent form matters more without older results being dismissed entirely. |
| Raw weight | 0.10 | held-out | How much of a rating comes from goals per game on their own, without accounting for who the opposition was. |
| Elo weight | 0.30 | held-out | How much of a rating comes from Elo rather than from scorelines. The higher it goes, the more attack and defence end up as mirror images of each other. |
| Squad value weight | 0.07 | held-out | Shifts ratings up or down by squad market value, helping teams who have stronger squads. |
| Rho | −0.1075 | maximum likelihood, on the 675 fixtures between the 48 finalists | Bumps up 0-0 and 1-1 and pulls down 1-0 and 0-1, which a plain Poisson model doesn't get right on its own. Mostly shows up in the draw rate. |
| Goal baseline | 1.0559 | maximum likelihood, on the 675 fixtures between the 48 finalists | Scales both teams' expected goals up from the level the ratings are normalised on to the level World Cup matches are actually played at. It changes scorelines rather than who wins. |
| Home advantage | 1.1597 | maximum likelihood, on all 5,328 fixtures in the window | What a host gets playing in their own country, halved when they're in one of the other two. It's applied as a ratio between the two sides, so the match doesn't gain goals overall. |

## Function of weighting

A competition's multiplier says how much one match counts, not how much that competition shapes the ratings. 
That depends on how many of those matches fall inside the window too, which is why the World Cup has the 
biggest multiplier and the smallest share of the total. Reproduced by `python -m wcmodel.weights`.

| competition | multiplier | vs friendly | matches | share of weight |
|---|---:|---:|---:|---:|
| World Cup qualification | 25.0 | 3.3x | 1,493 | 36.9% |
| Continental finals | 37.5 | 5.0x | 472 | 16.2% |
| Continental qualification | 25.0 | 3.3x | 670 | 15.6% |
| Nations League | 20.0 | 2.7x | 674 | 12.2% |
| Friendly | 7.5 | 1.0x | 1,408 | 11.1% |
| Other | 10.0 | 1.3x | 547 | 5.8% |
| World Cup | 55.0 | 7.3x | 64 | 2.2% |

The same applies to the rating inputs. A weight in the parameters table is a share of the blend,
not a share of the result, because each input separates teams by a different amount. Squad value
is not a part of the blend.

| component | weight | share of attack | share of defence |
|---|---:|---:|---:|
| solved ratings | 0.60 | 59.2% | 58.6% |
| Elo prior | 0.30 | 20.8% | 20.6% |
| squad value | 0.07 | 16.5% | 16.1% |
| raw goal rates | 0.10 | 3.5% | 4.7% |

## Rationale for fitting variables

Rho, the goal baseline and home advantage are fitted by maximum likelihood. Rho and the goal
baseline use the 675 fixtures played between the 48 finalists inside the window, since those
are the matches closest to the ones being simulated. Home advantage uses all 5,328 fixtures
instead. Only 3,523 of those were played at a real home venue, and restricting to finalists
would leave too few to estimate from.

These three have a right answer in the data. Set rho too low and the model predicts too few
1-1 draws. Set the goal baseline too high and it predicts too many goals. The scorelines show
it either way, so the likelihood can find the value that fits best.

The three rating weights can't be picked that way. Relaxing them always improves the fit on the
data the model trained on, so a likelihood would set the priors to 0 every time, leaving a
model that explains the past perfectly and predicts nothing. They have to be judged on matches
the fit never saw.

## How the weights were chosen

The data is split by date twice, one split inside the other. A grid search on the inner split
proposes the raw and Elo weights, and the squad value weight ships as a design choice.

The shipped raw and Elo weights don't come from that search. The inner validation set only
holds 83 World Cup fixtures, which isn't enough to choose them on, so I read those two off the
held-out profile curves from the outer split instead.

This means the split that chose a weight can't also judge it. The backtest does that job. It
refits the ratings and the engine constants inside each of 17 folds and scores on matches none
of them saw.

## Intervals

Every value above sits inside its 95% profile likelihood interval. The curves are in
`data/calibration/`, one CSV per parameter.

```bash
python -m wcmodel.fit_mle
```

| Parameter | Shipped | Profile best | 95% interval |
|---|---:|---:|---:|
| Raw weight | 0.10 | 0.10 | [0.00, 0.24] |
| Elo weight | 0.30 | 0.30 | [0.10, 0.45] |
| Squad value weight | 0.07 | 0.04 | [0.01, 0.08] |
| Rho | −0.1075 | −0.105 | [−0.20, −0.005] |
| Goal baseline | 1.0559 | 1.07 | [1.03, 1.12] |
| Home advantage | 1.1597 | 1.16 | [1.14, 1.18] |

Some of these bounds are the edge of the search grid rather than a point the data picks out.
The raw weight's interval covers its entire grid, which means nothing in
that range could be told apart. The engine constants are identified, the rating weights are
not, so they stay where the documented procedure put them.

## Two sets of engine constants

`fit_mle` reports rho, the goal baseline and home advantage twice.

The final refit uses whatever Stage 1 weights its own grid search picked, which is the right
comparison for judging the search. The shipped refit uses the weights in the table above,
which is where the constants in `match.py` come from. Both are written to
`data/calibration/calibration_summary.json`, under `final_engine` and `shipped_refit`.
