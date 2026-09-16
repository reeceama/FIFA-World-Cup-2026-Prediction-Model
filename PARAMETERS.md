# Parameters

This file lists every number the model runs on, what it does to a simulated game and how it
was set. It also covers why the engine constants and the rating weights were chosen by
different methods, and where the values here differ from what the calibration script reports.

| Parameter | Value | How it was set | Description |
|---|---|---|---|
| Window | 5 years, frozen 10 June 2026 | judgement | Which matches the model is allowed to see. Anything before June 2021, or after the freeze date, is ignored. |
| Decay rate | 0.00095 | held-out | Older matches count for less, halving in weight roughly every two years, so recent form matters more without older results being thrown away entirely. |
| Shrinkage | 0.84 | held-out | Compresses the rating spread, since ratings solved from few matches overstate how far the best and worst teams sit from the field. It sits at the low end of its likelihood interval. |
| Raw weight | 0.10 | held-out | How much of a rating comes from goals per game on their own, without accounting for who the opposition was. |
| Elo weight | 0.30 | held-out | How much of a rating comes from Elo rather than from scorelines. The higher it goes, the more attack and defence end up as mirror images of each other. |
| Squad value weight | 0.07 | held-out | Shifts ratings up or down by squad market value, which helps with teams whose results haven't caught up with their talent. |
| Rho | −0.1079 | maximum likelihood, on the 675 fixtures between the 48 finalists | Bumps up 0-0 and 1-1 and pulls down 1-0 and 0-1, which a plain Poisson model doesn't get right on its own. Mostly shows up in the draw rate. |
| Goal baseline | 1.0972 | maximum likelihood, on the 675 fixtures between the 48 finalists | Scales both teams' expected goals up from the level the ratings are normalised on to the level World Cup matches are actually played at. It changes scorelines rather than who wins. |
| Home advantage | 1.1778 | maximum likelihood, on all 5,328 fixtures in the window | What a host gets playing in their own country, halved when they're in one of the other two. It's applied as a ratio between the two sides, so the match doesn't gain goals overall. |

## Why some are fitted and others aren't

Rho, the goal baseline and home advantage are fitted by maximum likelihood. Rho and the goal
baseline use the 675 fixtures played between the 48 finalists inside the window, since those
are the matches closest to the ones being simulated. Home advantage uses all 5,328 fixtures
instead. Only 3,523 of those were played at a real home venue, and restricting to finalists
would leave too few to estimate from.

These three have a right answer in the data. Set rho too low and the model predicts too few
1-1 draws. Set the goal baseline too high and it predicts too many goals. The scorelines show
it either way, so the likelihood can find the value that fits best.

The four rating weights can't be picked that way. Relaxing them always improves the fit on the
data the model trained on, so a likelihood would set shrinkage to 1 and the priors to 0 every
time, leaving a model that explains the past perfectly and predicts nothing. They have to be
judged on matches the fit never saw.

## How the weights were chosen

The data is split by date twice, one split inside the other. A grid search on the inner split
picked shrinkage 0.84, raw 0.05, Elo 0.27, value 0.07.

Shrinkage and value ship as the search found them. Raw and Elo don't, since the inner
validation set only holds 83 World Cup fixtures, which isn't enough to choose four weights on.
I read those two off the held-out profile curves from the outer split instead.

This means the outer split can't then be used to judge the model, since it helped pick it.
The backtest does that job instead. It refits the ratings and the engine constants inside each
of 17 folds and scores on matches none of them saw.

## Intervals

Every value above sits inside its 95% profile likelihood interval. The curves are in
`data/calibration/`, one CSV per parameter.

```bash
python -m wcmodel.fit_mle
```

Shrinkage sits at the bottom of its interval. Below 0.84 the profile likelihood falls away
sharply, and its own minimum is at 0.96. Rho has no identified lower bound, and its profile
runs to the edge of the grid.

## Why the calibration summary disagrees

`data/calibration/calibration_summary.json` reports different engine constants from the ones
above. The script searches for its own Stage 1 weights, then fits rho and the goal baseline
against the ratings those weights produce, so its numbers belong to the configuration the
search picked rather than the one that ships. Rerunning it won't reproduce these values.
