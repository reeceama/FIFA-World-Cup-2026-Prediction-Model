# Changelog

## v1.0.1 Patch Notes

- `evaluate.py` now scores the final and reports the predicted champion.
- `run_forecast.py` default seed set back to 2026, the seed that produced the published forecast.
- Added a table of the 95% profile likelihood intervals to `PARAMETERS.md`.

## v1.0.0 Patch Notes

- Output is now probabilistic. 100,000 Monte Carlo tournaments produce a probability for every
  team reaching every round, replacing the single deterministic bracket.
- `monte_carlo_gs` and `monte_carlo_ko` now draw a scoreline from the distribution instead of
  returning the most likely one, so a single simulation is a sample rather than a point
  estimate. The `threshold` parameter is removed.
- Team ratings now come from an iterative opponent-adjusted solve (`solve_team_ratings`), where
  a team's attack is its goals scored divided by the defensive strength of everyone it faced.
  Replaces the weighted goal averages of `calc_avg_weighted_goals` and `calc_avg_weighted_conceded`.
- `elo_modifier` removed entirely. Elo now enters as a prior on the ratings inside
  `blend_ratings` rather than as an xG tilt applied at match time.
- Squad market value added as a rating component (`apply_squad_value`), and a shrinkage
  exponent added to compress the spread of the blended ratings.
- `rho` and home advantage are now fitted by maximum likelihood in `fit_mle` instead of being
  set by hand (−0.1079 and 1.1778).
- Added a fitted goal baseline (1.0972) that scales expected goals to the level matches
  between the finalists are played at. This replaces `performance_weight`, where the goal
  level was matched by tuning the exponents until the weighted stats lined up with historical
  totals. `performance_weight` is removed entirely, with opponent strength now handled inside
  the rating solve.
- Added profile likelihoods for every fitted value, written to `data/calibration/` as a curve
  per parameter with a 95% interval.
- Elo is now read per fixture from `wf_elo_history.csv` rather than as a multi-year average,
  so a match is rated using what was known before it was played.
- Time decay changed from a yearly to a daily rate and retuned on held-out
  validation, moving the half-life from about 2.3 years to 2.0.
- Data window extended to 10 June 2026.
- Cards reverted from a binomial back to a Poisson distribution, with red card rates now
  including second yellows.
- Added `match_probabilities` and `score_matrix`, which report win, draw and loss probabilities
  and the full scoreline distribution for any pairing without simulating it.
- `avg_weighted_goals` and `avg_weighted_conceded` renamed to `attack_rating` and
  `defence_rating`, which describe what they now are.
- Code moved into an installable package at `src/wcmodel`, with acquisition scripts in
  `data_acquisition/`. Install with `pip install -e .`. `notebook.ipynb` is split into
  `notebooks/01_wc_forecast.ipynb`, carrying the methodology and results, and
  `notebooks/02_wc_simulation.ipynb`, which plays a single sampled tournament.
- Added validation by rolling-origin backtest across 17 retraining points and 2,931 held-out
  international matches, scored against an Elo-only baseline.
- Added a calibration check to the backtest, grouping every held-out prediction by size and
  comparing it to how often those outcomes happened.
- Added the ranked probability score alongside log loss, which scores the three match outcomes
  as ordered rather than as unrelated categories.
- Added `PARAMETERS.md`, documenting every parameter, what it does, and how it was chosen.
- Added charts: knockout bracket, model against market, scoreline distribution, attack against
  defence, and per-match heatmaps.
