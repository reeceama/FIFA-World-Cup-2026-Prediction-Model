"""
Rolling-origin backtest. Train on everything up to a date, predict the matches that follow,
score them, move the date forward and repeat. Ratings and engine constants are refit inside
every fold, so no fixture is ever scored by a model that saw it.

The model is scored against an Elo-only baseline, which is the same pipeline with the rating
blend cut back to the Elo prior, and against always backing the home team.

Usage: python -m wcmodel.backtest
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

from wcmodel.fit_mle import build_team_match_data
from wcmodel.team_data import *


RETRAINING_POINTS = 17
INITIAL_FRACTION = 0.45  # first 45% of fixtures is training only

SHIPPED = (SHRINKAGE, RAW_WEIGHT, ELO_WEIGHT, VALUE_WEIGHT, SCHEDULE_WEIGHT)
ELO_ONLY = (SHRINKAGE, 0.0, 1.0, 0.0, 0.0)

MAX_GOALS = 9

CALIBRATION_EDGES = np.array([0, .1, .2, .3, .4, .5, .6, .7, 1.01])


def build_ratings(training, weights, teams, team_index, squad_value):

    """
    Builds ratings from one fold's training data, using the given set of weights.

    Parameters
    ----------
    training : pd.DataFrame
        Fixtures up to this fold's cut date.
    weights : tuple of float
        Shrinkage, raw, Elo, value and schedule weights. SHIPPED or ELO_ONLY.
    teams : list of str
        Every team in the window, sorted.
    team_index : dict of {str : int}
        Team name to its position in teams.
    squad_value : np.ndarray
        Standardised log squad value, built once outside the loop.

    Returns
    -------
    tuple of np.ndarray
        Attack and defence, indexed by position in teams.
    """

    shrinkage, raw_weight, elo_weight, value_weight, schedule_weight = weights

    prior = calc_avg_elo(training, teams, team_index)
    schedule = calc_relative_schedule(training, teams, team_index)
    solved_attack, solved_defence, sum_weight = solve_team_ratings(training, teams, team_index)
    raw_scored, raw_conceded = calc_raw_rates(training, teams, team_index)

    attack = blend_ratings(solved_attack, raw_scored, prior,
                           raw_weight, elo_weight, sum_weight) ** shrinkage
    defence = blend_ratings(solved_defence, raw_conceded, -prior,
                            raw_weight, elo_weight, sum_weight) ** shrinkage
    attack, defence = apply_schedule_adjustment(attack, defence, schedule, schedule_weight)

    return attack * np.exp(value_weight * squad_value), defence * np.exp(-value_weight * squad_value)


def engine_nll(parameters, attack, defence, fixtures, team_index):

    """
    Negative log likelihood of the scorelines, used to refit the engine in each fold.

    Parameters
    ----------
    parameters : sequence of float
        Rho and the goal baseline, in that order.
    attack, defence : np.ndarray
        This fold's ratings.
    fixtures : pd.DataFrame
        This fold's training fixtures, one row per match.
    team_index : dict of {str : int}
        Team name to its position in teams.

    Returns
    -------
    float
        Negative log likelihood, or 1e12 where the Dixon-Coles correction is invalid.
    """

    rho, g = parameters

    home = fixtures['team'].map(team_index).to_numpy()
    away = fixtures['opponent'].map(team_index).to_numpy()
    home_goals = fixtures['goals_scored'].to_numpy(int)
    away_goals = fixtures['goals_conceded'].to_numpy(int)

    x_home, x_away = attack[home] * defence[away] * g, attack[away] * defence[home] * g
    if not np.all(np.isfinite(x_home) & np.isfinite(x_away) & (x_home > 0) & (x_away > 0)):
        return 1e12

    tau = np.ones(len(home_goals))
    nil_nil = (home_goals == 0) & (away_goals == 0)
    nil_one = (home_goals == 0) & (away_goals == 1)
    one_nil = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)

    tau[nil_nil] = 1 - (x_home * x_away * rho)[nil_nil]
    tau[nil_one] = 1 + (x_home * rho)[nil_one]
    tau[one_nil] = 1 + (x_away * rho)[one_nil]
    tau[one_one] = 1 - rho

    if np.any(tau <= 0):
        return 1e12

    return -(stats.poisson.logpmf(home_goals, x_home) +
             stats.poisson.logpmf(away_goals, x_away) + np.log(tau)).sum()


def outcome_probabilities(x_home, x_away, rho):

    """
    Win, draw and loss probabilities for one fixture.

    Parameters
    ----------
    x_home, x_away : float
        Expected goals for each side.
    rho : float
        Dixon-Coles correction for this fold.

    Returns
    -------
    np.ndarray
        Home win, draw and away win, summing to 1.
    """

    home = stats.poisson.pmf(np.arange(MAX_GOALS), x_home)
    away = stats.poisson.pmf(np.arange(MAX_GOALS), x_away)
    matrix = np.outer(home, away)

    matrix[0, 0] *= 1 - x_home * x_away * rho
    matrix[0, 1] *= 1 + x_home * rho
    matrix[1, 0] *= 1 + x_away * rho
    matrix[1, 1] *= 1 - rho

    matrix = np.clip(matrix, 0, None)
    matrix = matrix / matrix.sum()

    return np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])


def score_fold(attack, defence, fixtures, rho, g, team_index):

    """
    Scores one fold's held-out fixtures.

    Parameters
    ----------
    attack, defence : np.ndarray
        This fold's ratings.
    fixtures : pd.DataFrame
        Held-out matches, one row each.
    rho, g : float
        Engine constants refitted inside this fold.
    team_index : dict of {str : int}
        Team name to its position in teams.

    Returns
    -------
    tuple of np.ndarray
        Per-match log loss, whether the most likely outcome happened, the outcome that did
        happen as 0, 1 or 2, and the full probabilities for the calibration table.
    """

    home = fixtures['team'].map(team_index).to_numpy()
    away = fixtures['opponent'].map(team_index).to_numpy()
    home_goals = fixtures['goals_scored'].to_numpy(int)
    away_goals = fixtures['goals_conceded'].to_numpy(int)

    losses, hits, outcomes, predictions = [], [], [], []
    for i in range(len(fixtures)):
        probabilities = outcome_probabilities(attack[home[i]] * defence[away[i]] * g,
                                              attack[away[i]] * defence[home[i]] * g, rho)
        outcome = 0 if home_goals[i] > away_goals[i] else (1 if home_goals[i] == away_goals[i] else 2)

        losses.append(-np.log(max(probabilities[outcome], 1e-12)))
        hits.append(int(probabilities.argmax()) == outcome)
        outcomes.append(outcome)
        predictions.append(probabilities)

    return np.array(losses), np.array(hits), np.array(outcomes), np.array(predictions)


def ranked_probability_score(predictions, outcomes):

    """
    Ranked probability score, which respects the ordering of home win, draw and away win.

    Log loss and Brier treat the three outcomes as unrelated categories, so calling a home win
    when it finished a draw costs the same as calling one when it finished an away win. RPS
    compares cumulative probabilities instead, so being wrong by one step costs less than being
    wrong by two. Lower is better.

    Parameters
    ----------
    predictions : np.ndarray
        One row per match, holding the probability given to each outcome, in order.
    outcomes : np.ndarray
        Which outcome happened in each match, as 0, 1 or 2.

    Returns
    -------
    np.ndarray
        Score per match.
    """

    happened = np.zeros_like(predictions)
    happened[np.arange(len(outcomes)), outcomes] = 1

    predicted = np.cumsum(predictions, axis = 1)[:, :-1]
    actual = np.cumsum(happened, axis = 1)[:, :-1]

    return ((predicted - actual) ** 2).sum(axis = 1) / (predictions.shape[1] - 1)


def calibration_table(predictions, outcomes, edges = CALIBRATION_EDGES):

    """
    Groups every predicted probability by size and compares each group's mean prediction to how
    often those outcomes actually happened.

    A calibrated forecast sits on the diagonal, predicting 70% for things that happen 70% of
    the time.

    Parameters
    ----------
    predictions : np.ndarray
        One row per match, holding the probability given to each outcome.
    outcomes : np.ndarray
        Which outcome happened in each match, as 0, 1 or 2.
    edges : np.ndarray, default CALIBRATION_EDGES
        Boundaries of the probability groups.

    Returns
    -------
    pd.DataFrame
        One row per group, with the mean prediction, how often it happened, and the gap.
    """

    happened = np.zeros_like(predictions)
    happened[np.arange(len(outcomes)), outcomes] = 1

    predicted, happened = predictions.ravel(), happened.ravel()
    bucket = np.digitize(predicted, edges) - 1

    rows = []
    for i in range(len(edges) - 1):
        inside = bucket == i
        if inside.sum():
            rows.append({'range' : f'{edges[i]:.0%}-{min(edges[i + 1], 1):.0%}',
                         'matches' : int(inside.sum()),
                         'predicted' : predicted[inside].mean(),
                         'observed' : happened[inside].mean()})

    table = pd.DataFrame(rows)
    table['gap'] = table['observed'] - table['predicted']

    return table


def run():

    """
    Runs every fold, then prints accuracy, log loss, the baseline comparison and calibration.
    """

    df = build_team_match_data()
    teams = sorted(set(df['team']) | set(df['opponent']))
    team_index = {team : i for i, team in enumerate(teams)}
    squad_value = calc_squad_value(teams)

    home_rows = df[df['is_home']].reset_index(drop = True)
    cuts = [int(len(home_rows) * (INITIAL_FRACTION +
                                  (1 - INITIAL_FRACTION) * k / RETRAINING_POINTS))
            for k in range(RETRAINING_POINTS + 1)]

    results = {'model' : [[], [], []], 'elo-only' : [[], [], []]}
    outcomes, predictions = [], []

    for k in range(RETRAINING_POINTS):
        training = df[df['date'] <= home_rows.iloc[cuts[k] - 1]['date']]
        validation = home_rows.iloc[cuts[k]:cuts[k + 1]]
        if len(validation) < 20:
            continue

        for label, weights in (('model', SHIPPED), ('elo-only', ELO_ONLY)):
            attack, defence = build_ratings(training, weights, teams, team_index, squad_value)
            fitted = optimize.minimize(engine_nll, [-0.10, 1.05], method = 'Nelder-Mead',
                                       args = (attack, defence, training[training['is_home']],
                                               team_index))
            losses, hits, outcome, prediction = score_fold(attack, defence, validation,
                                                           *fitted.x, team_index)
            results[label][0].append(losses)
            results[label][1].append(hits)
            results[label][2].append(ranked_probability_score(prediction, outcome))
            if label == 'model':
                outcomes.append(outcome)
                predictions.append(prediction)

        print(f'  point {k + 1:2d}  train {len(training):6,}  validate {len(validation):4,}  '
              f'to {str(validation["date"].max())[:10]}', flush = True)

    outcomes = np.concatenate(outcomes)
    predictions = np.concatenate(predictions)
    base_rates = np.bincount(outcomes, minlength = 3) / len(outcomes)

    print()
    print(f'ROLLING-ORIGIN BACKTEST  ({RETRAINING_POINTS} retraining points, '
          f'{len(outcomes):,} held-out matches)')
    print()

    for label in ('model', 'elo-only'):
        losses = np.concatenate(results[label][0])
        hits = np.concatenate(results[label][1])
        rps = np.concatenate(results[label][2])
        print(f'  {label:<26} accuracy {hits.mean():.1%}   log loss {losses.mean():.4f}   '
              f'RPS {rps.mean():.4f}')

    constant = np.tile(base_rates, (len(outcomes), 1))
    print(f'  {"always back the home team":<26} accuracy {base_rates[0]:.1%}   '
          f'log loss {-np.sum(base_rates * np.log(base_rates)):.4f}   '
          f'RPS {ranked_probability_score(constant, outcomes).mean():.4f}')

    model = np.concatenate(results['model'][0])
    baseline = np.concatenate(results['elo-only'][0])
    difference = baseline - model
    standard_error = difference.std(ddof = 1) / np.sqrt(len(difference))

    print()
    print(f'  model beats the Elo-only baseline by {difference.mean():.4f} nats per match '
          f'(se {standard_error:.4f}, t = {difference.mean() / standard_error:.2f})')

    table = calibration_table(predictions, outcomes)
    error = np.average(table['gap'].abs(), weights = table['matches'])

    print()
    print('CALIBRATION  (every predicted probability, grouped by size)')
    print()
    print(table.to_string(index = False,
                          formatters = {'predicted' : '{:.1%}'.format,
                                        'observed' : '{:.1%}'.format,
                                        'gap' : '{:+.1%}'.format}))
    print()
    print(f'  mean absolute gap {error:.1%}, largest {table["gap"].abs().max():.1%}')


if __name__ == '__main__':
    run()
