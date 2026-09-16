"""
Fits and reports the model's parameters, in two stages.

Stage 1 is the rating weights, chosen on held-out data by grid search inside a nested
chronological split, since a likelihood always prefers less regularisation. Stage 2 is the
engine constants, rho and the goal baseline, fitted by maximum likelihood on the fixtures
played between the 48 finalists. Home advantage is fitted separately on the full window.

This reports parameters, it does not set them. The values the model runs on live in team_data
and match, so refitting does not change a forecast on its own. Profile likelihood curves and a
summary are written to data/calibration/.

Usage: python -m wcmodel.fit_mle
"""

import os

import json

import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy.stats import poisson

from wcmodel.team_data import *


# ---------- Configuration ----------


VALIDATION_FRACTION = 0.20

COARSE_SHRINKAGE = np.round(np.arange(0.60, 1.21, 0.05), 3)
COARSE_RAW = np.round(np.arange(0.00, 0.61, 0.05), 3)
COARSE_ELO = np.round(np.arange(0.00, 0.61, 0.05), 3)

SHRINKAGE_BOUNDS = (0.50, 1.30)
RAW_BOUNDS = (0.00, 0.60)
ELO_BOUNDS = (0.00, 0.60)

ENGINE_BOUNDS = [(-0.20, 0.20), (0.50, 2.00)]

LR_THRESHOLD = 3.841

PROFILE_GRIDS = {
    'rho' : np.round(np.arange(-0.20, 0.201, 0.005), 4),
    'g' : np.round(np.arange(0.80, 1.401, 0.01), 3),
    'home_advantage' : np.round(np.arange(1.00, 1.351, 0.005), 3),
}

STAGE_ONE_GRIDS = {
    'shrinkage' : np.round(np.arange(0.70, 1.001, 0.02), 3),
    'raw_weight' : np.round(np.arange(0.00, 0.241, 0.02), 3),
    'elo_weight' : np.round(np.arange(0.00, 0.501, 0.05), 3),
    'schedule_weight' : np.round(np.arange(0.00, 0.081, 0.01), 3),
    'value_weight' : np.round(np.arange(0.00, 0.151, 0.01), 3),
}

HOME_BOUNDS = [(0.8, 1.8), (0.5, 2.0)]

OUTPUT_PATH = os.path.join(CALIBRATION_DIR, 'calibration_summary.json')
PROFILE_PATH = os.path.join(CALIBRATION_DIR, '{parameter}_profile.csv')


# ---------- Chronological Splits ----------


def chronological_split(df, fraction = VALIDATION_FRACTION):

    """
    Splits by date, training first, so nothing from after the cut reaches a training row.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data, already sorted by date.
    fraction : float, default VALIDATION_FRACTION
        Share of fixtures held out.

    Returns
    -------
    tuple
        Training frame, validation frame, and the date the split falls on.
    """

    split = int(len(df) * (1 - fraction))

    return df.iloc[:split].copy(), df.iloc[split:].copy(), df.iloc[split]['date']

def nested_split(df):

    """
    Two chronological splits, one inside the other.

    The inner split picks the rating weights and the outer one is held back to judge them, so
    the data that chose a parameter is never the data that scores it.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data.

    Returns
    -------
    dict
        Inner and outer training and validation frames, with the dates they split on.
    """

    outer_training, outer_validation, outer_date = chronological_split(df)
    inner_training, inner_validation, inner_date = chronological_split(outer_training)

    return {'inner_training' : inner_training, 'inner_validation' : inner_validation,
            'outer_training' : outer_training, 'outer_validation' : outer_validation,
            'inner_date' : inner_date, 'outer_date' : outer_date}


# ---------- Stage 1 ----------


def goal_nll(attack, defence, df, team_index):

    """
    Negative log likelihood of the goals scored, under a plain Poisson with these ratings.

    Scores a candidate set of rating weights. Expected goals are rescaled to the observed mean
    first, so this measures how well the ratings separate teams rather than whether the goal
    level is right, which the goal baseline handles.

    Parameters
    ----------
    attack, defence : np.ndarray
        Candidate ratings, indexed by position in teams.
    df : pd.DataFrame
        Fixtures to score.
    team_index : dict of {str : int}
        Team name to its position in teams.

    Returns
    -------
    float
        Negative log likelihood, or inf if the ratings are degenerate.
    """

    team_indices = df['team'].map(team_index).to_numpy()
    opponent_indices = df['opponent'].map(team_index).to_numpy()
    goals_scored = df['goals_scored'].to_numpy(float)

    expected_goals = attack[team_indices] * defence[opponent_indices]
    if not np.all(np.isfinite(expected_goals)) or expected_goals.mean() <= 0:
        return np.inf

    expected_goals = expected_goals * goals_scored.mean() / expected_goals.mean()
    log_probability = poisson.logpmf(goals_scored, expected_goals)

    return float(-log_probability.sum()) if np.all(np.isfinite(log_probability)) else np.inf

def refine(value, step, bounds, span = 4):

    """
    Builds a finer grid around the coarse search's answer, clipped to the parameter's bounds.

    Parameters
    ----------
    value : float
        The coarse winner to search around.
    step : float
        Spacing of the fine grid.
    bounds : tuple of float
        Lowest and highest value allowed.
    span : int, default 4
        How many steps either side to cover.

    Returns
    -------
    np.ndarray
        The fine grid, sorted and deduplicated.
    """

    grid = np.round(np.arange(value - span * step, value + (span + 0.5) * step, step), 4)
    grid = grid[(grid >= bounds[0] - 1e-9) & (grid <= bounds[1] + 1e-9)]

    return np.unique(np.round(np.clip(grid, *bounds), 4))

def search_stage_one(split, teams, team_index, elo_prior, schedule_faced, squad_value, grids,
                     schedule_weight = SCHEDULE_WEIGHT, home_advantage = 1.0):

    """
    Grid searches the rating weights on the inner validation split.

    Every combination rebuilds the ratings from the inner training data and scores them on
    fixtures the build never saw. Fitting these by likelihood instead would set shrinkage to 1
    and the priors to 0 every time.

    Parameters
    ----------
    split : dict
        From nested_split.
    teams : list of str
        Every team in the window, sorted.
    team_index : dict of {str : int}
        Team name to its position in teams.
    elo_prior, schedule_faced, squad_value : np.ndarray
        Rating inputs built from the inner training data.
    grids : tuple of np.ndarray
        Values to try for shrinkage, raw weight and Elo weight.
    schedule_weight : float, default SCHEDULE_WEIGHT
        Held fixed rather than searched.
    home_advantage : float, default 1.0
        Passed through to the rating build.

    Returns
    -------
    dict
        The best combination and its validation likelihood.
    """

    inner_training, inner_validation = split['inner_training'], split['inner_validation']
    raw_scored, raw_conceded = calc_raw_rates(inner_training, teams, team_index, home_advantage)
    solved_attack, solved_defence, sum_weight = solve_team_ratings(inner_training, teams,
                                                                   team_index, home_advantage)

    shrinkage_grid, raw_grid, elo_grid = grids
    best = None

    for raw_weight in raw_grid:
        for elo_weight in elo_grid:

            if raw_weight + elo_weight > 0.80:
                continue

            blended_attack = blend_ratings(solved_attack, raw_scored, elo_prior,
                                           raw_weight, elo_weight, sum_weight)
            blended_defence = blend_ratings(solved_defence, raw_conceded, -elo_prior,
                                            raw_weight, elo_weight, sum_weight)

            for shrinkage in shrinkage_grid:

                attack, defence = apply_schedule_adjustment(blended_attack ** shrinkage,
                                                            blended_defence ** shrinkage,
                                                            schedule_faced, schedule_weight)
                attack, defence = apply_squad_value(attack, defence, squad_value, VALUE_WEIGHT)

                nll = goal_nll(attack, defence, inner_validation, team_index)

                if best is None or nll < best['inner_validation_nll']:
                    best = {'shrinkage' : round(float(shrinkage), 4),
                            'raw_weight' : round(float(raw_weight), 4),
                            'elo_weight' : round(float(elo_weight), 4),
                            'inner_validation_nll' : float(nll)}

    return best

def fit_stage_one(split, teams, team_index, schedule_weight = SCHEDULE_WEIGHT,
                  home_advantage = 1.0):

    """
    Chooses the rating weights, coarse then fine, and rebuilds on the outer training window.

    Parameters
    ----------
    split : dict
        From nested_split.
    teams : list of str
        Every team in the window, sorted.
    team_index : dict of {str : int}
        Team name to its position in teams.
    schedule_weight : float, default SCHEDULE_WEIGHT
        Held fixed rather than searched.
    home_advantage : float, default 1.0
        Passed through to the rating build.

    Returns
    -------
    dict
        The selected weights, with the likelihood on the inner validation and outer training
        data, and the ratings rebuilt with them.
    """

    inner_training = split['inner_training']
    inner_prior = calc_avg_elo(inner_training, teams, team_index)
    inner_schedule = calc_relative_schedule(inner_training, teams, team_index)
    squad_value = calc_squad_value(teams)

    coarse = search_stage_one(split, teams, team_index, inner_prior, inner_schedule, squad_value,
                              (COARSE_SHRINKAGE, COARSE_RAW, COARSE_ELO),
                              schedule_weight, home_advantage)

    fine = search_stage_one(split, teams, team_index, inner_prior, inner_schedule, squad_value,
                            (refine(coarse['shrinkage'], 0.01, SHRINKAGE_BOUNDS),
                             refine(coarse['raw_weight'], 0.01, RAW_BOUNDS),
                             refine(coarse['elo_weight'], 0.01, ELO_BOUNDS)),
                            schedule_weight, home_advantage)

    best = fine if fine['inner_validation_nll'] <= coarse['inner_validation_nll'] else coarse
    best['schedule_weight'] = schedule_weight
    best['value_weight'] = VALUE_WEIGHT
    best['home_advantage_applied'] = round(float(home_advantage), 4)

    # Rebuilt on the full outer-training window with the selected parameters.
    outer_training = split['outer_training']
    outer_prior = calc_avg_elo(outer_training, teams, team_index)
    outer_schedule = calc_relative_schedule(outer_training, teams, team_index)

    solved_attack, solved_defence, sum_weight = solve_team_ratings(outer_training, teams,
                                                                   team_index, home_advantage)
    raw_scored, raw_conceded = calc_raw_rates(outer_training, teams, team_index, home_advantage)

    attack, defence = apply_schedule_adjustment(
        blend_ratings(solved_attack, raw_scored, outer_prior, best['raw_weight'],
                      best['elo_weight'], sum_weight) ** best['shrinkage'],
        blend_ratings(solved_defence, raw_conceded, -outer_prior, best['raw_weight'],
                      best['elo_weight'], sum_weight) ** best['shrinkage'],
        outer_schedule, schedule_weight)

    attack, defence = apply_squad_value(attack, defence, squad_value, VALUE_WEIGHT)

    best['outer_train_nll'] = goal_nll(attack, defence, outer_training, team_index)

    return best, attack, defence


# ---------- Stage 2 ----------


def match_frame(df, wc_teams):

    """
    Fixtures played between two of the 48 finalists, one row per match.

    These are the matches closest to the ones being simulated, so the engine constants are
    fitted on them rather than on the whole international calendar. Only 43 are World Cup
    games, the rest are friendlies, qualifiers and continental ties between finalists.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data.
    wc_teams : set of str
        The 48 finalists, from get_wc_teams.

    Returns
    -------
    pd.DataFrame
        One row per fixture, taken from the home side's perspective.
    """

    fixtures = df[df['team'].isin(wc_teams) & df['opponent'].isin(wc_teams)]

    return fixtures[fixtures['is_home']].reset_index(drop = True)

def engine_nll(parameters, attack, defence, fixtures, team_index, home_advantage = 1.0):

    """
    Negative log likelihood of the scorelines under the Dixon-Coles corrected Poisson.

    The correction only holds while every tau stays positive, so a parameter set that breaks
    it returns a large penalty rather than failing.

    Parameters
    ----------
    parameters : sequence of float
        Rho and the goal baseline, in that order.
    attack, defence : np.ndarray
        Fixed ratings, indexed by position in teams.
    fixtures : pd.DataFrame
        One row per match, from match_frame.
    team_index : dict of {str : int}
        Team name to its position in teams.
    home_advantage : float, default 1.0
        Applied at non-neutral venues while fitting.

    Returns
    -------
    float
        Negative log likelihood, or 1e12 where the correction is invalid.
    """

    rho, g = parameters

    home_indices = fixtures['team'].map(team_index).to_numpy()
    away_indices = fixtures['opponent'].map(team_index).to_numpy()
    home_goals = fixtures['goals_scored'].to_numpy(int)
    away_goals = fixtures['goals_conceded'].to_numpy(int)

    venue_boost = np.where(~fixtures['neutral'].to_numpy(bool), home_advantage, 1.0)

    x_home = attack[home_indices] * defence[away_indices] * g * venue_boost
    x_away = attack[away_indices] * defence[home_indices] * g / venue_boost

    if not np.all(np.isfinite(x_home) & np.isfinite(x_away) & (x_home > 0) & (x_away > 0)):
        return 1e12

    if not (np.all(1 - x_home * x_away * rho > 0) and np.all(1 + x_home * rho > 0)
            and np.all(1 + x_away * rho > 0) and 1 - rho > 0):
        return 1e12

    tau = np.ones(len(home_goals))
    tau[(home_goals == 0) & (away_goals == 0)] = (1 - x_home * x_away * rho)[(home_goals == 0) & (away_goals == 0)]
    tau[(home_goals == 0) & (away_goals == 1)] = (1 + x_home * rho)[(home_goals == 0) & (away_goals == 1)]
    tau[(home_goals == 1) & (away_goals == 0)] = (1 + x_away * rho)[(home_goals == 1) & (away_goals == 0)]
    tau[(home_goals == 1) & (away_goals == 1)] = 1 - rho

    if np.any(tau <= 0):
        return 1e12

    log_probability = (poisson.logpmf(home_goals, x_home) +
                       poisson.logpmf(away_goals, x_away) + np.log(tau))

    return float(-log_probability.sum()) if np.all(np.isfinite(log_probability)) else 1e12

def fit_stage_two(training, validation, attack, defence, team_index, home_advantage = 1.0):

    """
    Fits rho and the goal baseline by maximum likelihood, holding the ratings fixed.

    Started from three points, since the surface is flat enough near the optimum that one start
    can settle early.

    Parameters
    ----------
    training, validation : pd.DataFrame
        Fixtures to fit on and to score on. The final fit passes the same frame twice.
    attack, defence : np.ndarray
        Ratings from Stage 1.
    team_index : dict of {str : int}
        Team name to its position in teams.
    home_advantage : float, default 1.0
        Applied at non-neutral venues while fitting.

    Returns
    -------
    dict
        The fitted rho and g, and the likelihood on both frames.
    """

    best = None

    for start in ([-0.11, 1.05], [-0.09, 1.00], [-0.02, 1.10]):

        result = optimize.minimize(engine_nll, start,
                                   args = (attack, defence, training, team_index, home_advantage),
                                   method = 'L-BFGS-B', bounds = ENGINE_BOUNDS)

        if best is None or result.fun < best.fun:
            best = result

    rho, g = best.x

    return {'rho' : round(float(rho), 4),
            'g' : round(float(g), 4),
            'train_nll' : float(best.fun),
            'home_advantage_applied' : round(float(home_advantage), 4),
            'validation_nll' : float(engine_nll([rho, g], attack, defence,
                                                validation, team_index, home_advantage))}

def profile_parameter(parameter, fitted, attack, defence, fixtures, team_index,
                      home_advantage = 1.0):

    """
    Profile likelihood curve for one engine parameter, and its 95% interval.

    Each grid value is held fixed while the other is refitted, then the deviance against the
    best fit is compared to a chi-squared cutoff. An interval running to the end of the grid is
    reported as unbounded on that side rather than as a bound.

    Parameters
    ----------
    parameter : {'rho', 'g'}
        Which one to profile.
    fitted : dict
        The maximum likelihood fit from fit_stage_two.
    attack, defence : np.ndarray
        Ratings the fit was conditional on.
    fixtures : pd.DataFrame
        Fixtures to evaluate on.
    team_index : dict of {str : int}
        Team name to its position in teams.
    home_advantage : float, default 1.0
        Applied at non-neutral venues.

    Returns
    -------
    tuple
        The curve as a DataFrame, and the interval as a pair, with None where unbounded.
    """

    order = ['rho', 'g']
    held = order.index(parameter)
    free = [i for i in range(len(order)) if i != held]

    minimum = engine_nll([fitted[name] for name in order], attack, defence, fixtures,
                         team_index, home_advantage)

    rows = []

    for value in PROFILE_GRIDS[parameter]:

        def refit(pair):

            parameters = [0.0] * len(order)
            parameters[held] = value
            for slot, number in zip(free, pair):
                parameters[slot] = number

            return engine_nll(parameters, attack, defence, fixtures, team_index, home_advantage)

        best = optimize.minimize(refit, [fitted[order[i]] for i in free],
                                 method = 'L-BFGS-B', bounds = [ENGINE_BOUNDS[i] for i in free])

        likelihood_ratio = 2 * (best.fun - minimum)

        rows.append({parameter : float(value), 'nll' : float(best.fun),
                     'likelihood_ratio' : float(likelihood_ratio),
                     'inside_95_interval' : bool(likelihood_ratio <= LR_THRESHOLD)})

    profile = pd.DataFrame(rows)
    accepted = profile[profile['inside_95_interval']]

    if accepted.empty:
        return profile, None

    grid = PROFILE_GRIDS[parameter]
    low, high = accepted[parameter].min(), accepted[parameter].max()

    # An interval touching either end of the grid is a floor or a ceiling, not a bound.
    return profile, (None if low <= grid[0] else round(float(low), 4),
                     None if high >= grid[-1] else round(float(high), 4))


def profile_stage_one(parameter, shipped, components, fixtures, team_index):

    """
    Likelihood profile for a Stage 1 weight. Each grid value rebuilds the ratings, refits home
    advantage and the engine on top of them, and the deviance is taken against the best value on
    the grid. Unlike the engine parameters these are not maximum likelihood estimates, so the
    interval says which values the training data tolerates, not where an estimator landed.

    Parameters
    ----------
    parameter : str
        Which Stage 1 weight to profile.
    shipped : dict
        The weights the model runs on, held fixed apart from the one being profiled.
    components : tuple
        Rating inputs built once outside the loop, so each grid value only reblends them.
    fixtures : pd.DataFrame
        Fixtures to evaluate on, one row per match.
    team_index : dict of {str : int}
        Team name to its position in teams.

    Returns
    -------
    tuple
        The curve as a DataFrame, and the interval as a pair, with None where unbounded.
    """

    solved_attack, solved_defence, sum_weight, raw_scored, raw_conceded, prior, schedule, value = components

    rows = []

    for candidate in STAGE_ONE_GRIDS[parameter]:

        weights = dict(shipped)
        weights[parameter] = float(candidate)

        attack, defence = apply_schedule_adjustment(
            blend_ratings(solved_attack, raw_scored, prior, weights['raw_weight'],
                          weights['elo_weight'], sum_weight) ** weights['shrinkage'],
            blend_ratings(solved_defence, raw_conceded, -prior, weights['raw_weight'],
                          weights['elo_weight'], sum_weight) ** weights['shrinkage'],
            schedule, weights['schedule_weight'])

        attack, defence = apply_squad_value(attack, defence, value, weights['value_weight'])

        if not (np.all(np.isfinite(attack)) and np.all(np.isfinite(defence))
                and np.all(attack > 0) and np.all(defence > 0)):
            continue

        engine = fit_stage_two(fixtures, fixtures, attack, defence, team_index)
        rows.append({parameter : float(candidate), 'nll' : engine['train_nll'],
                     'rho' : engine['rho'], 'g' : engine['g']})

    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile, None

    profile['likelihood_ratio'] = 2 * (profile['nll'] - profile['nll'].min())
    profile['inside_95_interval'] = profile['likelihood_ratio'] <= LR_THRESHOLD

    accepted = profile[profile['inside_95_interval']]
    grid = STAGE_ONE_GRIDS[parameter]
    low, high = accepted[parameter].min(), accepted[parameter].max()

    return profile, (None if low <= grid[0] else round(float(low), 4),
                     None if high >= grid[-1] else round(float(high), 4))


def fit_home_advantage(df, attack, defence, team_index):

    """
    Fits home advantage and the overall goal level together, on the full window.

    Uses every fixture rather than only those between finalists, since only 3,523 were played
    at a real home venue and restricting further would leave too few.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data.
    attack, defence : np.ndarray
        Final ratings, indexed by position in teams.
    team_index : dict of {str : int}
        Team name to its position in teams.

    Returns
    -------
    dict
        The fitted advantage and level, the 95% interval, how many non-neutral matches it was
        fitted on, and what it gains over assuming no advantage.
    """

    fixtures = df[df['is_home']].reset_index(drop = True)
    home_indices = fixtures['team'].map(team_index).to_numpy()
    away_indices = fixtures['opponent'].map(team_index).to_numpy()
    home_goals = fixtures['goals_scored'].to_numpy(int)
    away_goals = fixtures['goals_conceded'].to_numpy(int)

    played_at_home = ~fixtures['neutral'].to_numpy(bool)

    def negative_log_likelihood(parameters):

        advantage, level = parameters
        venue_boost = np.where(played_at_home, advantage, 1.0)

        x_home = attack[home_indices] * defence[away_indices] * level * venue_boost
        x_away = attack[away_indices] * defence[home_indices] * level / venue_boost

        if not np.all(np.isfinite(x_home) & np.isfinite(x_away) & (x_home > 0) & (x_away > 0)):
            return 1e12

        return float(-(poisson.logpmf(home_goals, x_home) +
                       poisson.logpmf(away_goals, x_away)).sum())

    best = optimize.minimize(negative_log_likelihood, [1.15, 1.0], method = 'L-BFGS-B',
                             bounds = HOME_BOUNDS)

    no_advantage_nll = negative_log_likelihood([1.0, best.x[1]])
    profile, interval = profile_home_advantage(negative_log_likelihood, best)
    profile.to_csv(PROFILE_PATH.format(parameter = 'home_advantage'), index = False)

    return {'home_advantage' : round(float(best.x[0]), 4),
            'level' : round(float(best.x[1]), 4),
            'interval' : interval,
            'non_neutral' : int(played_at_home.sum()),
            'matches' : int(len(fixtures)),
            'gain_over_no_advantage' : round(no_advantage_nll - best.fun, 2)}

def profile_home_advantage(negative_log_likelihood, fitted):

    """
    Profile likelihood curve for home advantage, refitting the goal level at each grid value.

    Parameters
    ----------
    negative_log_likelihood : callable
        The objective from fit_home_advantage, taking advantage and level.
    fitted : OptimizeResult
        The maximum likelihood fit.

    Returns
    -------
    tuple
        The curve as a DataFrame, and the interval as a pair, with None where unbounded.
    """

    grid = PROFILE_GRIDS['home_advantage']
    rows = []

    for advantage in grid:

        refit = optimize.minimize(lambda level: negative_log_likelihood([advantage, level[0]]),
                                  [fitted.x[1]], method = 'L-BFGS-B', bounds = [HOME_BOUNDS[1]])

        likelihood_ratio = 2 * (refit.fun - fitted.fun)

        rows.append({'home_advantage' : float(advantage), 'level' : float(refit.x[0]),
                     'nll' : float(refit.fun), 'likelihood_ratio' : float(likelihood_ratio),
                     'inside_95_interval' : bool(likelihood_ratio <= LR_THRESHOLD)})

    profile = pd.DataFrame(rows)
    accepted = profile[profile['inside_95_interval']]

    if accepted.empty:
        return profile, None

    low, high = accepted['home_advantage'].min(), accepted['home_advantage'].max()

    return profile, (None if low <= grid[0] else round(float(low), 4),
                     None if high >= grid[-1] else round(float(high), 4))


# ---------- Main ----------


def main():

    """
    Runs both stages, writes the profile curves and the summary, then prints the comparison.

    The printed values belong to the configuration this search picked, not the one that ships.
    PARAMETERS.md explains why the two differ.
    """

    df = build_team_match_data()
    teams = sorted(set(df['team']) | set(df['opponent']))
    team_index = {team : i for i, team in enumerate(teams)}
    wc_teams = get_wc_teams()

    split = nested_split(df)
    training, validation = split['outer_training'], split['outer_validation']

    print('TWO-STAGE CALIBRATION  (nested chronological split)')
    print(f'  window          : {START_DATE} to {FREEZE_DATE} ({WINDOW}), decay {DECAY_RATE}')
    print(f'  eligible teams  : {len(teams):,}')
    print(f'  inner split     : {split["inner_date"]}   '
          f'{len(split["inner_training"]):,} / {len(split["inner_validation"]):,} team-matches')
    print(f'  outer split     : {split["outer_date"]}   '
          f'{len(training):,} / {len(validation):,} team-matches')
    print()

    stage_one, attack, defence = fit_stage_one(split, teams, team_index)

    print(f'STAGE 1  schedule {stage_one["schedule_weight"]} (design)  '
          f'value {stage_one["value_weight"]} (design)  shrinkage {stage_one["shrinkage"]}  '
          f'raw {stage_one["raw_weight"]}  elo {stage_one["elo_weight"]}')
    print(f'         shrinkage, raw, elo on inner-validation NLL/match '
          f'{stage_one["inner_validation_nll"] / len(split["inner_validation"]):.6f}'
          f'   outer train {stage_one["outer_train_nll"] / len(training):.6f}')

    wc_training = match_frame(training, wc_teams)
    wc_validation = match_frame(validation, wc_teams)

    stage_two = fit_stage_two(wc_training, wc_validation, attack, defence, team_index)

    print(f'STAGE 2  rho = {stage_two["rho"]}  g = {stage_two["g"]}')
    print(f'         World Cup fixtures: {len(wc_training):,} training / '
          f'{len(wc_validation):,} validation')
    print(f'         train NLL/match {stage_two["train_nll"] / len(wc_training):.6f}'
          f'   validation {stage_two["validation_nll"] / len(wc_validation):.6f}')
    print()

    # Refit on the full pre-tournament window, which is what the forecast runs on.
    full_prior = calc_avg_elo(df, teams, team_index)
    full_schedule = calc_relative_schedule(df, teams, team_index)
    squad_value = calc_squad_value(teams)

    solved_attack, solved_defence, sum_weight = solve_team_ratings(df, teams, team_index)
    raw_scored, raw_conceded = calc_raw_rates(df, teams, team_index)

    final_attack, final_defence = apply_schedule_adjustment(
        blend_ratings(solved_attack, raw_scored, full_prior, stage_one['raw_weight'],
                      stage_one['elo_weight'], sum_weight) ** stage_one['shrinkage'],
        blend_ratings(solved_defence, raw_conceded, -full_prior, stage_one['raw_weight'],
                      stage_one['elo_weight'], sum_weight) ** stage_one['shrinkage'],
        full_schedule, stage_one['schedule_weight'])

    final_attack, final_defence = apply_squad_value(final_attack, final_defence,
                                                    squad_value, VALUE_WEIGHT)

    wc_all = match_frame(df, wc_teams)
    final_engine = fit_stage_two(wc_all, wc_all, final_attack, final_defence, team_index)

    home = fit_home_advantage(df, final_attack, final_defence, team_index)

    bounds = ('not identified' if home['interval'] is None else
              f'[{home["interval"][0]}, {home["interval"][1]}]')
    print(f'  home advantage {home["home_advantage"]} 95% {bounds} on '
          f'{home["non_neutral"]:,} non-neutral matches, worth '
          f'{home["gain_over_no_advantage"]:.0f} units (reported, not applied)')

    shipped = {'shrinkage' : SHRINKAGE, 'raw_weight' : RAW_WEIGHT, 'elo_weight' : ELO_WEIGHT,
               'schedule_weight' : SCHEDULE_WEIGHT, 'value_weight' : VALUE_WEIGHT}
    components = (solved_attack, solved_defence, sum_weight, raw_scored, raw_conceded,
                  full_prior, full_schedule, squad_value)

    stage_one_intervals = {}
    for parameter in STAGE_ONE_GRIDS:
        profile, interval = profile_stage_one(parameter, shipped, components, wc_all, team_index)
        stage_one_intervals[parameter] = interval
        profile.to_csv(PROFILE_PATH.format(parameter = parameter), index = False)

    print()
    print('STAGE 1 PROFILES (shipped value against the training likelihood)')
    for parameter, interval in stage_one_intervals.items():
        low = 'open' if interval is None or interval[0] is None else f'{interval[0]}'
        high = 'open' if interval is None or interval[1] is None else f'{interval[1]}'
        print(f'  {parameter:<17}{shipped[parameter]:<8}95% [{low}, {high}]')

    intervals = {}
    for parameter in ('rho', 'g'):
        profile, interval = profile_parameter(parameter, final_engine, final_attack,
                                              final_defence, wc_all, team_index)
        intervals[parameter] = interval
        profile.to_csv(PROFILE_PATH.format(parameter = parameter), index = False)

    print('FINAL REFIT (full pre-tournament window)')
    print(f'  rho = {final_engine["rho"]}  g = {final_engine["g"]}')

    for parameter, interval in intervals.items():

        if interval is None:
            bounds = 'not identified over the profile grid'
        else:
            low = 'unbounded' if interval[0] is None else f'{interval[0]}'
            high = 'unbounded' if interval[1] is None else f'{interval[1]}'
            bounds = f'[{low}, {high}]'

        print(f'  95% profile interval, {parameter:<13}: {bounds}')

    ratings = pd.DataFrame({
        'attack' : final_attack,
        'defence' : final_defence,
        'raw_scored' : raw_scored,
        'raw_conceded' : raw_conceded,
        'stage_1a_attack' : solved_attack,
        'stage_1a_defence' : solved_defence,
        'schedule_faced' : full_schedule,
        'squad_value' : squad_value,
        'weight' : sum_weight,
    }, index = teams)

    ratings = ratings.loc[sorted(wc_teams & set(teams))]
    ratings['attack_rank'] = ratings['attack'].rank(ascending = False).astype(int)
    ratings['defence_rank'] = ratings['defence'].rank().astype(int)
    ratings['overall_rank'] = (ratings['attack'] / ratings['defence']).rank(ascending = False).astype(int)

    print(f'  correlation, raw scored vs final attack   : '
          f'{ratings["raw_scored"].corr(ratings["attack"]):.3f}')
    print(f'  correlation, raw conceded vs final defence: '
          f'{ratings["raw_conceded"].corr(ratings["defence"]):.3f}')
    print()
    print(ratings.sort_values('overall_rank')
          [['attack', 'attack_rank', 'defence', 'defence_rank']].head(16).round(3).to_string())

    summary = {
        'window' : {'start' : START_DATE, 'freeze' : FREEZE_DATE,
                    'length' : WINDOW, 'decay_rate' : DECAY_RATE},
        'eligible_teams' : len(teams),
        'world_cup_teams' : len(wc_teams),
        'inner_split_date' : str(split['inner_date']),
        'outer_split_date' : str(split['outer_date']),
        'stage_1' : stage_one,
        'stage_2' : stage_two,
        'final_engine' : final_engine,
        'profile_intervals' : intervals,
        'stage_one_intervals' : stage_one_intervals,
        'home' : home,
    }

    with open(OUTPUT_PATH, 'w') as file:
        json.dump(summary, file, indent = 2)

    print()
    print(f'Saved {OUTPUT_PATH} and the profile curves')


if __name__ == '__main__':
    main()