"""
Reports what the weighting actually does, rather than what the constants say it does.

Two tables. The first is how much of the total match weight each competition carries, which is
not the same as its multiplier, since a competition with a large multiplier and few fixtures
contributes little. The second splits a team's final rating into the four inputs it is built
from, by how much each moves the rating across the 48 finalists, which is not the same as the
weight each is given in the blend.

Usage: python -m wcmodel.weights
"""

import numpy as np
import pandas as pd

from wcmodel.team_data import *


# Competitions are grouped so the table reads as seven rows rather than fifty.
COMPETITION_GROUPS = [('Friendly', lambda t: t == 'Friendly'),
                      ('Nations League', lambda t: 'Nations League' in t),
                      ('World Cup', lambda t: t == 'FIFA World Cup'),
                      ('World Cup qualification', lambda t: t == 'FIFA World Cup qualification'),
                      ('Continental qualification',
                       lambda t: 'qualification' in t and t in TOURNAMENT_WEIGHTS),
                      ('Continental finals', lambda t: t in TOURNAMENT_WEIGHTS)]


def windowed_results():

    """
    The raw fixtures inside the window, with their weights, keeping the competition column.

    build_team_match_data drops the competition once it has used it, so the window is rebuilt
    here the same way to keep it.

    Returns
    -------
    pd.DataFrame
        One row per fixture, with tournament, importance and weight columns.
    """

    results = load_csv('results.csv')
    results = results[(results['date'] >= START_DATE) & (results['date'] <= FREEZE_DATE) &
                      (results['home_score'].notna())].copy()
    results[['home_team', 'away_team']] = (results[['home_team', 'away_team']]
                                           .replace(CHANGED_TEAM_NAMES))

    eligible = set(load_csv(ELIGIBILITY, index_col = 'team').index)
    results = results[results['home_team'].isin(eligible) & results['away_team'].isin(eligible)]

    results['importance'] = results['tournament'].map(lambda t: TOURNAMENT_WEIGHTS.get(t, DEFAULT))
    results['weight'] = match_weight(results)

    return results


def competition_table():

    """
    How much of the model's total match weight each kind of competition carries.

    Returns
    -------
    pd.DataFrame
        One row per competition group, sorted by share of total weight.
    """

    results = windowed_results()

    def group(tournament):
        for name, test in COMPETITION_GROUPS:
            if test(tournament):
                return name
        return 'Other'

    results['group'] = results['tournament'].map(group)

    table = results.groupby('group').agg(importance = ('importance', 'max'),
                                         matches = ('weight', 'size'),
                                         total = ('weight', 'sum'))

    table['vs friendly'] = table['importance'] / FRIENDLY
    table['share of matches'] = table['matches'] / table['matches'].sum()
    table['share of weight'] = table['total'] / table['total'].sum()

    return (table[['importance', 'vs friendly', 'matches', 'share of matches', 'share of weight']]
            .sort_values('share of weight', ascending = False))


def contribution_table(df = None):

    """
    How much each rating input moves the final ratings, against the weight it is given.

    The blend is a weighted geometric mean, so a component's share of the spread depends on how
    far that component separates teams as well as on its weight. Each share is the covariance of
    that component with the total, over the variance of the total, so the four sum to one.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Match data from build_team_match_data. Built here if not given.

    Returns
    -------
    pd.DataFrame
        One row per component, with its weight and its share of the attack and defence ratings.
    """

    if df is None:
        df = build_team_match_data()

    teams = sorted(set(df['team']) | set(df['opponent']))
    team_index = {team : i for i, team in enumerate(teams)}

    solved_attack, solved_defence, sum_weight = solve_team_ratings(df, teams, team_index)
    raw_scored, raw_conceded = calc_raw_rates(df, teams, team_index)
    elo_prior = calc_avg_elo(df, teams, team_index)
    squad_value = calc_squad_value(teams)

    finalists = np.array([team in get_wc_teams() for team in teams])

    def shares(solved, raw, prior, value_sign):

        # Mirrors blend_ratings, which recentres both other views on the solved scale first.
        solved = np.where(solved > 0, solved, np.average(solved[solved > 0],
                                                         weights = sum_weight[solved > 0]))
        raw = np.where(raw > 0, raw, np.average(raw, weights = sum_weight))

        log_solved = np.log(solved)
        log_raw = (np.log(raw) - np.average(np.log(raw), weights = sum_weight)
                   + np.average(log_solved, weights = sum_weight))
        log_prior = prior * log_solved.std() + np.average(log_solved, weights = sum_weight)

        parts = {'solved ratings' : (1 - RAW_WEIGHT - ELO_WEIGHT) * log_solved,
                 'raw goal rates' : RAW_WEIGHT * log_raw,
                 'Elo prior' : ELO_WEIGHT * log_prior,
                 'squad value' : value_sign * VALUE_WEIGHT * squad_value}

        total = sum(parts.values())[finalists]

        return {name: np.cov(part[finalists], total, bias = True)[0, 1] / np.var(total)
                for name, part in parts.items()}

    attack = shares(solved_attack, raw_scored, elo_prior, 1)
    defence = shares(solved_defence, raw_conceded, -elo_prior, -1)

    return pd.DataFrame({'weight' : {'solved ratings' : 1 - RAW_WEIGHT - ELO_WEIGHT,
                                     'raw goal rates' : RAW_WEIGHT,
                                     'Elo prior' : ELO_WEIGHT,
                                     'squad value' : VALUE_WEIGHT},
                         'share of attack' : attack,
                         'share of defence' : defence})


def run():

    """
    Prints both tables.
    """

    percent = '{:.1%}'.format

    competitions = competition_table()
    print('MATCH WEIGHT BY COMPETITION')
    print(competitions.to_string(formatters = {'vs friendly' : '{:.1f}x'.format,
                                               'share of matches' : percent,
                                               'share of weight' : percent}))

    print()
    print('WHERE A RATING COMES FROM  (48 finalists)')
    print(contribution_table().to_string(formatters = {'weight' : '{:.2f}'.format,
                                                       'share of attack' : percent,
                                                       'share of defence' : percent}))


if __name__ == '__main__':
    run()
