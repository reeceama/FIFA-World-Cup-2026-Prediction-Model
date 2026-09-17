"""
Scores the published forecast against what actually happened at the 2026 World Cup.

Nothing here feeds back into the model. It exists so every number in the README's "How it
performed" section can be reproduced.

Group matches are scored on 90 minute probabilities, since a group match can end in a draw.
Knockout ties are scored on who advanced, which is the end of the tie including extra time and
shootouts, because that is what results.csv records.

Usage: python -m wcmodel.evaluate
"""

import numpy as np
import pandas as pd
import scipy.optimize as optimize

from wcmodel.backtest import (build_ratings, engine_nll, ranked_probability_score,
                              ELO_ONLY)
from wcmodel.match import MatchSimulator
from wcmodel.team_data import *


# results.csv has no round column and no tournament year, so dates do both jobs.
TOURNAMENT_START = '2026-06-11'
TOURNAMENT_END = '2026-07-19'
KNOCKOUT_START = '2026-06-28'

RUN_DIR = '100k_monte_carlo'

# host_boost only looks at the country behind a venue, so any venue in that country will do.
COUNTRY_VENUE = {'Mexico' : 'Estadio Azteca, Mexico City',
                 'Canada' : 'BMO Field, Toronto',
                 'United States' : 'MetLife Stadium, East Rutherford'}

ROUNDS = [('Round of 32', 16), ('Round of 16', 8), ('Quarter-final', 4),
          ('Semi-final', 2), ('Third-place playoff', 1), ('Final', 1)]


class BaselineSimulator(MatchSimulator):

    """
    The same engine with Elo-only ratings and its own fitted constants.

    Both sides of the comparison run through identical code, so any difference comes from the
    rating blend rather than the engine.
    """

    def __init__(self, team_stats, rho, g):

        super().__init__(team_stats)
        self.rho, self.g = rho, g

    def expected_goals(self, home_team, away_team, venue = None, g = None):

        # Forces this instance's own goal baseline rather than the shipped one.

        return super().expected_goals(home_team, away_team, venue, self.g)

    def rho_correction(self, home_goals, away_goals, x_home, x_away, rho = None):

        # Forces this instance's own rho rather than the shipped one.

        return super().rho_correction(home_goals, away_goals, x_home, x_away, self.rho)


def load_tournament():

    """
    Loads the 104 matches of the 2026 tournament, standardising names and labelling rounds.

    Returns
    -------
    pd.DataFrame
        One row per match, sorted by date, with home_team, away_team, home_score, away_score,
        venue, stage and round.
    """

    results = load_csv('results.csv')
    results['date'] = pd.to_datetime(results['date'])

    played = results[(results['tournament'] == 'FIFA World Cup') &
                     (results['date'] >= TOURNAMENT_START) &
                     (results['date'] <= TOURNAMENT_END)].sort_values('date')
    played = played.reset_index(drop = True)
    played[['home_team', 'away_team']] = played[['home_team', 'away_team']].replace(
        CHANGED_TEAM_NAMES)

    played['venue'] = played['country'].map(COUNTRY_VENUE)
    played['stage'] = np.where(played['date'] < KNOCKOUT_START, 'group', 'knockout')

    knockout = played[played['stage'] == 'knockout'].index
    labels = pd.Series(index = played.index, dtype = object)
    cut = 0
    for name, size in ROUNDS:
        labels.loc[knockout[cut:cut + size]] = name
        cut += size
    played['round'] = labels.fillna('Group stage')

    return played


def who_advanced(played):

    """
    Works out which team went through from each knockout tie.

    A tie level at the end of extra time is settled on penalties, which results.csv doesn't
    record, so the winner is taken from whoever turns up in a later round.

    Parameters
    ----------
    played : pd.DataFrame
        From load_tournament.

    Returns
    -------
    dict of {int : str}
        Row index to the team that advanced, for knockout rows only.
    """

    knockout = played[played['stage'] == 'knockout']
    advanced = {}

    for index, match in knockout.iterrows():
        if match['home_score'] != match['away_score']:
            advanced[index] = (match['home_team'] if match['home_score'] > match['away_score']
                               else match['away_team'])
            continue

        later = knockout[knockout['date'] > match['date']]
        in_later = set(later['home_team']) | set(later['away_team'])
        still_in = {match['home_team'], match['away_team']} & in_later

        advanced[index] = still_in.pop() if len(still_in) == 1 else match['home_team']

    return advanced


def score_forecast(simulator, played, advanced):

    """
    Scores every match on log loss and whether the most likely outcome happened.

    Parameters
    ----------
    simulator : MatchSimulator
        The engine to score with.
    played : pd.DataFrame
        From load_tournament.
    advanced : dict of {int : str}
        From who_advanced.

    Returns
    -------
    tuple of np.ndarray
        Per-match log loss, a boolean hit per match, and the ranked probability score for the
        group matches, which are the ones with three ordered outcomes.
    """

    losses, hits, group_rps = [], [], []

    for index, match in played.iterrows():
        home, away, venue = match['home_team'], match['away_team'], match['venue']

        if match['stage'] == 'group':
            probabilities = simulator.match_probabilities(home, away, venue = venue)
            p = np.array([probabilities['home_win'], probabilities['draw'],
                          probabilities['away_win']])
            outcome = (0 if match['home_score'] > match['away_score'] else
                       1 if match['home_score'] == match['away_score'] else 2)
            group_rps.append(ranked_probability_score(p[None, :], np.array([outcome]))[0])
        else:
            probabilities = simulator.match_probabilities(home, away, venue = venue,
                                                          stage = 'knockout')
            p = np.array([probabilities['home_advances'], probabilities['away_advances']])
            outcome = 0 if advanced[index] == home else 1

        losses.append(-np.log(max(p[outcome], 1e-12)))
        hits.append(int(p.argmax()) == outcome)

    return np.array(losses), np.array(hits), np.array(group_rps)


def build_baseline(df, teams, team_index):

    """
    Builds the Elo-only baseline, fitted on the same pre-tournament window as the model.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data.
    teams : list of str
        Every team in the window, sorted.
    team_index : dict of {str : int}
        Team name to its position in teams.

    Returns
    -------
    BaselineSimulator
        Ready to score, indexed by team name.
    """

    squad_value = calc_squad_value(teams)
    attack, defence = build_ratings(df, ELO_ONLY, teams, team_index, squad_value)

    finalists = get_wc_teams()
    fixtures = df[df['is_home'] & df['team'].isin(finalists) & df['opponent'].isin(finalists)]
    fitted = optimize.minimize(engine_nll, [-0.10, 1.05], method = 'Nelder-Mead',
                               args = (attack, defence, fixtures, team_index))

    stats_frame = pd.DataFrame({'attack_rating' : attack, 'defence_rating' : defence},
                               index = teams).loc[sorted(finalists & set(teams))]

    return BaselineSimulator(stats_frame, *fitted.x)


def escape_brier(played):

    """
    Brier score for the group escape probabilities, against a constant 32 of 48.

    Parameters
    ----------
    played : pd.DataFrame
        From load_tournament.

    Returns
    -------
    dict
        Model, constant and best-achievable Brier scores.
    """

    escape = load_csv(f'{RUN_DIR}/group_escape.csv', index_col = 'Team')['Escape']

    knockout = played[played['round'] == 'Round of 32']
    escaped = set(knockout['home_team']) | set(knockout['away_team'])
    happened = escape.index.isin(escaped).astype(float)

    constant = len(escaped) / len(escape)

    return {'model' : float(np.mean((escape - happened) ** 2)),
            'constant' : float(np.mean((constant - happened) ** 2)),
            'ceiling' : float(np.mean(escape * (1 - escape)))}


def bracket_hits(played):

    """
    How many of the teams predicted in each round actually reached it.

    Parameters
    ----------
    played : pd.DataFrame
        From load_tournament.

    Returns
    -------
    pd.DataFrame
        One row per round, predicted against actual.
    """

    bracket = load_csv(f'{RUN_DIR}/likely_bracket.csv')

    rows = []
    for name, _ in ROUNDS:
        actual = played[played['round'] == name]
        actual_teams = set(actual['home_team']) | set(actual['away_team'])
        predicted = bracket[bracket['Round'] == name]
        predicted_teams = set(predicted['Home']) | set(predicted['Away'])

        if actual_teams and predicted_teams:
            rows.append({'round' : name, 'correct' : len(predicted_teams & actual_teams),
                         'of' : len(actual_teams)})

    return pd.DataFrame(rows)


def champion(played):

    """
    The predicted champion against the team that actually won it.

    Parameters
    ----------
    played : pd.DataFrame
        From load_tournament.

    Returns
    -------
    dict
        Predicted and actual champion.
    """

    bracket = load_csv(f'{RUN_DIR}/likely_bracket.csv')
    predicted = bracket[bracket['Round'] == 'Final']['Winner'].iloc[0]

    final = played[played['round'] == 'Final']
    actual = who_advanced(played)[final.index[0]]

    return {'predicted' : predicted, 'actual' : actual}


def goals_per_match(played):

    """
    Goals per match, simulated against the tournament itself and against past World Cups.

    Parameters
    ----------
    played : pd.DataFrame
        From load_tournament.

    Returns
    -------
    dict
        Simulated average from the published run, the 2026 actual, and World Cups since 1998.
    """

    distribution = load_csv(f'{RUN_DIR}/scoreline_distribution.csv')
    simulated = float(((distribution['higher_score'] + distribution['lower_score']) *
                       distribution['probability']).sum())

    actual = float((played['home_score'] + played['away_score']).mean())

    results = load_csv('results.csv')
    history = results[(results['tournament'] == 'FIFA World Cup') &
                      (results['date'] >= '1998-06-01') & (results['date'] <= '2022-12-31')]
    historical = float((history['home_score'] + history['away_score']).mean())

    return {'simulated' : simulated, 'actual' : actual, 'historical' : historical}


def run():

    """
    Scores the published forecast and prints every figure the README quotes.
    """

    played = load_tournament()
    advanced = who_advanced(played)

    df = build_team_match_data()
    teams = sorted(set(df['team']) | set(df['opponent']))
    team_index = {team : i for i, team in enumerate(teams)}

    model = MatchSimulator(build_all_team_data(df))
    baseline = build_baseline(df, teams, team_index)

    model_loss, model_hits, model_rps = score_forecast(model, played, advanced)
    baseline_loss, baseline_hits, baseline_rps = score_forecast(baseline, played, advanced)

    difference = baseline_loss - model_loss
    standard_error = difference.std(ddof = 1) / np.sqrt(len(difference))

    print()
    print(f'HOW IT PERFORMED  ({len(played)} matches, inputs frozen {FREEZE_DATE})')
    print()
    print(f'  {"this model":<26} log loss {model_loss.mean():.3f}   '
          f'{model_hits.sum()}/{len(model_hits)}   RPS {model_rps.mean():.3f}')
    print(f'  {"Elo-only baseline":<26} log loss {baseline_loss.mean():.3f}   '
          f'{baseline_hits.sum()}/{len(baseline_hits)}   RPS {baseline_rps.mean():.3f}')
    print(f'  RPS is over the {len(model_rps)} group matches, the ones with three ordered '
          f'outcomes.')
    print()
    print(f'  {difference.mean() / baseline_loss.mean():.1%} lower log loss than the baseline '
          f'(t = {difference.mean() / standard_error:.1f})')

    group = played['stage'] == 'group'
    drawn = group & (played['home_score'] == played['away_score'])

    print()
    print(f'  decisive matches  {model_hits[~drawn.to_numpy()].sum()}/{(~drawn).sum()}')
    print(f'  draws             {model_hits[drawn.to_numpy()].sum()}/{drawn.sum()}')
    print(f'  knockout ties     {model_hits[~group.to_numpy()].sum()}/{(~group).sum()}')

    worst = model_loss.argmax()
    match = played.iloc[worst]
    print()
    print(f'  worst call  {match["home_team"]} {match["home_score"]}-{match["away_score"]} '
          f'{match["away_team"]}, at {np.exp(-model_loss[worst]):.1%} on what happened')

    goals = goals_per_match(played)
    print()
    print(f'  goals per match  simulated {goals["simulated"]:.2f}, '
          f'actual {goals["actual"]:.2f}, World Cups since 1998 {goals["historical"]:.2f}')

    brier = escape_brier(played)
    print()
    print(f'  group escape Brier  {brier["model"]:.3f} against {brier["constant"]:.3f} for a '
          f'constant, best possible {brier["ceiling"]:.3f}')

    print()
    print('  bracket')
    for _, row in bracket_hits(played).iterrows():
        print(f'    {row["round"]:<22} {row["correct"]}/{row["of"]}')

    won = champion(played)
    print()
    print(f'  champion  predicted {won["predicted"]}, actual {won["actual"]}')


if __name__ == '__main__':
    run()
