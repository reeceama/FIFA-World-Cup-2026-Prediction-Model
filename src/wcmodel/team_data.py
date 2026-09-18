"""
This module loads the international results, applies the time window, weighs every fixture by recency and tournament importance,
then turns those weighted results into an attack and a defence rating for each participating team.

Each team's attack is divided by the defensive strength of the teams they faced during the window, and vice versa.
After, the ratings are blended with an Elo prior, raw goal rates and squad market values.
"""

import os

import pandas as pd
import numpy as np

# Paths are anchored to the package so every entry point, notebook, script or module -
# resolves the same files regardless of the directory it was launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, 'data')
IMAGE_DIR = os.path.join(ROOT, 'images')
CALIBRATION_DIR = os.path.join(DATA_DIR, 'calibration')
FLAG_DIR = os.path.join(DATA_DIR, 'flags')

# ---------- Data Loading ----------

def load_csv(file_name, **kwargs):

    """
    Loads a CSV from the data and passes any extra keyword arguments through to pd.read_csv.

    Parameters
    ----------
    file_name : str
        File name including extension, relative to data/.
    **kwargs
        Passed through to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded file.
    """
    
    df = pd.read_csv(os.path.join(DATA_DIR, file_name), **kwargs)
    return df

group_fixtures = load_csv('group_fixtures.csv')
knockout_slots = load_csv('knockout_slots.csv')

# Fixing placeholder values for teams that qualiified through playoffs.
CONFIRMED_TEAMS = {
    'FIFA Playoff 1' : 'DR Congo',
    'FIFA Playoff 2' : 'Iraq',
    'UEFA Playoff A' : 'Bosnia and Herzegovina',
    'UEFA Playoff B' : 'Sweden',
    'UEFA Playoff C' : 'Turkey',
    'UEFA Playoff D' : 'Czech Republic',
}

# Standardising team names.
CHANGED_TEAM_NAMES = {
    'Cape Verde' : 'Cabo Verde',
    'Ivory Coast' : "Côte d'Ivoire",
    'United States' : 'USA',
    'Czechia' : 'Czech Republic',
    'Turkiye' : 'Turkey',
    'Bosnia-Herzegovina' : 'Bosnia and Herzegovina',
    'Congo DR' : 'DR Congo'
}


FLAG_CODES = load_csv('flag_codes.csv', index_col = 'team')['code'].to_dict()

group_fixtures = group_fixtures.replace(CONFIRMED_TEAMS)


# ----------  Match Weighting ----------


# Numbers loosely based on FIFA's official Elo weighting for each tournament.
FRIENDLY = 7.5
NATIONS_LEAGUE = 20
CONF_QUAL = 25
CONF = 37.5
WC_QUAL = 25
WC = 55
DEFAULT = 10

TOURNAMENT_WEIGHTS = {
    'Friendly': FRIENDLY,
    'CONCACAF Nations League': NATIONS_LEAGUE,
    'UEFA Nations League': NATIONS_LEAGUE,
    'UEFA Euro qualification': CONF_QUAL,
    'UEFA Euro': CONF,
    'Copa América qualification': CONF_QUAL,
    'Copa América': CONF,
    'African Cup of Nations qualification': CONF_QUAL,
    'African Cup of Nations': CONF,
    'AFC Asian Cup qualification': CONF_QUAL,
    'AFC Asian Cup': CONF,
    'Gold Cup qualification': CONF_QUAL,
    'Gold Cup': CONF,
    'Oceania Nations Cup qualification': CONF_QUAL,
    'Oceania Nations Cup': CONF,
    'FIFA World Cup qualification': WC_QUAL,
    'FIFA World Cup': WC,
}


DECAY_RATE = 0.00095
FREEZE_DATE = '2026-06-10'

WINDOW = '5y'
START_DATE = {'5y' : '2021-06-01', '8y' : '2018-06-01'}[WINDOW]

ELO_HISTORY = 'wf_elo_history.csv'
ELIGIBILITY = 'wf_avg_elo.csv'
SQUAD_VALUES = 'tm_squad_values.csv'

# Constants set by held-out data.
RAW_WEIGHT = 0.10
ELO_WEIGHT = 0.30
VALUE_WEIGHT = 0.07

SOLVER_ITERATIONS = 4000
SOLVER_TOLERANCE = 1e-12
SOLVER_DAMPING = 0.5
SOLVER_CLIP = 20.0


def match_weight(results):

    """
    Determines how much each fixture counts, by considering recency and tournament weight.

    Parameters
    ----------
    results : pd.DataFrame
        Match rows with date and tournament columns.

    Returns
    -------
    pd.Series
        Weight per row, aligned to the input index.
    """

    days_ago = (pd.Timestamp(FREEZE_DATE) - pd.to_datetime(results['date'])).dt.days
    decay = np.exp(-DECAY_RATE * days_ago)
    importance = results['tournament'].map(lambda t: TOURNAMENT_WEIGHTS.get(t, DEFAULT))

    return decay * importance


# ---------- Team Ratings ----------


def build_team_match_data():

    """
    Creates a dual perspective DataFrame within the allocated time window that allows stats to be calculated
    using a single groupby pass, and adds the weight and team elo at the point of each game to the table.

    Team Elo wasn't able to be added accurately for 100% of games, so the team's last known rating
    was used as a fill in to compensate, and for any game before a team's first rated fixture,
    a running median is used instead.

    Returns
    -------
    pd.DataFrame
        One row per team per fixture, sorted by date, with columns date, is_home, neutral,
        team, opponent, goals_scored, goals_conceded, weight, elo and opponent_elo.
    """

    results = load_csv('results.csv')
    recent_results = results[(results['date'] >= START_DATE) &
                             (results['date'] <= FREEZE_DATE) &
                             (results['home_score'].notna())].copy()
    recent_results[['home_team', 'away_team']] = (recent_results[['home_team', 'away_team']]
                                                  .replace(CHANGED_TEAM_NAMES))

    eligible_teams = set(load_csv(ELIGIBILITY, index_col = 'team').index)
    recent_results = recent_results[recent_results['home_team'].isin(eligible_teams) &
                                    recent_results['away_team'].isin(eligible_teams)]

    recent_results['weight'] = match_weight(recent_results)

    home_perspective = pd.DataFrame({
        'date' : recent_results['date'], 'is_home' : True,
        'neutral' : recent_results['neutral'], 'team' : recent_results['home_team'],
        'opponent' : recent_results['away_team'], 'goals_scored' : recent_results['home_score'],
        'goals_conceded' : recent_results['away_score'], 'weight' : recent_results['weight']})

    away_perspective = pd.DataFrame({
        'date' : recent_results['date'], 'is_home' : False,
        'neutral' : recent_results['neutral'], 'team' : recent_results['away_team'],
        'opponent' : recent_results['home_team'], 'goals_scored' : recent_results['away_score'],
        'goals_conceded' : recent_results['home_score'], 'weight' : recent_results['weight']})

    df = pd.concat([home_perspective, away_perspective], ignore_index = True).dropna()
    df['date'] = pd.to_datetime(df['date'])

    # Pre-match elo for each fixture, recovered from the rating the team carried afterwards.
    elo_history = load_csv(ELO_HISTORY)
    elo_history['date'] = pd.to_datetime(elo_history['date'])
    elo_history['elo'] = elo_history['elo_after'] - elo_history['elo_change']
    elo_history = elo_history[['date', 'team', 'opponent', 'elo']].drop_duplicates(
        subset = ['date', 'team', 'opponent'])

    df = df.merge(elo_history, on = ['date', 'team', 'opponent'], how = 'left')
    df = df.merge(elo_history.rename(columns = {'team' : 'opponent',
                                                'opponent' : 'team',
                                                'elo' : 'opponent_elo'}),
                  on = ['date', 'team', 'opponent'], how = 'left')

    # 2.8% of rows fail the fixture join. Carrying the last known rating forward keeps the fill
    # backward-looking, so nothing from after the split reaches a training row.
    df = df.sort_values('date').reset_index(drop = True)
    df['elo'] = df['elo'].fillna(df.groupby('team')['elo'].ffill())
    df['opponent_elo'] = df['opponent_elo'].fillna(df.groupby('opponent')['opponent_elo'].ffill())

    # What is left predates the team's first rated fixture, so a running median stands in.
    running_median = df['elo'].expanding().median().bfill()
    df['elo'] = df['elo'].fillna(running_median)
    df['opponent_elo'] = df['opponent_elo'].fillna(running_median)

    return df


def get_wc_teams():

    """
    Filters out teams that did not qualify for the World Cup, leaving only the 48 teams with standardised names.

    Returns
    -------
    set of str
        The 48 finalists.
    """

    wc_teams = set(group_fixtures['home_team']) | set(group_fixtures['away_team'])

    return {CHANGED_TEAM_NAMES.get(team, team) for team in wc_teams}


def calc_avg_elo(df, teams, team_index, column = 'elo'):

    """
    Each team's Elo over the window, logged, weighted and standardised.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data.
    teams : list of str
        Every team in the window, sorted.
    team_index : dict of {str : int}
        Team name to its position in teams.
    column : str, default 'elo'
        Which rating to average. calc_relative_schedule passes 'opponent_elo' to get the same
        figure for the teams faced.

    Returns
    -------
    np.ndarray
        Standardised log Elo, indexed by position in teams.
    """

    team_indices = df['team'].map(team_index).to_numpy()
    weight = df['weight'].to_numpy(float)
    log_elo = np.log(df[column].to_numpy(float)) # Takes the logarithm of each Elo value rather than the raw value.

    sum_weight = np.bincount(team_indices, weights = weight, minlength = len(teams))
    sum_weighted_elo = np.bincount(team_indices, weights = weight * log_elo, minlength = len(teams))

    played = sum_weight > 0
    avg_elo = np.full(len(teams), np.nan)
    avg_elo[played] = sum_weighted_elo[played] / sum_weight[played]
    avg_elo = np.where(np.isfinite(avg_elo), avg_elo, np.nanmean(avg_elo))

    return (avg_elo - avg_elo.mean()) / avg_elo.std()

def calc_relative_schedule(df, teams, team_index):

    """
    How strong a team's opponents were, relative to the team itself.

    Not used in the ratings. It is reported in the team stats table, since it is still
    informative about how hard a team's fixtures were, and it is the measurement behind the
    schedule strength entry in the README's cut features.

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
    np.ndarray
        Standardised schedule strength, indexed by position in teams.
    """

    relative = (calc_avg_elo(df, teams, team_index, 'opponent_elo') -
                calc_avg_elo(df, teams, team_index))

    return (relative - relative.mean()) / relative.std()


def calc_squad_value(teams):

    """
    Squad market value per team, logged and standardised.

    Only the 48 finalists have squad values. Everyone else gets zero, which keeps the array
    aligned with the other ratings and leaves them unadjusted.

    Parameters
    ----------
    teams : list of str
        Every team in the window, sorted.

    Returns
    -------
    np.ndarray
        Standardised log squad value, indexed by position in teams.
    """

    squads = load_csv(SQUAD_VALUES, index_col = 'team')
    log_values = np.log(squads['squad_value'].astype(float))  # Takes the logarithm of each squad market value than the raw value.
    standardised_values = (log_values - log_values.mean()) / log_values.std()

    return np.array([standardised_values.get(team, 0.0) for team in teams], dtype = float)


def calc_venue_boost(df, home_advantage):

    """
    The home multiplier for each row, so venue can be taken out of the raw goal rates.

    A team at a real home venue gets the boost, its opponent gets the reciprocal, and neutral
    fixtures get 1.0.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data, needs neutral and is_home.
    home_advantage : float
        The multiplier applied at a real home venue.

    Returns
    -------
    np.ndarray
        One multiplier per row of df.
    """

    played_at_home = ~df['neutral'].to_numpy(bool)
    is_home = df['is_home'].to_numpy(bool)

    return np.where(played_at_home & is_home, home_advantage,
           np.where(played_at_home & ~is_home, 1 / home_advantage, 1.0))


def solve_team_ratings(df, teams, team_index, home_advantage = 1.0):

    """
    Solve attack and defence ratings together, adjusting for the opposition faced.

    A team's attack is its weighted goals scored divided by what an average attack would have
    been expected to score against the defences it actually played, and its defence is the
    mirror. Each side depends on the other, so both start flat and are recomputed in turn until
    they stop moving. The fixed point is the maximum likelihood estimate of a weighted Poisson
    model, so this is a fit rather than a heuristic.

    Attack is renormalised to a weighted mean of 1 on every pass, which leaves defence carrying
    the overall goal level. Updates are damped by SOLVER_DAMPING and clipped by SOLVER_CLIP,
    since an undamped solve oscillates on teams with few matches. Stops when the largest change
    falls below SOLVER_TOLERANCE, or after SOLVER_ITERATIONS passes.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data.
    teams : list of str
        Every team in the window, sorted.
    team_index : dict of {str : int}
        Team name to its position in teams.
    home_advantage : float, default 1.0
        Divides out of goals scored and multiplies into goals conceded, taking venue out of the
        ratings. Left at 1.0 everywhere except the parameter search, since the model applies
        home advantage at match time instead.

    Returns
    -------
    tuple of np.ndarray
        Attack, defence, and the total match weight behind each team, all indexed by position
        in teams. The weights are reused by blend_ratings and reported in the team table.
    """

    team_indices = df['team'].map(team_index).to_numpy()
    opponent_indices = df['opponent'].map(team_index).to_numpy()
    venue_boost = calc_venue_boost(df, home_advantage)
    goals_scored = df['goals_scored'].to_numpy(float) / venue_boost
    goals_conceded = df['goals_conceded'].to_numpy(float) * venue_boost
    weight = df['weight'].to_numpy(float)

    sum_weight = np.bincount(team_indices, weights = weight, minlength = len(teams))
    sum_weight[sum_weight == 0] = 1e-9

    attack = np.ones(len(teams))
    defence = np.full(len(teams), np.average(goals_scored, weights = weight))

    for _ in range(SOLVER_ITERATIONS):
        expected_conceded = np.bincount(team_indices, weights = weight * defence[opponent_indices],
                                        minlength = len(teams))
        new_attack = ((1 - SOLVER_DAMPING) *
                      (np.bincount(team_indices, weights = weight * goals_scored,
                                   minlength = len(teams)) /
                       np.maximum(expected_conceded, 1e-9)) + SOLVER_DAMPING * attack)
        new_attack = new_attack / np.average(new_attack, weights = sum_weight)
        new_attack = np.clip(new_attack, 1 / SOLVER_CLIP, SOLVER_CLIP)

        expected_scored = np.bincount(team_indices, weights = weight * new_attack[opponent_indices],
                                      minlength = len(teams))
        new_defence = ((1 - SOLVER_DAMPING) *
                       (np.bincount(team_indices, weights = weight * goals_conceded,
                                    minlength = len(teams)) /
                        np.maximum(expected_scored, 1e-9)) + SOLVER_DAMPING * defence)

        average_defence = np.average(new_defence, weights = sum_weight)
        new_defence = np.clip(new_defence, average_defence / SOLVER_CLIP,
                              average_defence * SOLVER_CLIP)

        largest_change = max(np.max(np.abs(new_attack - attack)),
                             np.max(np.abs(new_defence - defence)))
        attack, defence = new_attack, new_defence
        if largest_change < SOLVER_TOLERANCE:
            break

    return attack, defence, sum_weight


def calc_raw_rates(df, teams, team_index, home_advantage = 1.0):

    """
    Weighted goals scored and conceded per game, without adjusting for the opposition.

    Parameters
    ----------
    df : pd.DataFrame
        Match data from build_team_match_data.
    teams : list of str
        Every team in the window, sorted.
    team_index : dict of {str : int}
        Team name to its position in teams.
    home_advantage : float, default 1.0
        Passed to calc_venue_boost. Left at 1.0 when venue is being handled elsewhere.

    Returns
    -------
    tuple of np.ndarray
        Goals scored per game and goals conceded per game, both indexed by position in teams.
    """
    
    team_indices = df['team'].map(team_index).to_numpy()
    weight = df['weight'].to_numpy(float)
    venue_boost = calc_venue_boost(df, home_advantage)

    sum_weight = np.bincount(team_indices, weights = weight, minlength = len(teams))
    sum_weight[sum_weight == 0] = 1e-9

    raw_scored = np.bincount(team_indices,
                             weights = weight * df['goals_scored'].to_numpy(float) / venue_boost,
                             minlength = len(teams)) / sum_weight
    raw_conceded = np.bincount(team_indices,
                               weights = weight * df['goals_conceded'].to_numpy(float) * venue_boost,
                               minlength = len(teams)) / sum_weight

    return raw_scored, raw_conceded

def blend_ratings(solved, raw, elo_prior, raw_weight, elo_weight, sum_weight):

    """
    Mix the solved ratings with raw goal rates and an Elo prior.

    Results alone are noisy for teams that play few competitive games, so the solved rating is
    pulled towards two other views of the same team. The three are combined in log space, with
    whatever share is left over after raw_weight and elo_weight going to the solved rating.

    Both of the other inputs are put on the solved rating's scale first, so they shift a team
    relative to the field without moving the field itself. The raw rates are recentred on the
    solved mean, and the prior, which arrives standardised, is rescaled to the solved spread.
    Defence is handled by passing the prior negated, since a stronger team should concede fewer.

    Parameters
    ----------
    solved : np.ndarray
        Opponent-adjusted ratings from solve_team_ratings.
    raw : np.ndarray
        Unadjusted goal rates from calc_raw_rates.
    elo_prior : np.ndarray
        Standardised log Elo from calc_avg_elo, negated when blending defence.
    raw_weight, elo_weight : float
        Share of the blend given to each. The remainder goes to solved.
    sum_weight : np.ndarray
        Total match weight per team, so the weighted means aren't dominated by teams with
        barely any data.

    Returns
    -------
    np.ndarray
        Blended ratings, indexed by position in teams.
    """

    # One log(0) would poison the weighted means for every team, so both series are floored.
    solved = np.where(solved > 0, solved, np.average(solved[solved > 0],
                                                     weights = sum_weight[solved > 0]))
    raw = np.where(raw > 0, raw, np.average(raw, weights = sum_weight))

    log_solved = np.log(solved)
    log_raw = np.log(raw)
    log_raw = (log_raw - np.average(log_raw, weights = sum_weight)
               + np.average(log_solved, weights = sum_weight))
    log_prior = elo_prior * log_solved.std() + np.average(log_solved, weights = sum_weight)

    return np.exp((1 - raw_weight - elo_weight) * log_solved
                  + raw_weight * log_raw + elo_weight * log_prior)


def apply_squad_value(attack, defence, squad_value, weight):

    """
    Shift ratings by squad market value, up on attack and down on defence.

    Helps teams whose results haven't caught up with their talent. 

    Parameters
    ----------
    attack, defence : np.ndarray
        Blended ratings, indexed by position in teams.
    squad_value : np.ndarray
        Standardised log squad value from calc_squad_value.
    weight : float
        How much value counts.

    Returns
    -------
    tuple of np.ndarray
        Adjusted attack and defence.
    """

    if weight == 0:
        return attack, defence

    return attack * np.exp(weight * squad_value), defence * np.exp(-weight * squad_value)


def build_wc_team_stats(attack_rating, defence_rating, raw_scored, raw_conceded, 
                        schedule_faced, squad_value, sum_weight, teams):

    """
    Collect every rating component into one table and cut it down to the 48 finalists.

    Parameters
    ----------
    attack_rating, defence_rating : np.ndarray
        The final ratings, after blending and both adjustments.
    raw_scored, raw_conceded : np.ndarray
        Unadjusted goal rates, kept for reference.
    schedule_faced, squad_value : np.ndarray
        The two rating inputs that aren't goals.
    sum_weight : np.ndarray
        Total match weight behind each team, a rough measure of how much data it has.
    teams : list of str
        Every team in the window, sorted. All arrays are indexed by position in it.

    Returns
    -------
    pd.DataFrame
        One row per finalist, indexed by team, with current Elo joined on for display.
    """

    team_stats = pd.DataFrame({
        'attack_rating' : attack_rating,
        'defence_rating' : defence_rating,
        'raw_scored' : raw_scored,
        'raw_conceded' : raw_conceded,
        'schedule_faced' : schedule_faced,
        'squad_value' : squad_value,
        'weight' : sum_weight,
    }, index = teams)

    team_stats.index.name = 'team'

    # A filter that removes all non-WC teams from team_stats.
    wc_team_stats = team_stats.loc[sorted(get_wc_teams() & set(teams))].copy()

    # The snapshot date is set in fetch_elo.py, at 2026-06-10.
    wc_current_elo = load_csv('wf_current_elo.csv', index_col = 'team')
    wc_team_stats['current_elo'] = wc_current_elo['current_elo']

    return wc_team_stats


def build_all_team_data(df = None):

    """
    Run the whole rating pipeline and return the finalists' stats.

    Solves the ratings, blends them with the raw rates and the Elo prior, applies squad value,
    then builds the table. This is the function the notebooks call.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Match data from build_team_match_data. Built here if not given, which is the usual case.

    Returns
    -------
    pd.DataFrame
        One row per finalist, sorted by current Elo.
    """

    if df is None:
        df = build_team_match_data()

    teams = sorted(set(df['team']) | set(df['opponent']))
    team_index = {team : i for i, team in enumerate(teams)}

    solved_attack, solved_defence, sum_weight = solve_team_ratings(df, teams, team_index)
    raw_scored, raw_conceded = calc_raw_rates(df, teams, team_index)
    elo_prior = calc_avg_elo(df, teams, team_index)
    schedule_faced = calc_relative_schedule(df, teams, team_index)
    squad_value = calc_squad_value(teams)

    attack_rating = blend_ratings(solved_attack, raw_scored, elo_prior,
                                  RAW_WEIGHT, ELO_WEIGHT, sum_weight)
    defence_rating = blend_ratings(solved_defence, raw_conceded, -elo_prior,
                                   RAW_WEIGHT, ELO_WEIGHT, sum_weight)

    attack_rating, defence_rating = apply_squad_value(
        attack_rating, defence_rating, squad_value, VALUE_WEIGHT)

    wc_team_stats = build_wc_team_stats(attack_rating, defence_rating,
                                        raw_scored, raw_conceded, schedule_faced,
                                        squad_value, sum_weight, teams)

    return wc_team_stats.sort_values('current_elo', ascending = False)
