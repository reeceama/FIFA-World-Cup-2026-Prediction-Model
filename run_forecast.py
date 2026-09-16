"""
Runs the full Monte Carlo and saves the results to a timestamped folder, alongside a
JSON record of the parameters and the data freeze that produced them.

Usage: python run_forecast.py [simulations]
"""

import os, sys, json, datetime as dt

import numpy as np

from wcmodel.team_data import *
from wcmodel.match import *
from wcmodel.tournament import *
from wcmodel.fit_mle import OUTPUT_PATH

def output_folder(simulations):

    """
    Timestamped folder name, so a rerun never overwrites an earlier forecast.

    Parameters
    ----------
    simulations : int
        How many tournaments the run played, used in the folder name.

    Returns
    -------
    str
        Full path to write the run into.
    """

    return os.path.join(DATA_DIR, f"{simulations // 1000}k_monte_carlo_"
                        f"{dt.datetime.now().strftime('%Y-%m-%d_%H%M')}")


def main(simulations = 100000, seed = 20260611):

    """
    Simulates the tournament and writes the forecast tables, the team ratings behind them
    and the run metadata into a new folder.

    Parameters
    ----------
    simulations : int, default 100000
        How many tournaments to play.
    seed : int, default 20260611
        Fixed so a run can be reproduced exactly.
    """

    np.random.seed(seed)

    started_at = dt.datetime.now()
    folder = output_folder(simulations)

    # Checked before simulating, so a name clash costs a second rather than the whole run.
    if os.path.exists(folder):
        raise FileExistsError(f'{folder} already exists. Rename or remove it before rerunning.')

    match_sim = MatchSimulator(build_all_team_data())

    knockout_slots = load_csv('knockout_slots.csv')
    tournament_sim = TournamentSimulator(match_sim, group_fixtures, knockout_slots)
    forecast = TournamentForecast(tournament_sim)
    forecast.run(simulations)

    os.makedirs(folder)

    forecast.progression_table().to_csv(f'{folder}/progression_table.csv')
    forecast.group_table().to_csv(f'{folder}/group_escape.csv')
    forecast.likely_bracket().to_csv(f'{folder}/likely_bracket.csv')
    match_sim.team_stats.to_csv(f'{folder}/team_stats.csv')
    forecast.scoreline_table().to_csv(f'{folder}/scoreline_distribution.csv', index = False)

    calibration = json.load(open(OUTPUT_PATH))

    json.dump({'simulations' : simulations,
               'seed' : seed,
               'rho' : RHO,
               'world_cup_baseline' : WORLD_CUP_BASELINE,
               'home_advantage' : HOME_ADVANTAGE,
               'shrinkage' : SHRINKAGE,
               'raw_weight' : RAW_WEIGHT,
               'elo_weight' : ELO_WEIGHT,
               'schedule_weight' : SCHEDULE_WEIGHT,
               'value_weight' : VALUE_WEIGHT,
               'data_frozen' : calibration['window']['freeze'],
               'calibration' : calibration,
               'started_at' : started_at.isoformat(timespec = 'seconds'),
               'finished_at' : dt.datetime.now().isoformat(timespec = 'seconds')},
              open(f'{folder}/forecast_run.json', 'w'), indent = 2)

    print(f'Finished {simulations} simulations. Saved to {folder}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100000)