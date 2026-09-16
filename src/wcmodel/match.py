"""
The match engine. Turns two team ratings into expected goals, a scoreline distribution, and
either a set of probabilities or a sampled result.

Goals are drawn from independent Poisson distributions with the Dixon-Coles correction applied
to the four lowest scoring cells, which plain Poisson gets wrong. Host nations get a boost at
venues in their own country, applied as a ratio between the two sides so a match doesn't gain
goals overall.

Constants here are the fitted engine values. Everything upstream of them is in team_data.
"""

import numpy as np
import scipy.stats as stats

from wcmodel.team_data import *


# ---------- Poisson Match Engine ----------


# Rho and the goal baseline are fitted by maximum likelihood on the 675 fixtures played between
# the 48 finalists inside the window, conditional on the shipped Stage 1 weights in team_data.
# Home advantage is fitted on all 5,328. The calibration summary reports different values
# because its search selects its own weights rather than using these.
RHO = -0.1079  # Dixon-Coles low-score correction.
WORLD_CUP_BASELINE = 1.0972  # Goal level between finalists against the level the ratings are normalised on.


HOME_ADVANTAGE = 1.1778  # The multiplicative stat boost given to host nations in their country.
CO_HOST_ADVANTAGE = HOME_ADVANTAGE - (HOME_ADVANTAGE - 1) / 2  # The multiplicative stat boost given to hosts out of their country.


class MatchSimulator:

    """
    Plays a single match between two teams, either exactly or by sampling.

    Everything the engine does starts from expected_goals. match_probabilities and score_matrix
    report the distribution without simulating, while monte_carlo_gs and monte_carlo_ko draw a
    result from it and are what the tournament simulator calls.

    Parameters
    ----------
    team_stats : pd.DataFrame
        From build_all_team_data, indexed by team, with attack_rating and defence_rating.

    Attributes
    ----------
    avg_yellow, avg_red : float
        Cards per team per game, averaged over the three most recent World Cups.
    """

    HOST_TEAMS = {'USA', 'Mexico', 'Canada'}

    CARD_TOURNAMENTS = ['WC-2014', 'WC-2018', 'WC-2022']

    VENUE_COUNTRY = {
        'Estadio Azteca, Mexico City' : 'Mexico',
        'Estadio Akron, Guadalajara' : 'Mexico',
        'Estadio BBVA, Monterrey' : 'Mexico',
        'BMO Field, Toronto' : 'Canada',
        'BC Place, Vancouver' : 'Canada',
        'SoFi Stadium, Los Angeles' : 'USA',
        "Levi's Stadium, Santa Clara" : 'USA',
        'MetLife Stadium, East Rutherford' : 'USA',
        'Gillette Stadium, Boston' : 'USA',
        'NRG Stadium, Houston' : 'USA',
        'AT&T Stadium, Dallas' : 'USA',
        'Lincoln Financial Field, Philadelphia' : 'USA',
        'Mercedes-Benz Stadium, Atlanta' : 'USA',
        'Lumen Field, Seattle' : 'USA',
        'GEHA Field at Arrowhead Stadium, Kansas City' : 'USA',
        'Hard Rock Stadium, Miami' : 'USA',
    }
    
    def __init__(self, team_stats):

        self.team_stats = team_stats
        self.historical_avg_cards()

    def host_boost(self, team, venue):
        
        """
        The multiplier a host nation gets for playing in North America.

        Full advantage in their own country, half of it in either of the other two, and 1 for
        every other team. The nerf a team takes for facing a host happens in expected_goals,
        which divides the two boosts by each other.

        Parameters
        ----------
        team : str
            Team name, as indexed in team_stats.
        venue : str
            Venue name, matched against VENUE_COUNTRY. An unknown venue counts as neutral.

        Returns
        -------
        float
            HOME_ADVANTAGE, CO_HOST_ADVANTAGE, or 1.
        """
    
        if team not in self.HOST_TEAMS:
            return 1

        host_country = self.VENUE_COUNTRY.get(venue)
        if team == host_country:
            return HOME_ADVANTAGE

        return CO_HOST_ADVANTAGE

    def expected_goals(self, home_team, away_team, venue = None, g = WORLD_CUP_BASELINE):

        """
        Expected goals for both sides.

        Parameters
        ----------
        home_team, away_team : str
            Team names, as indexed in team_stats.
        venue : str, optional
            Venue name, used for the host boost. None means no boost either way.
        g : float, default WORLD_CUP_BASELINE
            Corrects the goal level, which the ratings alone set too low for these fixtures.

        Returns
        -------
        tuple of float
            Expected goals for the home side and the away side.
        """

        home_attack = self.team_stats.loc[home_team, 'attack_rating']
        away_defense = self.team_stats.loc[away_team, 'defence_rating']
        away_attack = self.team_stats.loc[away_team, 'attack_rating']
        home_defense = self.team_stats.loc[home_team, 'defence_rating']

        x_home = (home_attack * away_defense) * g
        x_away = (away_attack * home_defense) * g

        x_home *= self.host_boost(home_team, venue) / self.host_boost(away_team, venue)
        x_away *= self.host_boost(away_team, venue) / self.host_boost(home_team, venue)

        return x_home, x_away

    def rho_correction(self, home_goals, away_goals, x_home, x_away, rho = RHO):

        """
        The Dixon-Coles correction factor for one scoreline.

        Independent Poisson produces too few 0-0 and 1-1 draws and too many 1-0 and 0-1 results.
        This scales those four cells and leaves every other scoreline alone. Only valid while
        the factor stays positive, which is what bounds how far the ratings can be spread.

        Parameters
        ----------
        home_goals, away_goals : int
            The scoreline being corrected.
        x_home, x_away : float
            Expected goals for each side.
        rho : float, default RHO
            Size of the correction.

        Returns
        -------
        float
            The factor to multiply that cell by, or 1 outside the four corrected cells.
        """
    
        if home_goals == 0 and away_goals == 0:
            return 1 - (x_home * x_away * rho)
        elif home_goals == 0 and away_goals == 1:
            return 1 + (x_home * rho)
        elif home_goals == 1 and away_goals == 0:
            return 1 + (x_away * rho)
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1

    def historical_avg_cards(self):

        """
        Set the average yellows and reds per team per game from the last three World Cups.

        Called once on construction. Cards don't affect a result, they only feed the fair play
        tiebreaker, so one tournament-wide rate is used rather than a per-team one.
        """

        bookings = load_csv('wc_bookings.csv')
        recent = bookings[bookings['tournament_id'].isin(self.CARD_TOURNAMENTS)]
    
        match_cards = recent.groupby('match_id').agg(
            yellows = ('yellow_card', 'sum'),
            reds = ('sending_off', 'sum')
        ).reset_index()

        self.avg_yellow = match_cards['yellows'].mean() / 2  # For average cards per team per game
        self.avg_red = match_cards['reds'].mean() / 2

    def _simulate_cards(self, x_yellow = None, x_red = None):

        # Draws yellows and reds for both sides from a Poisson at the historical rate.
        
        if x_yellow is None: x_yellow = self.avg_yellow
        if x_red is None: x_red = self.avg_red

        home_yellow = np.random.poisson(x_yellow)
        home_red = np.random.poisson(x_red)
        away_yellow = np.random.poisson(x_yellow)
        away_red = np.random.poisson(x_red)

        return home_yellow, home_red, away_yellow, away_red

    def score_matrix(self, x_home, x_away):

        """
        The probability of every scoreline up to 8-8, with the Dixon-Coles correction applied.

        Parameters
        ----------
        x_home, x_away : float
            Expected goals for each side.

        Returns
        -------
        np.ndarray
            A 9 by 9 matrix of probabilities summing to 1, home goals down, away goals across.
        """

        home_prob = stats.poisson.pmf(np.arange(0, 9), x_home)
        away_prob = stats.poisson.pmf(np.arange(0, 9), x_away)  # Calculates poisson possibility of each scoreline
        matrix = np.outer(home_prob, away_prob)  # Multiplies every element of home_poss against away_poss
            
        matrix[1][0] *= self.rho_correction(1, 0, x_home, x_away)
        matrix[0][1] *= self.rho_correction(0, 1, x_home, x_away)
        matrix[0][0] *= self.rho_correction(0, 0, x_home, x_away)
        matrix[1][1] *= self.rho_correction(1, 1, x_home, x_away)

        return matrix / matrix.sum()

    def simulate_regular_time(self, x_home, x_away):

        """
        Draw one scoreline from the score matrix.

        Parameters
        ----------
        x_home, x_away : float
            Expected goals for each side.

        Returns
        -------
        tuple of int
            Home goals and away goals.
        """

        flat_matrix = self.score_matrix(x_home, x_away).flatten()
        flat_matrix = np.clip(flat_matrix, 0, None)
        flat_matrix = flat_matrix / flat_matrix.sum()
        i = np.random.choice(len(flat_matrix), p = flat_matrix)

        home_goals = i // 9
        away_goals = i % 9
        
        return home_goals, away_goals

    def simulate_extra_time(self, home_team, away_team, venue = None):

        """
        Draw thirty minutes of extra time, at a third of the regular time rates.

        Parameters
        ----------
        home_team, away_team : str
            Team names, as indexed in team_stats.
        venue : str, optional
            Venue name, used for the host boost. None means no boost either way.

        Returns
        -------
        tuple of int
            Extra time goals and cards for both sides.
        """

        x_home, x_away = self.expected_goals(home_team, away_team, venue)
        x_home, x_away = x_home / 3, x_away / 3
        x_yellow, x_red = self.avg_yellow / 3, self.avg_red / 3
        
        et_home, et_away = self.simulate_regular_time(x_home, x_away)
        et_home_yellow, et_home_red, et_away_yellow, et_away_red = self._simulate_cards(x_yellow, x_red)

        return et_home, et_away, et_home_yellow, et_home_red, et_away_yellow, et_away_red

    def match_probabilities(self, home_team, away_team, venue = None, stage = 'group'):

        """
        Win, draw and loss probabilities for a fixture, without simulating it.

        Parameters
        ----------
        home_team, away_team : str
            Team names, as indexed in team_stats.
        venue : str, optional
            Venue name, used for the host boost. None means no boost either way.
        stage : {'group', 'knockout'}, default 'group'
            A knockout stage also returns the chance each side advances, which folds in extra
            time and treats a shootout as a coin flip.

        Returns
        -------
        dict
            Expected goals, the score matrix, and home_win, draw and away_win. Knockout adds
            home_advances and away_advances.
        """

        x_home, x_away = self.expected_goals(home_team, away_team, venue)
        matrix = self.score_matrix(x_home, x_away)

        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()

        result = {
            'x_home': x_home,
            'x_away': x_away, 
            'matrix': matrix,
            'home_win': home_win, 
            'draw': draw, 
            'away_win': away_win
        }

        if stage == 'knockout':
            extra = self.score_matrix(x_home / 3, x_away / 3)
            et_home, et_draw, et_away = (np.tril(extra, -1).sum(),
                                         np.trace(extra),
                                         np.triu(extra, 1).sum())
            result['home_advances'] = home_win + draw * (et_home + et_draw * 0.5)
            result['away_advances'] = away_win + draw * (et_away + et_draw * 0.5)

        return result

    def monte_carlo_gs(self, home_team, away_team, venue = None):

        """
        Play one group match, sampled from the scoreline distribution.

        Parameters
        ----------
        home_team, away_team : str
            Team names, as indexed in team_stats.
        venue : str, optional
            Venue name, used for the host boost.

        Returns
        -------
        dict
            Goals and cards for both sides, and the winning team or 'Draw'.
        """

        x_home, x_away = self.expected_goals(home_team, away_team, venue)
        home_goals, away_goals = self.simulate_regular_time(x_home, x_away)
        home_yellow, home_red, away_yellow, away_red = self._simulate_cards()
    
        if home_goals > away_goals:
            winning_team = home_team
        elif away_goals > home_goals:
            winning_team = away_team
        else:
            winning_team = 'Draw'
            
        return {
            'predicted_home_goals': home_goals,
            'predicted_away_goals': away_goals,
            'home_yellow_cards': home_yellow,
            'home_red_cards': home_red,
            'away_yellow_cards': away_yellow,
            'away_red_cards': away_red,
            'winning_team': winning_team,
        }

    def monte_carlo_ko(self, home_team, away_team, venue = None):

        """
        Play one knockout tie, through extra time and a shootout if needed.

        Level after ninety minutes goes to extra time. Level after that is settled by a coin
        flip.

        Parameters
        ----------
        home_team, away_team : str
            Team names, as indexed in team_stats.
        venue : str, optional
            Venue name, used for the host boost.

        Returns
        -------
        dict
            End of tie goals and cards for both sides, the winning team, and whether it went to
            extra time.
        """

        x_home, x_away = self.expected_goals(home_team, away_team, venue)
        home_goals, away_goals = self.simulate_regular_time(x_home, x_away)
        home_yellow, home_red, away_yellow, away_red = self._simulate_cards()

        extra_time = False
    
        if home_goals == away_goals:
            et_home, et_away, et_home_yellow, et_home_red, et_away_yellow, et_away_red = self.simulate_extra_time(home_team, away_team, venue)
            extra_time = True
            
            home_goals += et_home
            away_goals += et_away
            home_yellow += et_home_yellow 
            home_red += et_home_red
            away_yellow += et_away_yellow
            away_red += et_away_red
        
            if et_home > et_away:
                winning_team = home_team
            elif et_away > et_home:
                winning_team = away_team
            else:
                winning_team = home_team if np.random.random() > 0.5 else away_team
            
        elif home_goals > away_goals:
            winning_team = home_team
        else:
            winning_team = away_team
        
        return {
        'predicted_home_goals': home_goals,
        'predicted_away_goals': away_goals,
        'home_yellow_cards': home_yellow,
        'home_red_cards': home_red,
        'away_yellow_cards': away_yellow,
        'away_red_cards': away_red,
        'winning_team': winning_team,
        'extra_time': extra_time,
        'penalties': home_goals == away_goals
        }