"""
The tournament layer. Plays a whole World Cup, from the 72 group matches through to a champion,
and aggregates many of those into probabilities.

TournamentSimulator plays one tournament. TournamentForecast runs it many times and counts
what happened.
"""

import os
import collections
import json

import pandas as pd

from wcmodel.team_data import *
from wcmodel.match import *


# ---------- Tournament Engine ----------


class TournamentSimulator:

    """
    Plays one complete tournament, from the group stage to a champion.

    The methods run in order, starting with group_results and ending with run_knockout_stages.

    Parameters
    ----------
    match_sim : MatchSimulator
        The engine used to play each match.
    group_fixtures : pd.DataFrame
        The 72 group matches, with group, teams and venue.
    knockout_slots : pd.DataFrame
        The bracket template, saying which slot each match takes its teams from.

    Attributes
    ----------
    fifa_rank : pd.Series
        Final ranking, the last tiebreaker when everything else is level.
    third_place_table : dict
        Which combination of third placed groups feeds which Round of 32 slot.
    """

    def __init__(self, match_sim, group_fixtures, knockout_slots):
        
        self.match_sim = match_sim
        self.group_fixtures = group_fixtures
        self.knockout_slots = knockout_slots
        
        self.fifa_rank = load_csv('fifa_rankings_wc2026.csv').set_index('team')['rank']

        with open(os.path.join(DATA_DIR, 'third_place_table.json')) as f:
            self.third_place_table = json.load(f)

    def _write_results(self, predictions, i, result):

        # Writes one match result back into the predictions frame, column by column.

        for column, value in result.items():
            predictions.loc[i, column] = value

    def group_results(self, predictions):

        """
        Plays all 72 group matches and works out who goes through.

        Also sets group_rankings and qualified_thirds, which run_knockout_stages needs.

        Parameters
        ----------
        predictions : pd.DataFrame
            The fixture list, filled in with each result as it is played.

        Returns
        -------
        pd.DataFrame
            The same frame with goals, cards and a winner per match.
        """

        for i, match in self.group_fixtures.iterrows():
            result = self.match_sim.monte_carlo_gs(match['home_team'], match['away_team'], venue = match['venue'])
            self._write_results(predictions, i, result)

        count_columns = ['predicted_home_goals', 'predicted_away_goals', 'home_yellow_cards', 'home_red_cards', 'away_yellow_cards', 'away_red_cards']

        predictions[count_columns] = predictions[count_columns].astype(int)

        standings = self.group_standings(predictions)
        self.group_rankings = self.build_group_rankings(standings)
        self.qualified_thirds = self.best_third(standings).head(8)

        return predictions

    def group_standings(self, predictions):

        """
        Build every group table and sort it by FIFA's tiebreaker order.

        Order: points, goal difference, goals scored, head to head between the teams
        still level, fair play points, final FIFA ranking at the beginning of the tournament.

        Parameters
        ----------
        predictions : pd.DataFrame
            Played group matches, from group_results.

        Returns
        -------
        pd.DataFrame
            One row per team, sorted within each group, ready for build_group_rankings.
        """

        stats = {}

        for group_name in sorted(self.group_fixtures['group'].unique()):
            current_group = self.group_fixtures[self.group_fixtures['group'] == group_name]

            home_teams = set(current_group['home_team'])
            away_teams = set(current_group['away_team'])

            teams = sorted(home_teams | away_teams)

            for team in teams:
                stats[team] = {
                    'Pts' : 0,
                    'GD' : 0,
                    'GF' : 0,
                    'GA' : 0,
                    'TCS' : 0
                }

        for _, match in predictions.iterrows():

            hg, ag = match['predicted_home_goals'], match['predicted_away_goals']
            home, away = match['home_team'], match['away_team']

            stats[home]['GF'] += hg
            stats[home]['GA'] += ag
            stats[away]['GF'] += ag
            stats[away]['GA'] += hg
            stats[home]['GD'] = stats[home]['GF'] - stats[home]['GA']
            stats[away]['GD'] = stats[away]['GF'] - stats[away]['GA']
            
            if match['winning_team'] == home:
                stats[home]['Pts'] += 3
            elif match['winning_team'] == away:
                stats[away]['Pts'] += 3
            else:
                stats[home]['Pts'] += 1
                stats[away]['Pts'] += 1

            stats[home]['TCS'] += (-1 * match['home_yellow_cards']) + (-3 * match['home_red_cards'])
            stats[away]['TCS'] += (-1 * match['away_yellow_cards']) + (-3 * match['away_red_cards'])
            
        standings = pd.DataFrame(stats).T
    
        home_map = self.group_fixtures.set_index('home_team')['group']
        away_map = self.group_fixtures.set_index('away_team')['group']

        standings['rank'] = standings.index.map(self.fifa_rank)
    
        team_groups = pd.concat([home_map, away_map])
        team_groups = team_groups[~team_groups.index.duplicated(keep = 'first')]
        standings['group'] = team_groups
       
        standings['h2h'] = 0          
        for _, block in standings.groupby(['group', 'Pts']):
            if len(block) > 1:                         
                tied_teams = block.index.tolist()      
                order = self.h2h_tiebreaker(tied_teams, predictions) 
                for team, position in order.items():
                    standings.loc[team, 'h2h'] = position
        
        standings = standings.sort_values(['group', 'Pts', 'h2h', 'GD', 'GF', 'TCS', 'rank'], 
                                          ascending = [True, False, True, False, False, False, True])

        return standings

    def h2h_tiebreaker(self, tied_teams, predictions):

        """
        Ranks teams level on points and goal difference by the matches between themselves.

        Only fixtures involving the tied teams count, so three teams level are separated by
        that mini table rather than by their full records.

        Parameters
        ----------
        tied_teams : list of str
            The teams still level.
        predictions : pd.DataFrame
            Played group matches.

        Returns
        -------
        dict
            Points, goal difference, goals for and against from the matches between them.
        """

        tied = (predictions['home_team'].isin(tied_teams) & predictions['away_team'].isin(tied_teams))
        tied_group = predictions[tied]

        stats = {team: {
            'Pts': 0,
            'GD': 0,
            'GF': 0, 
            'GA': 0
        } for team in tied_teams}

        for _, match in tied_group.iterrows():

            hg, ag = match['predicted_home_goals'], match['predicted_away_goals']
            home, away = match['home_team'], match['away_team']

            stats[home]['GF'] += hg
            stats[home]['GA'] += ag
            stats[away]['GF'] += ag
            stats[away]['GA'] += hg
            stats[home]['GD'] = stats[home]['GF'] - stats[home]['GA']
            stats[away]['GD'] = stats[away]['GF'] - stats[away]['GA']
            
            if match['winning_team'] == home:
                stats[home]['Pts'] += 3
            elif match['winning_team'] == away:
                stats[away]['Pts'] += 3
            else:
                stats[home]['Pts'] += 1
                stats[away]['Pts'] += 1

        tied_table = pd.DataFrame(stats).T
        tied_table = tied_table.sort_values(['Pts', 'GD', 'GF'], ascending = False)

        order_map = {}
        position = 0
        previous_record = None
        for team, row in tied_table.iterrows():
            record = (row['Pts'], row['GD'], row['GF'])
            if previous_record is not None and record != previous_record:
                position += 1     
            
            order_map[team] = position
            previous_record = record

        return order_map
        
    def build_group_rankings(self, standings):

        """
        Turns the sorted standings into a finishing order for each group.

        Parameters
        ----------
        standings : pd.DataFrame
            From group_standings, already in order.

        Returns
        -------
        dict of {str : list of str}
            Group letter to its teams, first to fourth.
        """

        group_rankings = {}

        for group_name in sorted(self.group_fixtures["group"].unique()):
            teams = standings[standings['group'] == group_name].index.tolist()
            group_rankings[group_name] = teams  # Stores the teams into the group_rankings variable.

        return group_rankings

    def best_third(self, standings):

        """
        Ranks the twelve third placed teams, of which the top eight go through.

        Sorted on points, goal difference, goals scored, total cards, then FIFA ranking.

        Parameters
        ----------
        standings : pd.DataFrame
            From group_standings.

        Returns
        -------
        pd.DataFrame
            The twelve third placed teams in order.
        """

        third_place_teams = [teams[2] for teams in self.build_group_rankings(standings).values()]
        third_place_order = standings[standings.index.isin(third_place_teams)].sort_values(['Pts', 'GD', 'GF', 'TCS', 'rank'], 
                                                                                           ascending = [False, False, False, False, True])

        return third_place_order

    def get_team_slot(self, slot):

        """
        Finds the team that fills a group-based bracket slot.

        Parameters
        ----------
        slot : str
            Slot text from the bracket template, such as 'Winner Group A' or 'Runner-up Group B'.

        Returns
        -------
        str
            The team currently occupying that slot.
        """
    
        words = slot.split()

        if slot.startswith("Winner"):
            position = 0
            group_name = words[-1]
        
        elif slot.startswith("Runner-up"):
            position = 1
            group_name = words[-1]
        
        elif slot.startswith("Best 3rd"):
            return self.third_assignments[slot]

        return self.group_rankings[group_name][position]

    def assign_best_thirds(self):
        
        """
        Maps each qualifying group's 3rd-place team to its FIFA-assigned R32 opponent slot.
        """

        qualified_groups = sorted(self.qualified_thirds['group'].tolist())
        key = ''.join(qualified_groups)
        row = self.third_place_table[key]

        self.third_assignments = {}
        for fixed_slot, third_group_code in row.items():
            third_group = third_group_code[-1]
            team = self.qualified_thirds[self.qualified_thirds['group'] == third_group].index[0]
            self.third_assignments[fixed_slot] = team

    def run_knockout_stages(self, predictions):

        """
        Plays the bracket from the Round of 32 to the final.

        Each slot is a group finisher, a qualified third, or the winner or loser of an earlier
        match, so ties are resolved in order and the results fed forward.

        Parameters
        ----------
        predictions : pd.DataFrame
            Played group matches, used to fill the bracket.

        Returns
        -------
        pd.DataFrame
            One row per knockout tie, with the result and who went through.
        """

        match_winners = {}
        match_losers = {}
        self.assign_best_thirds()

        for i, slot in self.knockout_slots.iterrows():
            match_id = slot['match_id']

            if slot["slot_home"].startswith(("Winner Match", "Loser Match")):
                home_team = self.resolve_match_slot(slot["slot_home"], match_winners, match_losers)
            else:
                home_team = self.get_team_slot(slot["slot_home"])

            if slot["slot_away"].startswith("Best 3rd"):
                group_letter = slot["slot_home"].split()[-1]
                away_team = self.third_assignments['1' + group_letter]
            elif slot["slot_away"].startswith(("Winner Match", "Loser Match")):
                away_team = self.resolve_match_slot(slot["slot_away"], match_winners, match_losers)
            else:
                away_team = self.get_team_slot(slot["slot_away"])

            result = self.match_sim.monte_carlo_ko(home_team, away_team, venue = slot['venue'])
            winner = result['winning_team']
            loser = away_team if winner == home_team else home_team
            match_winners[match_id] = winner
            match_losers[match_id]  = loser

            predictions.loc[i, 'predicted_home_team'] = home_team
            predictions.loc[i, 'predicted_away_team'] = away_team
            self._write_results(predictions, i, result)
    
        count_columns = ['predicted_home_goals', 'predicted_away_goals',
                         'home_yellow_cards', 'home_red_cards',
                         'away_yellow_cards', 'away_red_cards']

        predictions[count_columns] = predictions[count_columns].astype(int)

        return predictions

    def resolve_match_slot(self, slot, match_winners, match_losers):
        
        """
        Resolves slots like 'Winner Match 73' or 'Loser Match 101' from earlier knockout results.

        Parameters
        ----------
        slot : str
            Slot text from the bracket template.
        match_winners, match_losers : dict of {int : str}
            Who won and lost each knockout match played so far.

        Returns
        -------
        str
            The team that fills the slot.
        """
        
        match_id = int(slot.split()[-1])
        if slot.startswith("Winner"):
            return match_winners[match_id]
        else:
            return match_losers[match_id]


# ---------- Forecast ----------


class TournamentForecast:

    """
    Runs many tournaments and turns the results into probabilities.

    Every published figure is counted from full tournaments rather than calculated per match,
    so the tables, tiebreakers and bracket routing all apply.

    Parameters
    ----------
    tournament_sim : TournamentSimulator
        The simulator to sample from.

    Attributes
    ----------
    ROUND_DEPTH : dict
        How far each round is from the group stage, used to record a team's best finish.
    teams : list of str
        The 48 finalists.
    team_group : dict of {str : str}
        Which group each team is in.
    """

    ROUND_DEPTH = {
        'group stage' : 0,
        'Round of 32' : 1,
        'Round of 16' : 2,
        'Quarter-final' : 3,
        'Semi-final' : 4,
        'Third-place playoff' : 4,
        'Final'  : 5,
        'Champion' : 6,
    }

    def __init__(self, tournament_sim):

        self.tournament_sim = tournament_sim
        fixtures = tournament_sim.group_fixtures
        self.teams = sorted(set(fixtures['home_team']) | set(fixtures['away_team']))

        team_groups = pd.concat([fixtures.set_index('home_team')['group'], fixtures.set_index('away_team')['group']])
        self.team_group = team_groups[~team_groups.index.duplicated()].to_dict()

    def sample_tournament(self):

        """
        Plays one complete tournament.

        Returns
        -------
        tuple
            How far each team got, and how each team finished its group.
        """

        sim = self.tournament_sim

        predictions = sim.group_fixtures.copy()
        predictions = sim.group_results(predictions)

        qualified_thirds = set(sim.qualified_thirds.index)

        group_finish = {}
        for _, order in sim.group_rankings.items():
            group_finish[order[0]] = 'Win group'
            group_finish[order[1]] = 'Runner-up'
            group_finish[order[2]] = 'Best third' if order[2] in qualified_thirds else 'Eliminated'
            group_finish[order[3]] = 'Eliminated'

        knockout = sim.knockout_slots.copy()
        knockout = sim.run_knockout_stages(knockout)

        reached = {team : 'group stage' for team in self.teams}
        for _, match in knockout.iterrows():
            for team in (match['predicted_home_team'], match['predicted_away_team']):
                if self.ROUND_DEPTH[match['round']] > self.ROUND_DEPTH[reached[team]]:
                    reached[team] = match['round']

        final = knockout[knockout['round'] == 'Final'].iloc[0]
        reached[final['winning_team']] = 'Champion'

        slots = {}
        for match_id, home, away, winner in zip(knockout['match_id'], knockout['predicted_home_team'], knockout['predicted_away_team'], knockout['winning_team']):
            slots[match_id] = (home, away, winner)

        scorelines = [tuple(sorted((hg, ag), reverse = True))
                  for hg, ag in zip(predictions['predicted_home_goals'], predictions['predicted_away_goals'])]
        scorelines += [tuple(sorted((hg, ag), reverse = True))
                   for hg, ag in zip(knockout['predicted_home_goals'], knockout['predicted_away_goals'])]

        return reached, group_finish, slots, scorelines

    def run(self, simulations):

        """
        Runs the tournament `simulations` times and records how each team did.

        Parameters
        ----------
        simulations : int
            How many tournaments to play.

        Returns
        -------
        TournamentForecast
            Itself, so the table methods can be chained onto the call.
        """

        self.simulations = simulations
        self.counts = {team : collections.Counter() for team in self.teams}
        self.group_counts = {team : collections.Counter() for team in self.teams}
        self.scoreline_counts = collections.Counter()

        self.slot_counts = {match_id : {'home' : collections.Counter(),
                                        'away' : collections.Counter(),
                                        'winner': collections.Counter()}
                            for match_id in self.tournament_sim.knockout_slots['match_id']}

        for _ in range(simulations):
            reached, group_finish, slots, scorelines = self.sample_tournament()

            for team, round_reached in reached.items():
                self.counts[team][round_reached] += 1
            for team, finish in group_finish.items():
                self.group_counts[team][finish] += 1

            self.scoreline_counts.update(scorelines)

            for match_id, (home, away, winner) in slots.items():
                self.slot_counts[match_id]['home'][home]     += 1
                self.slot_counts[match_id]['away'][away]     += 1
                self.slot_counts[match_id]['winner'][winner] += 1

        return self

    def progression_table(self):

        """
        Probability of reaching each stage or better, deepest first.

        Returns
        -------
        pd.DataFrame
            One row per team, one column per round.
        """

        ROUNDS = {
            'Win' : 'Champion',
            'Reach Final' : 'Final',
            'Reach Semi' : 'Semi-final',
            'Reach Quarter' : 'Quarter-final',
            'Reach R16' : 'Round of 16',
        }

        table = {}
        for team, counter in self.counts.items():
            table[team] = {
                label : sum(n for reached, n in counter.items()
                            if self.ROUND_DEPTH[reached] >= self.ROUND_DEPTH[round_name]) / self.simulations
                for label, round_name in ROUNDS.items()
            }

        table = pd.DataFrame(table).T.sort_values('Win', ascending = False)
        table.index.name = 'Team'

        return table

    def group_table(self):

        """
        How each team finishes its group, ordered by group then escape probability.

        'Escape' is the chance of reaching the knockouts by any of the three routes.

        Returns
        -------
        pd.DataFrame
            One row per team, with each route and the combined escape probability.
        """

        ROUTES = ['Win group', 'Runner-up', 'Best third']

        table = {}
        for team, counter in self.group_counts.items():
            row = {route : counter[route] / self.simulations for route in ROUTES}
            row['Escape'] = sum(row[route] for route in ROUTES)
            row['Group'] = self.team_group[team]
            table[team] = row

        table = pd.DataFrame(table).T
        table.index.name = 'Team'
        table[['Win group', 'Escape']] = table[['Win group', 'Escape']].astype(float)

        return (table.sort_values(['Group', 'Escape', 'Win group'], ascending = [True, False, False])
        [['Group', 'Win group', 'Escape']])

    def likely_bracket(self):

        """
        Bracket by progression odds. Each group's three qualifiers are the teams most likely to
        escape, with the top two ordered by how often they win the group. Each tie then goes to
        whichever team is more likely to reach the following round across all simulations.
        The third place playoff has no next round, so it compares final reach probability instead.

        Returns
        -------
        pd.DataFrame
            One row per knockout tie, with both teams, the likelier winner and its probability.
        """

        NEXT_ROUND = {
            'Round of 32' : 'Reach R16',
            'Round of 16' : 'Reach Quarter',
            'Quarter-final' : 'Reach Semi',
            'Semi-final' : 'Reach Final',
            'Third-place playoff' : 'Reach Final',
            'Final' : 'Win'
        }

        sim = self.tournament_sim
        progression = self.progression_table()
        escape = self.group_table()['Escape']
        counts  = self.group_counts

        teams_by_group = {}
        for team, group in self.team_group.items():
            teams_by_group.setdefault(group, []).append(team)

        occupant, thirds = {}, {}

        for group, teams in teams_by_group.items():
            top_two = sorted(teams, key = lambda t: -escape[t])[:2]
            first   = max(top_two, key = lambda t: counts[t]['Win group'])
            second  = next(t for t in top_two if t != first)

            occupant[f'Winner Group {group}']    = first
            occupant[f'Runner-up Group {group}'] = second
            thirds[group] = max((t for t in teams if t not in top_two), 
                                key = lambda t: escape[t])

        best_eight = sorted(thirds, key = lambda g: -escape[thirds[g]])[:8]
        sim.qualified_thirds = pd.DataFrame({'group' : best_eight}, index = [thirds[g] for g in best_eight])
        sim.assign_best_thirds()

        winners, losers, rows = {}, {}, []

        def team_for_slot(slot_text, home_slot_text = None):

            if slot_text.startswith('Best 3rd'):
                return sim.third_assignments['1' + home_slot_text.split()[-1]]
            if slot_text.startswith('Winner Match'):
                return winners[int(slot_text.split()[-1])]
            if slot_text.startswith('Loser Match'):
                return losers[int(slot_text.split()[-1])]
        
            return occupant[slot_text]

        for _, slot in sim.knockout_slots.iterrows():
            column = NEXT_ROUND.get(slot['round'])
            if column is None:
                continue

            home = team_for_slot(slot['slot_home'], slot['slot_home'])
            away = team_for_slot(slot['slot_away'], slot['slot_home'])

            home_prob = progression.loc[home, column]
            away_prob = progression.loc[away, column]

            winner = home if home_prob >= away_prob else away
            winners[slot['match_id']] = winner
            losers[slot['match_id']]  = away if winner == home else home

            rows.append({
                'Match' : slot['match_id'],
                'Round' : slot['round'],
                'Home' : home,
                'Away' : away,
                'Winner' : winner,
                'Winner reach %' : max(home_prob, away_prob)
            })

        return pd.DataFrame(rows).set_index('Match')

    def scoreline_table(self):

        """
        How often each scoreline came up, counting scorelines such as 2-1 and 1-2 together.

        Returns
        -------
        pd.DataFrame
            One row per scoreline, with a count and a probability.
        """

        total = sum(self.scoreline_counts.values())

        table = pd.DataFrame(
            [{'higher_score' : hi, 
              'lower_score' : lo, 
              'count' : n, 
              'probability' : n / total}
            for (hi, lo), n in self.scoreline_counts.items()]
        )

        table['scoreline'] = table['higher_score'].astype(str) + '-' + table['lower_score'].astype(str)

        return table.sort_values('count', ascending = False).reset_index(drop = True)
