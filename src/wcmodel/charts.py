"""
Every figure and styled table in the project.

Public plot_ functions take the run output or the simulator, draw the chart, save a PNG to
images/ and return the figure. Everything prefixed with an underscore is a drawing helper for
one of them.

Colours are defined once at the top so the charts read as a set.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from scipy.signal import fftconvolve

from wcmodel.team_data import *


# ---------- Colour palette ----------


BACKGROUND = '#ffffff'
TEXT    = '#0b0b0b'
TEXT2   = '#52514e'
GRIDLINE   = '#e4e3df'
WIN_BG = '#1c5cab'
WIN_FG = '#ffffff'
BOX_BG = '#eef1f5'
LINE   = '#c4c3be'
OUTLINE = '#8e9296'
BAR_BG = '#9db8d8'
GOLD   = '#c9a227'
SILVER = '#a8a9ad'
BRONZE = '#a97142'
POLY = '#ff7f0e'


# ---------- Chart Building ----------


def style_table(table, group_size = None, percent = True):

    """
    Formats a table for display, with readable spacing and consistent numbers.

    Parameters
    ----------
    table : pd.DataFrame
        The table to style.
    group_size : int, optional
        Draw a dividing line every n rows, used to separate groups.
    percent : bool, default True
        Format every numeric column as a percentage to 1dp. False formats floats to 2dp and
        leaves integers as they are.

    Returns
    -------
    pd.io.formats.style.Styler
        Ready to pass to display.
    """

    table = table.rename_axis(None)

    numeric = table.select_dtypes('number' if percent else 'float').columns
    number_format = '{:.1%}' if percent else '{:.2f}'

    styles = [
        {'selector' : 'th, td', 'props' : [('padding', '5px 18px')]},
        {'selector' : 'td', 'props' : [('text-align', 'right')]},
        {'selector' : 'th.col_heading', 'props' : [('text-align', 'right')]},
        {'selector' : 'th.row_heading', 'props' : [('text-align', 'left')]},
    ]

    if group_size:
        styles.append({
            'selector' : f'tbody tr:nth-child({group_size}n+1) th, tbody tr:nth-child({group_size}n+1) td',
            'props'    : [('border-top', '2px solid #c9c8c3')],
        })

    return (table.style
        .format({column : number_format for column in numeric})
        .set_table_styles(styles))


def _add_title(ax, title, subtitle = None, title_fontsize = 16, subtitle_fontsize = 10, 
               pad = 58, subtitle_x = 0.5, subtitle_y = 1.075, **kwargs):
    """
    Draws a standardised title and optional subtitle above the given axes.
    """

    ax.set_title(title, loc = 'center', fontsize = title_fontsize,
                 fontweight = 'bold', color = TEXT, pad = pad, **kwargs)

    if subtitle:
        ax.text(subtitle_x, subtitle_y, subtitle,
                transform = ax.transAxes, ha = 'center', fontsize = subtitle_fontsize,
                color = TEXT2, va = 'top', linespacing = 1.6)
        

def _save_png(fig, fig_name):

    "Saves a picture of the chart."

    path = os.path.join(IMAGE_DIR, f"{fig_name}.png")
    fig.savefig(path, dpi = 300, bbox_inches = 'tight')

    return path

# ---------- Team Ratings Chart ----------


def _label_team_ratings(ax, teams, x_values, y_values):

    #Slightly adjusts labels of teams whose labels would otherwise overlap.
    
    OVERLAP = {
        'DR Congo' : (6.5, -10),
        'Uzbekistan' : (6.5, -8),
        'Croatia' : (6.5, -10),
        'Japan' : (6.5, -10),
        'Paraguay' : (6.5, -10),
        'Scotland' : (-6.5, 10),
        'South Korea' : (-6.5, -4.5),
        'Czech Republic' : (-6.5, 1.5),
        'Cabo Verde' : (-6.5, 10),
        'New Zealand' : (6.5, -10),
        'England' : (6.5, -10),
        'Belgium' : (6.5, -10),
        'Portugal' : (6.5, -10),
        'Haiti' : (6.5, -10),
        'Switzerland' : (-6.5, 10),
        "Côte d'Ivoire" : (6.5, 0),
        'Bosnia and Herzegovina' : (-6.5, -4.5),
        'Argentina' : (6.5, 6),
        'Germany' : (-6.5, 4.5),
        'Uzbekistan' : (6.5, -8),
        'Tunisia' : (-6.5, 10)
    }
    
    LEFT = {'South Korea', 
            'Czech Republic',
            'Switzerland',
            'Scotland',
            'Cabo Verde',
            'Germany',
            'Tunisia',
            'Bosnia and Herzegovina'
    }
    
    for team in teams:
        if team in LEFT:
            ha, va = 'right', 'top'
        else:
            ha, va = 'left', 'baseline'
    
        ax.annotate(team, (x_values[team], y_values[team]),
                    xytext = OVERLAP.get(team, (6.5, 4.5)),
                    textcoords = 'offset points', fontsize = 8.4, color = TEXT,
                    ha = ha, va = va, zorder = 4)


def plot_team_ratings(team_stats, label_teams = None, top_elo = None, pad = 1.10, 
                      title = 'Team Ratings: Attack against Defence',
                      subtitle = 'The core inputs to every simulated match. Attack and defence are weighted by opponent strength,\n'
                        'tournament importance and recency, then blended with an Elo prior and squad market values.'):

    """
    Plots attack against defence for all 48 finalists, coloured by Elo.

    Parameters
    ----------
    team_stats : pd.DataFrame
        From build_all_team_data, with attack_rating, defence_rating and current_elo.
    label_teams : set of str, optional
        Which teams to label. Defaults to all of them, or to top_elo if that is given.
    top_elo : int, optional
        Label this many teams by Elo, plus the three extremes on each axis. Ignored when
        label_teams is given.
    pad : float, default 1.10
        How much room to leave around the outermost points.
    title, subtitle : str, optional
        Chart text.

    Returns
    -------
    plt.Figure
        The figure, also saved to images/team_ratings.png.
    """

    attack  = team_stats['attack_rating']
    defence = team_stats['defence_rating']
    elo = team_stats['current_elo']

    fig, ax = plt.subplots(figsize = (12.2, 10.4), facecolor = BACKGROUND)

    ax.axvline(defence.median(), color = GRIDLINE, lw = 1.1, zorder = 1)
    ax.axhline(attack.median(),  color = GRIDLINE, lw = 1.1, zorder = 1)

    dots = ax.scatter(defence, attack, c = elo, cmap = 'YlGnBu', s = 70,
                      edgecolor = OUTLINE, linewidth = 1.3, zorder = 3)

    if label_teams is None:
        if top_elo is None:
            label_teams = set(team_stats.index)
        else:
            label_teams = (set(elo.nlargest(top_elo).index)
                         | set(attack.nlargest(3).index)   | set(attack.nsmallest(3).index)
                         | set(defence.nsmallest(3).index) | set(defence.nlargest(3).index))

    _label_team_ratings(ax, label_teams, defence, attack)
            
    # Centre both axes on the field average so the crosshair sits in the middle
    for series, setter, invert in ((defence, ax.set_xlim, True), (attack, ax.set_ylim, False)):
        reach = max(series.max() - series.mean(), series.mean() - series.min()) * pad
        low, high = series.mean() - reach, series.mean() + reach
        setter((high, low) if invert else (low, high))

    ax.set_xlabel('Defence rating (smaller is better)',
                  fontsize = 11.5, color = TEXT, labelpad = 9)
    ax.set_ylabel('Attack rating (bigger is better)',
                  fontsize = 11.5, color = TEXT, labelpad = 9)
    ax.tick_params(colors = TEXT2, labelsize = 10, length = 0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(GRIDLINE)

    bar = fig.colorbar(dots, ax = ax, pad = 0.015, fraction = 0.035)
    bar.set_label('Elo rating', fontsize = 10.5, color = TEXT2)
    bar.ax.tick_params(colors = TEXT2, labelsize = 9, length = 0)
    bar.outline.set_visible(False)

    _add_title(ax, title, subtitle)

    fig.tight_layout()
    _save_png(fig, 'team_ratings')

    return fig


# ---------- Modal Bracket Chart ----------


BOX_W, BOX_H, COL_GAP = 0.86, 0.78, 1.0

# Names that don't fit inside a bracket box at full length.
SHORT_NAMES = {'Bosnia and Herzegovina' : 'Bosnia'}

def _feeders(knockout_slots):

    # Which earlier matches feed each bracket slot, read off the knockout template.

    feeders = {}
    for _, slot in knockout_slots.iterrows():
        feeder_ids = [int(slot[column].split()[-1])
                for column in ('slot_home', 'slot_away')
                if slot[column].startswith('Winner Match')]
        feeders[slot['match_id']] = feeder_ids
    return feeders

def _subtree(match_id, feeders):

    # Every match that eventually feeds this one, used to lay out one half of the bracket.

    out = {match_id}
    for id in feeders[match_id]:
        out |= _subtree(id, feeders)
    return out


def _assign_positions(match_id, feeders, y, counter):

    # Vertical position of each match, placing first-round ties in order and every later tie
    # midway between the two it feeds from.

    feeder_ids = feeders[match_id]
    if not feeder_ids:
        y[match_id] = counter[0]
        counter[0] += 2.6
        return y[match_id]

    positions = [_assign_positions(feeder_id, feeders, y, counter) for feeder_id in feeder_ids]
    y[match_id] = sum(positions) / len(positions)
    return y[match_id]


def _bracket_layout(bracket, knockout_slots):

    # Turns the bracket into coordinates: which side of the page each match sits on, how deep
    # it is, and where vertically.

    ROUND_ORDER = ['Round of 32', 
                   'Round of 16', 
                   'Quarter-final', 
                   'Semi-final', 
                   'Final'
    ]

    feeders = _feeders(knockout_slots)

    final_id = bracket.index[bracket['Round'] == 'Final'][0]
    left_id, right_id = feeders[final_id]

    y = {}
    for semi_final in (left_id, right_id):
        _assign_positions(semi_final, feeders, y, [0])

    depth = {name : i for i, name in enumerate(ROUND_ORDER)}
    left_tree = _subtree(left_id, feeders)
    side = {match_id : ('left' if match_id in left_tree else 'right') for match_id in y}

    y[final_id] = (y[left_id] + y[right_id]) / 2
    side[final_id] = 'left'

    return feeders, y, side, depth, final_id


_FLAG_CACHE = {}

def _flag(team, zoom):

    """
    Cached OffsetImage for a team's flag, or None if the file is missing.
    """

    if team not in _FLAG_CACHE:
        code = FLAG_CODES.get(team)
        path = os.path.join(FLAG_DIR, f'{code}.png') if code else None
        _FLAG_CACHE[team] = plt.imread(path) if path and os.path.exists(path) else None

    image = _FLAG_CACHE[team]

    return OffsetImage(image, zoom = zoom) if image is not None else None


def _draw_team_box(ax, team, box_x, box_y, highlight, flags):

    # One team's box, with flag, name and a highlight if they went through.

    ax.add_patch(FancyBboxPatch((box_x - BOX_W / 2, box_y - BOX_H / 2), BOX_W, BOX_H,
                                boxstyle = 'round,pad = 0,rounding_size = 0.06',
                                facecolor = WIN_BG if highlight else BOX_BG,
                                edgecolor = LINE, linewidth = 0.8, zorder = 3))

    flag = _flag(team, 0.20) if flags else None
    if flag is not None:
        ax.add_artist(AnnotationBbox(flag, (box_x - BOX_W / 2 + 0.13, box_y), frameon = False, zorder = 5))

    ax.text(box_x + (0.10 if flag is not None else 0), box_y, SHORT_NAMES.get(team, team),
            ha = 'center', va = 'center', fontsize = 8.4,
            color = WIN_FG if highlight else TEXT,
            fontweight = 'bold' if highlight else 'normal', zorder = 4)


def _draw_bracket_tree(ax, bracket, feeders, y, side, depth, flags):

    # Draws one half of the bracket, working inwards from the Round of 32 to the final.

    ROUND_ORDER = list(depth)

    def x_for(round_name, which_side):

        # Horizontal position of a round, measured outwards from the final in the middle.
        column = depth[round_name]
        return column * COL_GAP if which_side == 'left' else (8 - column) * COL_GAP

    for match_id, match in bracket.iterrows():
        which_side = side[match_id]
        match_x = x_for(match['Round'], which_side)

        if match['Round'] in ('Round of 32', 'Final'):
            slots = [(match['Home'], y[match_id] + 0.5), (match['Away'], y[match_id] - 0.5)]
        else:
            slots = [(bracket.loc[feeder_id, 'Winner'], y[feeder_id]) for feeder_id in feeders[match_id]]

        for team, box_y in slots:
            _draw_team_box(ax, team, match_x, box_y, team == match['Winner'], flags)

        if match['Round'] == 'Final':
            continue

        next_x = x_for(ROUND_ORDER[depth[match['Round']] + 1], which_side)
        edge = BOX_W / 2 if which_side == 'left' else -BOX_W / 2
        mid_x = (match_x + next_x) / 2

        for _, box_y in slots:
            ax.plot([match_x + edge, mid_x], [box_y, box_y], color = LINE, lw = 0.9, zorder = 1)
        ax.plot([mid_x, mid_x], [slots[0][1], slots[1][1]], color = LINE, lw = 0.9, zorder = 1)
        ax.plot([mid_x, next_x - edge], [y[match_id], y[match_id]], color = LINE, lw = 0.9, zorder = 1)

    top = max(y.values()) + 1.6
    for round_name in ROUND_ORDER:
        for which_side in ('left', 'right'):
            if round_name == 'Final' and which_side == 'right':
                continue
            ax.text(x_for(round_name, which_side), top, round_name.upper(), ha = 'center',
                    va = 'bottom', fontsize = 9, color = TEXT2, fontweight = 'bold')

    return x_for, top


def _draw_results_panel(ax, centre_x, y, final_id, playoff, champion, runner_up, title_odds, flags):

    # The podium, the third-place playoff and the title probability bars below the bracket.

    PODIUM_BASE = -3.6
    BAR_TOP, ROW_GAP = -5.6, 0.86

    third = None
    if not playoff.empty:
        tie = playoff.iloc[0]
        playoff_y = y[final_id] - 4.0

        ax.text(centre_x, playoff_y + 1.05, 'THIRD-PLACE PLAYOFF', ha = 'center',
                va = 'center', fontsize = 9, color = TEXT2, fontweight = 'bold')
        for team, box_y in ((tie['Home'], playoff_y + 0.45), (tie['Away'], playoff_y - 0.45)):
            _draw_team_box(ax, team, centre_x, box_y, team == tie['Winner'], flags)

        third = tie['Winner']

    steps = [(2, runner_up, SILVER, -0.92, 1.05),
             (1, champion,  GOLD,    0.00, 1.70),
             (3, third,     BRONZE,  0.92, 0.78)]

    ax.text(centre_x, PODIUM_BASE + 3.55, 'P R E D I C T E D   M E D A L L I S T S', ha = 'center',
            va = 'center', fontsize = 9.5, color = TEXT2, fontweight = 'bold')

    for rank, team, colour, offset_x, height in steps:
        if team is None:
            continue

        step_x = centre_x + offset_x

        ax.add_patch(FancyBboxPatch((step_x - 0.42, PODIUM_BASE), 0.84, height,
                                    boxstyle = 'round,pad = 0,rounding_siz e= 0.05',
                                    facecolor = colour, edgecolor = 'none', zorder = 3))
        ax.text(step_x, PODIUM_BASE + height / 2, str(rank), ha = 'center', va = 'center',
                fontsize = 17, color = WIN_FG, fontweight = 'bold', zorder = 4)
        ax.text(step_x, PODIUM_BASE + height + 0.30, team, ha = 'center', va = 'center',
                fontsize = 13 if rank == 1 else 11.5, color = TEXT,
                fontweight = 'bold' if rank == 1 else 'normal', zorder = 4)

        flag = _flag(team, 0.30) if flags else None
        if flag is not None:
            ax.add_artist(AnnotationBbox(flag, (step_x, PODIUM_BASE + height + 1.02),
                                         frameon = False, zorder = 5))

    if title_odds is None:
        return -6.2

    top_eight = title_odds.head(8)
    bar_left = centre_x - 0.62
    scale = 2.0 / top_eight.iloc[0]

    ax.text(centre_x, BAR_TOP + 1.0, 'TITLE PROBABILITY', ha = 'center', va = 'center',
            fontsize = 13, color = TEXT2, fontweight = 'bold')

    for i, (team, probability) in enumerate(top_eight.items()):
        row_y = BAR_TOP - i * ROW_GAP

        ax.text(bar_left - 0.10, row_y, team, ha = 'right', va = 'center',
                fontsize = 12.5, color = TEXT)
        ax.add_patch(FancyBboxPatch((bar_left, row_y - 0.31),
                                    max(probability * scale, 0.01), 0.62, 
                                    boxstyle = 'round,pad = 0, rounding_size = 0.05',
                                    facecolor = WIN_BG if team == champion else BAR_BG, 
                                    edgecolor = 'none', zorder = 3))
        ax.text(bar_left + probability * scale + 0.10, row_y, f'{probability:.1%}',
                ha = 'left', va = 'center', fontsize = 12.5, color = TEXT,
                fontweight = 'bold' if team == champion else 'normal')

    return BAR_TOP - (len(top_eight) - 1) * ROW_GAP - 1.0


def plot_bracket(bracket, knockout_slots, title_odds = None, flags = True, save = True,
                 title = 'FIFA World Cup 2026: Predicted Knockout Bracket',
                 subtitle = 'Based on 100,000 simulated tournaments, each tie shows the team most likely to progress,\n'
                            'so this is 32 separate calls rather than one predicted tournament. '
                            'Inputs frozen 10 June 2026.'):

    """
    Draws the knockout bracket, with a medallists panel and title probability bars.

    Each tie shows the team most likely to progress from it across every simulation, so this is
    32 separate calls rather than one predicted tournament.

    Parameters
    ----------
    bracket : pd.DataFrame
        From likely_bracket, either indexed by match id or with it as a column.
    knockout_slots : pd.DataFrame
        The bracket template, used to work out which match feeds which.
    title_odds : pd.Series, optional
        Title probability per team, for the bars at the bottom.
    flags : bool, default True
        Draw flag images next to team names.
    save : bool, default True
        Write the PNG to images/.
    title, subtitle : str, optional
        Chart text.

    Returns
    -------
    plt.Figure
        The figure, also saved to images/likely_bracket.png.
    """

    # A bracket read back from CSV has the match ids as a column; the live forecast has them
    # as the index. The layout looks them up in the index, so normalise either shape here.
    if 'Match' in bracket.columns:
        bracket = bracket.set_index('Match')

    playoff = bracket[bracket['Round'] == 'Third-place playoff']
    bracket = bracket[bracket['Round'] != 'Third-place playoff']

    feeders, y, side, depth, final_id = _bracket_layout(bracket, knockout_slots)

    fig, ax = plt.subplots(figsize = (16, 12.0), facecolor = BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    x_for, top = _draw_bracket_tree(ax, bracket, feeders, y, side, depth, flags)
    centre_x = x_for('Final', 'left')

    final = bracket.loc[final_id]
    champion = final['Winner']
    runner_up = final['Away'] if champion == final['Home'] else final['Home']

    bottom = _draw_results_panel(ax, centre_x, y, final_id, playoff,
                                 champion, runner_up, title_odds, flags)

    _add_title(ax, title, subtitle, pad = 10, subtitle_y = 0.995, title_fontsize = 20, subtitle_fontsize = 12.5)

    ax.set_xlim(-0.8, 8 * COL_GAP + 0.8)
    ax.set_ylim(bottom, top + 3.2)
    ax.axis('off')
    fig.tight_layout()
    if save:
        _save_png(fig, 'likely_bracket')

    return fig


# ---------- Dixon-Coles Scoreline Heatmap ----------


def _draw_probability_dashboard(top, home_team, away_team, probs, x_home, x_away,
                                stage = 'knockout'):

    # The win, draw and loss headline above a heatmap, with the expected goals badges.
    if stage == 'knockout':
        cards = [(f'{home_team} wins', probs['home_advances'], x_home, home_team),
                 (f'{away_team} wins', probs['away_advances'], x_away, away_team)]
    else:
        cards = [(f'{home_team} win', probs['home_win'], x_home, home_team),
                 ('Draw', probs['draw'], None, None),
                 (f'{away_team} win', probs['away_win'], x_away, away_team)]

    for position, (label, probability, xg, team) in enumerate(cards):
        x = (position + 0.5) / len(cards)

        flag = _flag(team, zoom = 0.26)
        if flag is not None:
            top.add_artist(AnnotationBbox(flag, (x, 0.9), xycoords = top.transAxes,
                                          frameon = False, zorder = 5))

        top.text(x, 0.75, label, fontsize = 12, color = TEXT2, transform = top.transAxes,
                 va = 'top', ha = 'center')
        top.text(x, 0.55, f'{probability:.1%}', fontsize = 38, fontweight = 'bold',
                 color = TEXT, transform = top.transAxes, va = 'top', ha = 'center')

        if xg is not None:
            label = f'xG {xg:.2f} (90 min)' if stage == 'knockout' else f'xG {xg:.2f}'
            top.annotate(label, (x, -0.2), xycoords = top.transAxes, fontsize = 10.5,
                         color = WIN_FG, va = 'center', ha = 'center',
                         bbox = dict(boxstyle = 'round,pad=0.4', facecolor = WIN_BG,
                                     edgecolor = 'none'))


def _combined_et_matrix(match_sim, home_team, away_team, venue = None):

    # Score matrix at the end of a knockout tie: ninety minutes convolved with extra time,
    # which is why the knockout heatmap runs to higher scorelines than the group one.

    x_home, x_away = match_sim.expected_goals(home_team, away_team, venue)

    regular = match_sim.score_matrix(x_home, x_away)
    extra = match_sim.score_matrix(x_home / 3, x_away / 3)

    drawn = np.diag(np.diag(regular))    # Level after 90 minutes, so extra time follows.
    combined = regular - drawn

    extended = np.clip(fftconvolve(drawn, extra, mode = 'full'), 0, None)

    size = regular.shape[0]
    combined[:size, :size] += extended[:size, :size]
    combined[-1, -1] += extended.sum() - extended[:size, :size].sum()

    return combined / combined.sum(), x_home, x_away


def plot_heatmap(match_sim, home_team, away_team, venue = None, stage = 'knockout',
                 cmap = 'Blues', max_goals = 8, save = True):

    """
    Draws a scoreline heatmap for one fixture, with the outcome probabilities above it.

    Parameters
    ----------
    match_sim : MatchSimulator
        The engine to take probabilities from.
    home_team, away_team : str
        Team names, as indexed in team_stats.
    venue : str, optional
        Venue name, used for the host boost.
    stage : {'knockout', 'group'}, default 'knockout'
        A group match is scored after ninety minutes and can end in a draw. A knockout tie is
        scored at the end of the tie, so it includes extra time and has no draw column.

    Returns
    -------
    plt.Figure
        The figure, also saved to images/.
    """
    
    if stage == 'knockout':
        title = 'Dixon-Coles Predicted World Cup Knockout Matchup'
        combined, x_home, x_away = _combined_et_matrix(match_sim, home_team, away_team, venue)

        # Ties still level after extra time go to a shootout, which the engine treats as a
        # coin flip. Saying so stops two numbers summing to 100% implying it never happens.
        shootout = np.trace(combined) / combined.sum()
        subtitle = ('Scorelines are the score at the end of the tie, including extra time.\n'
                    f'Level after extra time {shootout:.1%} of the time, decided by a shootout '
                    'the model treats as 50/50.')
    else:
        title = 'Dixon-Coles Predicted World Cup Group Matchup'
        subtitle = ('Win, draw, and loss probabilities after 90 minutes\n'
                    'of a group stage game.')
        x_home, x_away = match_sim.expected_goals(home_team, away_team, venue)
        combined = match_sim.score_matrix(x_home, x_away)

    # Anything past the grid is folded into the top corner so the cells still sum to one.
    size = max_goals + 1
    overflow = combined.sum() - combined[:size, :size].sum()
    combined = combined[:size, :size].copy()
    combined[-1, -1] += overflow
    combined /= combined.sum()

    probs = match_sim.match_probabilities(home_team, away_team, venue, stage)
    goals = np.arange(size)

    fig, (top, ax) = plt.subplots(2, 1, figsize = (9.6, 7.6), facecolor = BACKGROUND,
                                  gridspec_kw = {'height_ratios' : [1, 5], 'hspace' : 0.2})
    fig.subplots_adjust(top = 0.88, bottom = 0.08, left = 0.08, right = 0.96)

    _add_title(top, title, subtitle, subtitle_y = 1.7)
    top.set_facecolor(BACKGROUND)
    top.axis('off')

    _draw_probability_dashboard(top, home_team, away_team, probs, x_home, x_away, stage)

    sns.heatmap(combined, cmap = cmap, cbar = False, ax = ax)

    threshold = combined.max() * 0.55

    for (row, column), value in np.ndenumerate(combined):
        top_cell = value >= threshold
        ax.text(column + 0.5, row + 0.5, f'{value:.1%}', ha = 'center', va = 'center',
                fontsize = 10.5, color = 'white' if top_cell else TEXT2,
                fontweight = 'bold' if top_cell else 'normal')

    ax.invert_yaxis()
    ax.set_facecolor(BACKGROUND)
    ax.set_xlabel(f"{away_team}'s predicted goals", fontsize = 11, color = TEXT)
    ax.set_ylabel(f"{home_team}'s predicted goals", fontsize = 11, color = TEXT)

    ax.set_xticks(goals + 0.5)
    ax.set_yticks(goals + 0.5)
    ax.set_xticklabels(goals, color = TEXT)
    ax.set_yticklabels(goals, color = TEXT)
    ax.tick_params(length = 0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if save:
        _save_png(fig, f"{home_team}_{away_team}_{stage}_heatmap".replace(' ', '_'))

    return fig


# ---------- Market Comparison ----------


def _find_column(df, hints, exclude = ()):

    # First column whose name contains one of the hints, so the sources can be read without
    # hardcoding their column names.

    for hint in hints:
        for column in df.columns:
            if column not in exclude and hint in column.lower():
                return column

    raise ValueError(f'No column matching {hints} in {list(df.columns)}')


def _probability_column(df, label):

    # Pulls team and probability out of one source and normalises it to percent.

    df = df.reset_index() if df.index.name else df

    team = _find_column(df, ['team'])
    value = _find_column(df, ['win', 'probability', 'pct', 'odds'], exclude = [team])

    out = df[[team, value]].rename(columns = {team : 'team', value : label})
    out['team'] = out['team'].replace(CHANGED_TEAM_NAMES)

    # Fractions and percentages both appear in the sources, so normalise to percent.
    if out[label].max() <= 1:
        out[label] = out[label] * 100

    out[label] = out[label].round(1)

    return out


def build_comparison_data(left_df, right_df, key_col, left_label, right_label, focus_label, output_csv = None):

    """
    Joins two sets of probabilities on team and works out the gap between them.

    Raises rather than dropping rows when a team doesn't match, since a silent inner join would
    leave the chart quietly short of a bar.

    Parameters
    ----------
    left_df, right_df : pd.DataFrame
        The two sources, each with a team column and a probability column.
    key_col : str
        Column to join on, normally 'team'.
    left_label, right_label : str
        What to call each source's probability column.
    focus_label : str
        Which side the gap is measured from, and which the rows are sorted by.
    output_csv : str, optional
        Path to write the joined table to.

    Returns
    -------
    pd.DataFrame
        One row per team, both probabilities and the gap, sorted by the focus side.

    Raises
    ------
    ValueError
        If a team appears in one source but not the other.
    """

    df = pd.merge(left_df, right_df, on = key_col, how = 'outer', indicator = True)

    # An inner join would drop a renamed team silently and leave a chart quietly short of one bar.
    unmatched = df.loc[df['_merge'] != 'both', key_col].tolist()
    if unmatched:
        raise ValueError(f'unmatched teams between sources: {unmatched}')

    df = df.drop(columns = '_merge')
    other_label = left_label if focus_label == right_label else right_label
    df['gap'] = (df[focus_label] - df[other_label]).round(1)
    df = df.sort_values(focus_label, ascending = False)

    if output_csv:
        df.to_csv(output_csv, index = False)

    return df


def _draw_dumbbell(df, key_col, left_label, right_label, left_color = POLY, right_color = WIN_BG,
                   sort_by = None, top_n = None, xlabel = 'Value', title = None, subtitle = None):

    # Draws the paired dots and connecting line for each team, with the gap labelled.
    sort_by = sort_by or right_label
    plot_df = (df if top_n is None else df.head(top_n)).sort_values(sort_by)
    positions = range(len(plot_df))

    fig, ax = plt.subplots(figsize = (10, 8))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    x_min = min(plot_df[left_label].min(), plot_df[right_label].min())
    x_max = max(plot_df[left_label].max(), plot_df[right_label].max())
    offset = (x_max - x_min) * 0.02

    for y, left, right in zip(positions, plot_df[left_label], plot_df[right_label]):
        ax.plot([left, right], [y, y], color = LINE, linewidth = 1.5, zorder = 1)

        # Each value sits on the outside of its own marker, whichever way round the pair is.
        outward = -1 if left <= right else 1
        ax.text(left + outward * offset, y, f'{left:.1f}', va = 'center', fontsize = 9,
                ha = 'right' if outward < 0 else 'left', color = left_color, fontweight = 'bold')
        ax.text(right - outward * offset, y, f'{right:.1f}', va = 'center', fontsize = 9,
                ha = 'left' if outward < 0 else 'right', color = right_color, fontweight = 'bold')

    ax.scatter(plot_df[left_label], positions, color = left_color, s = 90, zorder = 2,
               label = left_label, edgecolor = OUTLINE, linewidth = 0.5)
    ax.scatter(plot_df[right_label], positions, color = right_color, s = 90, zorder = 2,
               label = right_label, edgecolor = OUTLINE, linewidth = 0.5)

    ax.set_xlim(left = x_min - offset * 6)

    if 'gap' in plot_df.columns:
        gap_x = x_max + offset * 8

        for y, gap in zip(positions, plot_df['gap']):
            ax.text(gap_x, y, f'{gap:+.1f}', va = 'center', ha = 'left', fontsize = 9, color = TEXT2)

        ax.text(gap_x, len(plot_df) - 0.4, 'Gap', va = 'bottom', ha = 'left', fontsize = 10,
                fontweight = 'bold', color = TEXT)
        ax.set_xlim(right = gap_x + offset * 8)

    ax.set_yticks(positions)
    ax.set_yticklabels(plot_df[key_col], color = TEXT)
    ax.set_xlabel(xlabel, color = TEXT)
    ax.tick_params(colors = TEXT)
    ax.grid(axis = 'x', linestyle = '--', alpha = 0.6, color = GRIDLINE)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(OUTLINE)

    _add_title(ax, title, subtitle, subtitle_y = 1.1)

    legend = fig.legend(*ax.get_legend_handles_labels(), loc = 'upper left',
                        bbox_to_anchor = (0.02, 0.98), facecolor = BOX_BG, edgecolor = OUTLINE)
    for text in legend.get_texts():
        text.set_color(TEXT)

    plt.tight_layout()
    plt.show()

    return fig


def plot_market_comp(df_a, df_b, label_a = 'Monte Carlo Model', label_b = 'Polymarket Odds', top_n = None,
                     teams = None, by_gap = False,
                     color_a = WIN_BG, color_b = POLY,
                     title = 'Market vs Monte Carlo',
                     subtitle = 'A dumbbell plot that compares the champion odds of the Monte Carlo simulation '
                     '\nto Polymarket World Cup winner odds as of 10 June 2026. Market prices sum to 102.2%'):

    """
    Champion odds against Polymarket's, as a dumbbell plot.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Two sources, each with a team column and a probability column.
    label_a, label_b : str, optional
        What to call each source.
    top_n : int, optional
        Show only this many teams.
    teams : list of str, optional
        Show only these teams.
    by_gap : bool, default False
        Sort by disagreement rather than by probability.
    color_a, color_b : str, optional
        Dot colours.
    title, subtitle : str, optional
        Chart text.

    Returns
    -------
    tuple
        The joined comparison table and the figure, also saved to images/market_comp.png.
    """
    df_a = _probability_column(df_a, label_a)
    df_b = _probability_column(df_b, label_b)

    comparison_df = build_comparison_data(df_a, df_b, 'team', label_a, label_b, focus_label = label_a,)

    selection = comparison_df

    if teams is not None:
        missing = [team for team in teams if team not in set(selection['team'])]
        if missing:
            raise ValueError(f'teams not in the comparison: {missing}')
        selection = selection[selection['team'].isin(teams)]

    if by_gap:
        selection = selection.reindex(selection['gap'].abs().sort_values(ascending = False).index)

    fig = _draw_dumbbell(selection, 'team', left_label = label_a, right_label = label_b,
                         left_color = color_a, right_color = color_b, sort_by = label_a,
                         top_n = top_n, xlabel = 'Win probability (%)',
                         title = title, subtitle = subtitle)

    _save_png(fig, 'market_comp')

    return comparison_df, fig


# ---------- Scoreline Calibration Chart ----------


def _observed_scorelines(start = '1998-06-01', end = '2022-12-31'):

    # How often each scoreline came up at past World Cups, with 2-1 and 1-2 counted together.

    results = load_csv('results.csv')
    results = results[(results['tournament'] == 'FIFA World Cup') &
                      (results['date'] >= start) & (results['date'] <= end)]

    folded = [f'{max(home, away)}-{min(home, away)}'
              for home, away in zip(results['home_score'].astype(int),
                                    results['away_score'].astype(int))]

    return pd.Series(folded).value_counts(normalize = True)


def plot_scorelines(distribution, max_total = 6,
                    title = 'Simulated Scoreline Distribution',
                    subtitle = 'Simulated 2026 scorelines against every World Cup match since 1998, with results such as '
                    '\n2-1 and 1-2 counted together. The 48-team field contains more mismatches than the 32-team tournaments shown.'):

    """
    Plots simulated scoreline frequencies against past World Cups, labelled with the gap.

    Parameters
    ----------
    distribution : pd.DataFrame
        From scoreline_table, with a scoreline and a probability per row.
    max_total : int, default 6
        Scorelines above this many goals are collected into one bucket.
    title, subtitle : str, optional
        Chart text.

    Returns
    -------
    plt.Figure
        The figure, also saved to images/scoreline_distribution.png.
    """
    observed = _observed_scorelines()

    split = distribution['scoreline'].str.split('-', expand = True).astype(int)
    table = (distribution.assign(goals = split.sum(axis = 1), margin = split[0] - split[1])
             .sort_values(['goals', 'margin']))

    shown = table[table['goals'] <= max_total]
    labels = shown['scoreline'].tolist()
    model = shown['probability'].to_numpy() * 100
    actual = np.array([observed.get(label, 0.0) for label in labels]) * 100

    # Everything above the cut is one bar rather than dropped, so both series still sum to 100%
    # and the reader can see how much of the distribution sits in the tail.
    labels.append(f'{max_total + 1}+')
    model = np.append(model, max(table['probability'].sum() * 100 - model.sum(), 0.0))
    actual = np.append(actual, max(observed.sum() * 100 - actual.sum(), 0.0))

    positions = np.arange(len(labels))

    y_max = max(model.max(), actual.max())

    fig, ax = plt.subplots(figsize = (11, 5.5), facecolor = BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    ax.bar(positions - 0.2, model, width = 0.4, color = WIN_BG, label = 'Monte Carlo Model',
           edgecolor = OUTLINE, linewidth = 0.5)
    ax.bar(positions + 0.2, actual, width = 0.4, color = POLY, label = 'Observed Scorelines',
           edgecolor = OUTLINE, linewidth = 0.5)

    for x, predicted_value, actual_value in zip(positions, model, actual):
        ax.text(x, max(predicted_value, actual_value) + 0.3,
                f'{predicted_value - actual_value:+.1f}', ha = 'center', va = 'bottom',
                fontsize = 8.5, color = TEXT2)

    y_max = max(model.max(), actual.max())
    ax.set_ylim(top = y_max * 1.1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, color = TEXT)
    ax.set_ylabel('Share of matches (%)', color = TEXT)
    ax.tick_params(colors = TEXT)
    ax.grid(axis = 'y', linestyle = '--', alpha = 0.6, color = GRIDLINE)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(OUTLINE)

    _add_title(ax, title, subtitle, subtitle_y = 1.16)

    legend = ax.legend(facecolor = BOX_BG, edgecolor = OUTLINE)
    for text in legend.get_texts():
        text.set_color(TEXT)

    plt.tight_layout()
    _save_png(fig, 'scoreline_distribution')
    plt.show()

    return fig