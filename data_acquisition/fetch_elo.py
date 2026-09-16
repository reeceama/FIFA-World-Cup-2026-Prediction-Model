"""
Builds data/wf_elo_history.csv and data/wf_current_elo.csv from World Football Elo.

The source gives the rating after each match along with the change, so the pre-match rating is
recovered by subtracting one from the other. That matters because using a team's current rating
to model a match it played three years ago would leak information backwards.

Also writes the 10 June 2026 snapshot, which is display only and never reaches the ratings.

Nothing in the model calls this at run time, so it only needs running to refresh the
committed data.

Usage: python data_acquisition/fetch_elo.py
"""

import os
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ---------- Configuration ----------


START_DATE = pd.Timestamp('2021-06-01')
CUTOFF_DATE = pd.Timestamp('2026-06-10')
SNAPSHOT_DATE = pd.Timestamp('2026-06-10')

MINIMUM_GAMES = 15
REQUEST_DELAY = 1.0
MAX_PAGES = 20
CHECKPOINT_EVERY = 25

CURRENT_URL = ('https://www.international-football.net/elo-ratings-table'
               f'?year={SNAPSHOT_DATE.year}&month={SNAPSHOT_DATE.month:02d}'
               f'&day={SNAPSHOT_DATE.day:02d}&confed=&prev-year=&prev-month=&prev-day=')

MATCH_URL = 'https://www.international-football.net/search-matches'

AVERAGE_OUTPUT = 'data/wf_avg_elo.csv'
CURRENT_OUTPUT = 'data/wf_current_elo.csv'
HISTORY_OUTPUT = 'data/wf_elo_history.csv'
CHECKPOINT_OUTPUT = 'data/temp_history.csv'

USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

DATE_PATTERN = (r'((?:January|February|March|April|May|June|July|August|'
                r'September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})')

SCORE_PATTERN = r'\d+\s*[-–]\s*\d+'


# ---------- Team Names ----------


CHANGED_TEAM_NAMES = {
    'Cape Verde' : 'Cabo Verde',
    'Ivory Coast' : "Côte d'Ivoire",
    'United States' : 'USA',
    'Czechia' : 'Czech Republic',
    'Dem. Rep. of Congo' : 'DR Congo',
    'Ireland' : 'Republic of Ireland',
    'Chinese Taipei' : 'Taiwan',
    'US Virgin Islands' : 'United States Virgin Islands',
    'São Tomé e Príncipe' : 'São Tomé and Príncipe',
    'Turks and Caicos' : 'Turks and Caicos Islands',
}

REVERSE_TEAM_NAMES = {value : key for key, value in CHANGED_TEAM_NAMES.items()}

# Names the site spells differently to the results file, searched in turn until one hits.
TEAM_NAME_VARIATIONS = {
    'USA' : ['United States', 'USA'],
    'Republic of Ireland' : ['Ireland', 'Republic of Ireland'],
    'DR Congo' : ['Dem. Rep. of Congo', 'DR Congo'],
    'Bosnia and Herzegovina' : ['Bosnia and Herzegovina', 'Bosnia & Herzegovina'],
    'Cabo Verde' : ['Cape Verde', 'Cabo Verde'],
    'Trinidad and Tobago' : ['Trinidad and Tobago', 'Trinidad & Tobago'],
    'São Tomé and Príncipe' : ['São Tomé e Príncipe', 'Sao Tome and Principe'],
    'Saint Vincent and the Grenadines' : ['Saint Vincent and the Grenadines',
                                          'St Vincent and the Grenadines',
                                          'St. Vincent and the Grenadines',
                                          'Saint Vincent & the Grenadines',
                                          'St Vincent & the Grenadines',
                                          'St Vincent', 'Saint Vincent'],
    'Saint Kitts and Nevis' : ['Saint Kitts and Nevis', 'St Kitts and Nevis',
                               'St. Kitts and Nevis', 'Saint Kitts & Nevis',
                               'St Kitts & Nevis', 'St Kitts', 'Saint Kitts'],
    'Saint Lucia' : ['Saint Lucia', 'St Lucia', 'St. Lucia'],
    'Saint Barthelemy' : ['Saint Barthelemy', 'St Barthelemy', 'St. Barthelemy',
                          'Saint-Barthélemy'],
    'Saint Pierre and Miquelon' : ['Saint Pierre and Miquelon', 'St Pierre and Miquelon',
                                   'St. Pierre and Miquelon', 'Saint-Pierre and Miquelon',
                                   'St Pierre', 'Saint Pierre'],
    'Saint Martin' : ['Saint Martin', 'St Martin', 'St. Martin'],
    'Sint Maarten' : ['Sint Maarten', 'St Maarten', 'St. Maarten'],
    'Antigua and Barbuda' : ['Antigua and Barbuda', 'Antigua & Barbuda', 'Antigua'],
    'Taiwan' : ['Chinese Taipei', 'Taiwan'],
    'Mayotte' : ['Mayotte', 'Mahoré'],
    'Czech Republic' : ['Czechia', 'Czech Republic'],
    "Côte d'Ivoire" : ['Ivory Coast', "Côte d'Ivoire"],
    'Curaçao' : ['Curacao', 'Curaçao'],
    'Guinea-Bissau' : ['Guinea Bissau', 'Guinea-Bissau'],
    'Timor-Leste' : ['East Timor', 'Timor-Leste'],
    'Eswatini' : ['Swaziland', 'Eswatini'],
    'North Macedonia' : ['Macedonia', 'North Macedonia'],
    'South Korea' : ['South Korea', 'Korea Republic'],
    'North Korea' : ['North Korea', 'Korea DPR'],
    'Wallis and Futuna' : ['Wallis and Futuna', 'Wallis'],
    'Turks and Caicos Islands' : ['Turks and Caicos', 'Turks and Caicos Islands', 'Turks'],
    'Marshall Islands' : ['Marshall Islands', 'Marshall'],
}

# The match tables truncate long names, so these tails are dropped before comparing.
SUFFIX_PATTERNS = [r'\s+and\s+the\s+grenadines$', r'\s+and\s+nevis$', r'\s+and\s+tobago$',
                   r'\s+and\s+barbuda$', r'\s+and\s+miquelon$', r'\s+and\s+futuna$',
                   r'\s+islands$']


def clean_team_name(name, apply_mapping = True):

    if pd.isna(name) or name is None:
        return ''

    name = re.sub(r'Image:', '', str(name).strip(), flags = re.IGNORECASE)
    name = name.replace('\xa0', ' ')
    name = re.sub(r'\s*&\s*', ' and ', name)
    name = re.sub(r'^\d+\.\s*', '', name)
    name = re.sub(r'[^\w\s\-\.\']', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return CHANGED_TEAM_NAMES.get(name, name) if apply_mapping else name


def normalise_saint(name):

    return re.sub(r'\bst\.?\b', 'saint', name, flags = re.IGNORECASE)


def strip_suffix(name):

    for pattern in SUFFIX_PATTERNS:
        name = re.sub(pattern, '', name, flags = re.IGNORECASE)

    return name.strip()


def teams_match(scraped, requested):

    scraped = clean_team_name(scraped).lower()
    requested = clean_team_name(requested).lower()

    if scraped == requested:
        return True

    scraped_saint, requested_saint = normalise_saint(scraped), normalise_saint(requested)

    if scraped_saint == requested_saint:
        return True
    if strip_suffix(scraped_saint) == strip_suffix(requested_saint):
        return True

    variations = {clean_team_name(name).lower()
                  for name in TEAM_NAME_VARIATIONS.get(requested, [])}
    variations |= {normalise_saint(name) for name in variations}
    variations |= {strip_suffix(name) for name in variations}

    if {scraped, scraped_saint, strip_suffix(scraped_saint)} & variations:
        return True

    # Long names only, so short ones like 'Chad' cannot swallow an unrelated team.
    return len(scraped) >= 6 and len(requested) >= 6 and (scraped in requested or requested in scraped)


def search_variations(team):

    names = [team] + TEAM_NAME_VARIATIONS.get(team, [])

    names += [main for main, group in TEAM_NAME_VARIATIONS.items() if team in group]

    if team in REVERSE_TEAM_NAMES:
        names.append(REVERSE_TEAM_NAMES[team])

    names.append(team.encode('ascii', 'ignore').decode())
    names.append(''.join(character for character in unicodedata.normalize('NFKD', team)
                         if not unicodedata.combining(character)))

    seen, unique = set(), []
    for name in names:
        if name and name.strip() and name.lower() not in seen:
            seen.add(name.lower())
            unique.append(name)

    return unique


# ---------- Fetching ----------


def fetch_page(url, params = None, attempts = 3):

    for attempt in range(attempts):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(url, params = params, timeout = 30,
                                    headers = {'User-Agent' : USER_AGENT})
            if response.status_code == 500:
                return None
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException:
            if attempt == attempts - 1:
                return None
            time.sleep(3)

    return None


def find_match_table(tables):

    for table in tables:
        if 'matchs' in (table.get('class') or []):
            return table

    # Fall back to whichever table carries the most scorelines.
    best, best_rows = None, 0
    for table in tables:
        rows = sum(any(re.search(SCORE_PATTERN, cell.get_text(strip = True))
                       for cell in row.find_all(['td', 'th']))
                   for row in table.find_all('tr'))
        if rows > best_rows:
            best, best_rows = table, rows

    return best


def parse_elo(cell):

    cell = cell.strip()

    signed = re.match(r'(\d{3,4})([+-]\d{1,3})', cell)
    if signed:
        return int(signed.group(1)), int(signed.group(2))

    # An unsigned zero change is glued straight onto the rating.
    if len(cell) == 5 and cell.endswith('0') and cell.isdigit():
        return int(cell[:4]), 0
    if len(cell) in (3, 4) and cell.isdigit():
        return int(cell), 0

    found = re.search(r'(\d{3,4})', cell)
    if found and 500 <= int(found.group(1)) <= 3000:
        return int(found.group(1)), 0

    return None, None


def parse_match_row(cells, team):

    if len(cells) < 6:
        return None

    elo_after, elo_change = parse_elo(cells[1])
    if elo_after is None or not 500 <= elo_after <= 3000:
        return None

    found = re.search(DATE_PATTERN, cells[2], flags = re.IGNORECASE)
    if not found:
        return None

    date = pd.to_datetime(re.sub(r'(st|nd|rd|th)', '', found.group(1), flags = re.IGNORECASE),
                          errors = 'coerce')
    if pd.isna(date) or not START_DATE <= date <= CUTOFF_DATE:
        return None

    home, away = clean_team_name(cells[3]), clean_team_name(cells[5])
    if not home or not away:
        return None

    requested = clean_team_name(team)

    if teams_match(cells[3], requested):
        opponent = away
    elif teams_match(cells[5], requested):
        opponent = home
    else:
        return None

    return {'date' : date, 'team' : requested, 'opponent' : opponent,
            'elo_after' : elo_after, 'elo_change' : elo_change, 'score' : cells[4].strip()}


def fetch_team_history(team):

    for variation in search_variations(team):
        matches, page = [], 1

        while page <= MAX_PAGES:
            soup = fetch_page(MATCH_URL, attempts = 2,
                              params = {'team' : variation, 'page' : page,
                                        'datemin' : START_DATE.strftime('%Y-%m-%d'),
                                        'datemax' : CUTOFF_DATE.strftime('%Y-%m-%d')})
            if soup is None:
                break

            table = find_match_table(soup.find_all('table'))
            if table is None:
                break

            for row in table.find_all('tr')[1:]:
                cells = [cell.get_text(strip = True) for cell in row.find_all(['td', 'th'])]
                if len(cells) > 4 and re.search(SCORE_PATTERN, cells[4]):
                    parsed = parse_match_row(cells, team)
                    if parsed:
                        matches.append(parsed)

            total = re.search(r'Page\s+\d+\s*/\s*(\d+)', soup.get_text())
            if total:
                if page >= int(total.group(1)):
                    break
            elif not re.search(r'Next', soup.get_text(), re.IGNORECASE):
                break

            page += 1

        if matches:
            return pd.DataFrame(matches)

    return pd.DataFrame()


def fetch_current_elo():

    soup = fetch_page(CURRENT_URL)
    if soup is None:
        raise ValueError(f'Could not fetch the current Elo table. Check this URL: {CURRENT_URL}')

    rows = []
    for table in soup.find_all('table'):
        for row in table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue

            text = ' '.join(cell.get_text(strip = True) for cell in cells)
            found = re.match(r'(\d+\.\s*)?(.+?)\s+(\d{3,4})$', text)
            if found and 500 <= int(found.group(3)) <= 3000:
                rows.append({'team' : found.group(2).strip(),
                             'current_elo' : int(found.group(3))})

    if not rows:
        raise ValueError('Could not parse the current Elo table. The page format may have changed.')

    current_elo = pd.DataFrame(rows)
    current_elo['team'] = current_elo['team'].apply(clean_team_name)

    return current_elo.drop_duplicates(subset = 'team', keep = 'first')


# ---------- Output ----------


def build_elo_files():

    os.makedirs('data', exist_ok = True)

    current_elo = fetch_current_elo()
    teams = current_elo['team'].tolist()
    print(f'{START_DATE.date()} to {CUTOFF_DATE.date()}, {len(teams)} teams, '
          f'minimum {MINIMUM_GAMES} games')

    history = []
    done = set()

    if os.path.exists(CHECKPOINT_OUTPUT):
        checkpoint = pd.read_csv(CHECKPOINT_OUTPUT)
        history.append(checkpoint)
        done = set(checkpoint['team'])
        print(f'Resuming from {len(checkpoint):,} matches already fetched')

    for position, team in enumerate(teams, start = 1):
        if team in done:
            continue

        team_history = fetch_team_history(team)
        if not team_history.empty:
            history.append(team_history)

        print(f'[{position}/{len(teams)}] {team}: {len(team_history)} matches')

        if position % CHECKPOINT_EVERY == 0 and history:
            pd.concat(history, ignore_index = True).to_csv(CHECKPOINT_OUTPUT, index = False)

    if not history:
        raise ValueError('No match history was fetched.')

    history = pd.concat(history, ignore_index = True)
    history = history.drop_duplicates(subset = ['date', 'team', 'opponent', 'elo_after'])
    history = history.sort_values(['team', 'date'])
    history.to_csv(HISTORY_OUTPUT, index = False)

    # The games threshold is applied to the average table, so the current table follows it.
    average_elo = (history.groupby('team')
                   .agg(games = ('elo_after', 'count'), avg_elo = ('elo_after', 'mean'))
                   .reset_index())
    average_elo = average_elo[average_elo['games'] >= MINIMUM_GAMES].copy()
    average_elo['avg_elo'] = average_elo['avg_elo'].round(2)
    average_elo = average_elo.sort_values('avg_elo', ascending = False)

    current_elo = current_elo[current_elo['team'].isin(average_elo['team'])]
    current_elo = current_elo.sort_values('current_elo', ascending = False)

    average_elo.to_csv(AVERAGE_OUTPUT, index = False)
    current_elo.to_csv(CURRENT_OUTPUT, index = False)

    if os.path.exists(CHECKPOINT_OUTPUT):
        os.remove(CHECKPOINT_OUTPUT)

    print()
    print(f'  Wrote {len(history):,} matches to {HISTORY_OUTPUT}')
    print(f'  Wrote {len(average_elo):,} teams to {AVERAGE_OUTPUT}')
    print(f'  Wrote {len(current_elo):,} teams to {CURRENT_OUTPUT}')


if __name__ == '__main__':
    build_elo_files()