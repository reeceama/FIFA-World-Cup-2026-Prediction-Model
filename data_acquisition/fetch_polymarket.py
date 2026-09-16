"""
Builds data/wc_2026_polymarket_odds.csv, a snapshot of market title odds on 10 June 2026.

Prices are recorded on 10 June rather than read live, since prices move and a live fetch
would mean the comparison changed every time anyone ran the notebook and no longer matched the
freeze date. The API needs a browser user agent or it returns 403.

Nothing in the model calls this at run time, so it only needs running to refresh the
committed data.

Usage: python data_acquisition/fetch_polymarket.py
"""

import json
import re
import time

import pandas as pd
import requests

# ---------- Configuration ----------

SNAPSHOT_DATE = pd.Timestamp('2026-06-10')
REFERENCE_TS = int(SNAPSHOT_DATE.timestamp())

LOOKBACK_SECONDS = 2 * 86400     # two days before the snapshot
LOOKAHEAD_SECONDS = 1 * 86400    # one day after
FIDELITY = 60                    # minutes per candle
REQUEST_DELAY = 0.5

EVENT_SLUG = 'world-cup-winner'
EVENT_URL = f'https://gamma-api.polymarket.com/events/slug/{EVENT_SLUG}'
PRICE_HISTORY_URL = 'https://clob.polymarket.com/prices-history'

USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

OUTPUT_FILE = 'data/wc_2026_polymarket_odds.csv'

# Polymarket carries placeholder slots for teams that had not qualified yet.
PLACEHOLDER_PATTERN = re.compile(r'^Team\s+[A-Z]{1,3}$', re.IGNORECASE)

# ---------- Fetching ----------

def fetch_json(url, params = None, attempts = 3):

    for attempt in range(attempts):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(url, params = params, timeout = 30,
                                    headers = {'User-Agent' : USER_AGENT})
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == attempts - 1:
                return None
            time.sleep(2)


def fetch_team_markets():

    event = fetch_json(EVENT_URL)
    if event is None:
        raise ValueError(f'Could not fetch the event. Check this URL: {EVENT_URL}')

    markets = []

    for market in event.get('markets', []):
        team = market.get('groupItemTitle')

        if not team:
            question = market.get('question', '')
            found = re.search(r'Will\s+(.+?)\s+win', question, flags = re.IGNORECASE)
            team = found.group(1) if found else question

        team = team.strip()

        if PLACEHOLDER_PATTERN.match(team) or team.lower() == 'other':
            continue

        try:
            outcomes = json.loads(market.get('outcomes', '[]'))
            token_ids = json.loads(market.get('clobTokenIds', '[]'))
        except (json.JSONDecodeError, TypeError):
            continue

        if not outcomes or len(outcomes) != len(token_ids):
            continue

        yes = next((i for i, outcome in enumerate(outcomes)
                    if str(outcome).strip().lower() == 'yes'), 0)

        markets.append({'team' : team, 'token_id' : token_ids[yes]})

    return markets


def fetch_price_at(token_id, reference_ts):

    response = fetch_json(PRICE_HISTORY_URL,
                          params = {'market' : token_id, 'fidelity' : FIDELITY,
                                    'startTs' : reference_ts - LOOKBACK_SECONDS,
                                    'endTs' : reference_ts + LOOKAHEAD_SECONDS})

    history = response.get('history', []) if response else []
    if not history:
        return None, None

    # The last candle at or before the snapshot, or the nearest one if trading started later.
    before = [point for point in history if point['t'] <= reference_ts]
    point = (max(before, key = lambda point: point['t']) if before
             else min(history, key = lambda point: abs(point['t'] - reference_ts)))

    return point['p'], point['t']

# ---------- Output ----------

def build_polymarket_odds():

    markets = fetch_team_markets()
    rows = []

    for market in markets:
        price, timestamp = fetch_price_at(market['token_id'], REFERENCE_TS)
        if price is None:
            continue

        rows.append({'team' : market['team'],
                     'implied_probability_pct' : round(price * 100, 2),
                     'decimal_odds' : round(1 / price, 2) if price > 0 else None,
                     'snapshot_date' : SNAPSHOT_DATE.date(),
                     'actual_price_date' : pd.to_datetime(timestamp, unit = 's')})

    odds = pd.DataFrame(rows).sort_values('implied_probability_pct', ascending = False)
    odds.to_csv(OUTPUT_FILE, index = False)

    print(f'Snapshot {SNAPSHOT_DATE.date()}')
    print(f'  Wrote {len(odds)} teams to {OUTPUT_FILE}')

    if len(markets) > len(odds):
        print(f'  No price history for {len(markets) - len(odds)} of {len(markets)} markets')


if __name__ == '__main__':
    build_polymarket_odds()