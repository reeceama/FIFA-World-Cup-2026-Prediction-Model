"""
Builds data/third_place_table.json, FIFA's Annex C mapping of qualifying
third-place groups to Round of 32 slots.

Nothing in the model calls this at run time, so it only needs running to refresh the
committed data.

Usage: python data_acquisition/third_place_table.py
"""

import io
import json
import pandas as pd
import requests

URL = 'https://en.wikipedia.org/wiki/Template:2026_FIFA_World_Cup_third-place_table'
HEADERS = {'User-Agent': 'WorldCupPredictionModel/1.1 (personal project)'}

# The eight group winners who face a third placed team
SLOTS = ['1A', '1B', '1D', '1E', '1G', '1I', '1K', '1L']


def build_third_place_table():

    """
    Maps each combination of eight qualifying groups to the third placed team
    facing each group winner.
    """

    response = requests.get(URL, headers = HEADERS, timeout = 20)
    response.raise_for_status()

    raw = pd.read_html(io.StringIO(response.text))[0]

    third_place_table = {}
    for _, row in raw.iterrows():

        groups = ''.join(sorted(g for g in row.iloc[1:13] if isinstance(g, str)))
        third_place_table[groups] = {slot: row.iloc[14 + i] for i, slot in enumerate(SLOTS)}

    with open('data/third_place_table.json', 'w') as f:
        json.dump(third_place_table, f, indent = 2)

    print(f'Wrote {len(third_place_table)} combinations')


if __name__ == '__main__':
    build_third_place_table()


