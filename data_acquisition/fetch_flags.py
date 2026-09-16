"""
Downloads a PNG flag for each of the 48 finalists from flagcdn, into data/flags/.

Names are looked up through FLAG_CODES, which maps a team to its ISO code. England and
Scotland need the subdivision codes rather than a country code, which is why that mapping is a
CSV rather than derived.

Nothing in the model calls this at run time, so it only needs running to refresh the
committed data.

Usage: python data_acquisition/fetch_flags.py
"""

import os
import requests

from wcmodel.team_data import FLAG_CODES, FLAG_DIR

# ---------- Flags ----------


def fetch_flags():
    
    os.makedirs(FLAG_DIR, exist_ok = True)
    for _, code in FLAG_CODES.items():
        path = os.path.join(FLAG_DIR, f"{code}.png")

        if os.path.exists(path):
            continue

        url = f"https://flagcdn.com/w80/{code}.png"
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)


if __name__ == '__main__':
    fetch_flags()