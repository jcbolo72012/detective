"""NASA POWER monthly point fetcher (no login). Test:  python -m planets.sources.power"""
import datetime as dt

import numpy as np
import pandas as pd
import requests

URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
LAST_FULL_YEAR = dt.date.today().year - 1  # POWER monthly serves completed years only


def parse_power_json(js, param):
    """properties.parameter.<PARAM> = {"198101": v, ..., "198113": annual mean}. Fill value -999."""
    raw = js["properties"]["parameter"][param]
    rows = {}
    for k, v in raw.items():
        if len(k) != 6 or not k.isdigit() or not 1 <= int(k[4:]) <= 12:
            continue  # skips the "YYYY13" annual entry
        rows[pd.Timestamp(int(k[:4]), int(k[4:]), 1)] = np.nan if v is None or v <= -900 else float(v)
    s = pd.Series(rows, name=param).sort_index().dropna()
    if s.empty:
        raise ValueError(f"POWER returned no usable {param} data for this point.")
    return s


def fetch_point(param, lat, lon, start=1981, end=None):
    end = min(end or LAST_FULL_YEAR, LAST_FULL_YEAR)
    start = max(start, 1984 if param.startswith("ALLSKY") else 1981)
    r = requests.get(URL, timeout=90, params=dict(
        parameters=param, community="AG", latitude=round(lat, 3), longitude=round(lon, 3),
        start=start, end=end, format="JSON"))
    if r.status_code != 200:
        raise RuntimeError(f"POWER API error {r.status_code}: {r.text[:300]}")
    return parse_power_json(r.json(), param)


def neighbor_points(lat, lon, d=1.0):
    """4 surrounding points ~100 km away (POWER met grid is 0.5 x 0.625 deg, so these are distinct cells)."""
    pts = [(lat + d, lon), (lat - d, lon), (lat, lon + d), (lat, lon - d)]
    return [(max(-89.5, min(89.5, a)), ((b + 180) % 360) - 180) for a, b in pts]


if __name__ == "__main__":
    s = fetch_point("T2M", 33.45, -112.07, 1981)
    print(s.head(3), s.tail(3), f"{len(s)} months OK", sep="\n")
