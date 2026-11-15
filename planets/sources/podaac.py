"""NASA PO.DAAC ocean time series (sea surface temperature, sea level) for one point, with the user's Earthdata token.

Data are read straight from PO.DAAC's cloud OPeNDAP server (Hyrax): for each month one tiny request returns just the
grid cell asked for, so nothing large is downloaded. Granules are found with NASA's public CMR search.

  POD:SST  NOAA ERSST v5 monthly sea surface temperature, 2 degree grid, 1854-present (distributed by PO.DAAC)
  POD:SSH  NASA-SSH simple gridded sea surface height anomaly, 0.5 degree grid, 1993-present. The source maps come
           every 5 days; one map per month (nearest the 15th) is used, so it is a monthly sample, not a monthly mean.

Both are ocean-only: a point on land uses the nearest open-ocean cell within a few cells, and says so.
Test (needs EARTHDATA_TOKEN in .env):  python -m planets.sources.podaac"""
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import netCDF4
import numpy as np
import pandas as pd
import requests

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
OPENDAP = "https://opendap.earthdata.nasa.gov/collections"
PREFIX = "POD:"
WORKERS = 10
_nc_lock = threading.Lock()  # the netCDF C library is not thread-safe; downloads run in parallel, parsing does not

PRODUCTS = {
    "SST": dict(concept="C2036878116-POCLOUD", var="sst", lat="lat", lon="lon", ce="/sst[0][0][{i}][{j}]",
                window="/sst[0][0][{i0}:{i1}][{j0}:{j1}]", scale=1.0, radius=2,
                title_month=re.compile(r"ersst\.v5\.(\d{4})(\d{2})$")),  # file name says the month; its time span is mid-month to mid-month
    "SSH": dict(concept="C4155232533-POCLOUD", var="ssha", lat="latitude", lon="longitude", ce="/ssha[{i}][{j}]",
                window="/ssha[{i0}:{i1}][{j0}:{j1}]", scale=1000.0, radius=3),
}

ENTRIES = {
    "SST": {
        "name": "Sea surface temperature (NOAA ERSST v5, via PO.DAAC)", "short_name": "Sea surface temperature",
        "units": "°C", "source": "NASA PO.DAAC · NOAA ERSST v5",
        "definition": "Average temperature of the very top of the ocean, from NOAA's Extended Reconstructed Sea "
                      "Surface Temperature v5 (monthly, 2-degree grid cells, about 200 km) distributed by NASA PO.DAAC. "
                      "Ocean only: a place on the coast uses the nearest open-ocean cell.",
        "keywords": ["ocean", "sea", "sst", "marine"], "first_year": 1950, "default_years": 20},
    "SSH": {
        "name": "Sea level anomaly (NASA-SSH, via PO.DAAC)", "short_name": "Sea level anomaly",
        "units": "mm", "source": "NASA PO.DAAC · NASA-SSH altimetry",
        "definition": "How high the sea surface is compared with its long-term average, measured by radar altimeter "
                      "satellites (NASA-SSH, 0.5-degree grid). One map per month (nearest the 15th) is used. Ocean "
                      "only: a place on the coast uses the nearest open-ocean cell.",
        "keywords": ["ocean", "sea", "altimetry", "satellite"], "first_year": 1993, "default_years": 20},
}


class PodaacError(Exception):
    pass


def load():
    """Catalog entries in the shape retrieve.py expects, keyed 'POD:<name>'."""
    return {f"{PREFIX}{k}": {**v, "provider": "podaac", "podaac_id": k, "step": "monthly", "last_year": None,
                             "region": "oceans only", "verified": True} for k, v in ENTRIES.items()}


def clean_token(raw):
    return re.sub(r"\s+", "", (raw or "").replace("Bearer ", ""))


def _get(url, params, token, timeout=60, tries=3):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    err = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            err = f"{type(e).__name__}"
        else:
            if r.status_code in (401, 403):
                raise PodaacError("PO.DAAC rejected your Earthdata token. Generate a fresh one at "
                                  "urs.earthdata.nasa.gov/profile and paste it in the sidebar.")
            if r.status_code == 404:
                return None
            if r.ok:
                return r
            err = f"HTTP {r.status_code}"
            if r.status_code < 500:
                break
        time.sleep(1.5 * (k + 1))
    raise PodaacError(f"PO.DAAC did not answer ({err}). Try again in a moment.")


def _month_start(ts):
    return pd.Timestamp(ts.year, ts.month, 1)


def _granules(product, start_year, end_year):
    """[(month start, granule ur)], one per month. For 5-day maps: the one centred nearest the 15th."""
    best = {}
    page = 1
    while page <= 4:
        r = _get(CMR_GRANULES, dict(collection_concept_id=product["concept"], page_size=2000, page_num=page,
                                    temporal=f"{start_year}-01-01T00:00:00Z,{end_year}-12-31T23:59:59Z",
                                    sort_key="start_date"), None, timeout=60)
        entries = r.json()["feed"]["entry"] if r is not None else []
        for g in entries:
            t = product.get("title_month") and product["title_month"].search(g["title"])
            if t:  # one granule per month, named for it
                m = pd.Timestamp(int(t.group(1)), int(t.group(2)), 1)
                if start_year <= m.year <= end_year:
                    best.setdefault(m, (0, g["title"]))
                continue
            t0 = pd.Timestamp(g["time_start"]).tz_localize(None)
            t1 = pd.Timestamp(g["time_end"]).tz_localize(None) if g.get("time_end") else t0
            mid = t0 + (t1 - t0) / 2
            if not (start_year <= mid.year <= end_year):
                continue
            m = _month_start(mid)
            gap = abs((mid - (m + pd.Timedelta(days=14))).total_seconds())
            if m not in best or gap < best[m][0]:
                best[m] = (gap, g["title"])
        if len(entries) < 2000:
            break
        page += 1
    return sorted((m, ur) for m, (_, ur) in best.items())


def _read(product, ur, ce, token):
    """One OPeNDAP request -> {variable: masked array}, or None if the granule is missing."""
    r = _get(f"{OPENDAP}/{product['concept']}/granules/{ur}.dap.nc4", {"dap4.ce": ce}, token, timeout=90)
    if r is None:
        return None
    with _nc_lock:
        ds = netCDF4.Dataset("mem.nc", memory=r.content)
        try:
            return {k: np.ma.array(ds[k][:]) for k in ds.variables}
        finally:
            ds.close()


def _nearest_index(axis, value, wrap=None):
    if wrap:
        value = value % wrap
    return int(np.abs(axis - value).argmin())


def _find_cell(product, ur, lat, lon, token):
    """Index and centre of the open-ocean grid cell nearest (lat, lon), from one granule."""
    axes = _read(product, ur, f"/{product['lat']};/{product['lon']}", token)
    if axes is None:
        raise PodaacError("PO.DAAC returned no map to read the grid from.")
    la_ax, lo_ax = np.asarray(axes[product["lat"]], float), np.asarray(axes[product["lon"]], float)
    wrap = 360.0 if lo_ax.min() >= 0 else None
    i, j = _nearest_index(la_ax, lat), _nearest_index(lo_ax, lon, wrap)
    rad = product["radius"]
    i0, i1 = max(0, i - rad), min(len(la_ax) - 1, i + rad)
    j0, j1 = max(0, j - rad), min(len(lo_ax) - 1, j + rad)
    win = _read(product, ur, product["window"].format(i0=i0, i1=i1, j0=j0, j1=j1), token)
    if win is None:
        raise PodaacError("PO.DAAC returned no data for this place.")
    vals = np.ma.masked_invalid(np.ma.masked_greater(np.asarray(win[product["var"]]).squeeze(), 1e20))
    best = None
    for a in range(vals.shape[0]):
        for b in range(vals.shape[1]):
            if np.ma.is_masked(vals[a, b]):
                continue
            ci, cj = i0 + a, j0 + b
            d = math.hypot(la_ax[ci] - lat, ((lo_ax[cj] - lon + 180) % 360 - 180) * math.cos(math.radians(lat)))
            if best is None or d < best[0]:
                best = (d, ci, cj)
    if best is None:
        raise PodaacError("This is an ocean variable and there is no open ocean near this place. Try a coastal city "
                          "or an island.")
    _, ci, cj = best
    return ci, cj, float(la_ax[ci]), float(((lo_ax[cj] + 180) % 360) - 180)


def fetch(code, lat, lon, start_year, end_year, token, step=None):
    """One monthly Series for `code` ('POD:SST' or 'POD:SSH'), plus a note about the grid cell used.
    Safe in a worker thread: no Streamlit calls. `token` is never stored or logged."""
    key = str(code)[len(PREFIX):] if str(code).startswith(PREFIX) else str(code)
    if key not in PRODUCTS:
        raise PodaacError(f"Unknown PO.DAAC variable '{code}'.")
    token = clean_token(token)
    if not token:
        raise PodaacError("PO.DAAC needs your free Earthdata token (sidebar).")
    product = PRODUCTS[key]
    grans = _granules(product, int(start_year), int(end_year))
    if len(grans) < 24:
        raise PodaacError("PO.DAAC has fewer than two years of data for those years.")
    ci, cj, clat, clon = _find_cell(product, grans[0][1], lat, lon, token)
    ce = product["ce"].format(i=ci, j=cj)

    def one(item):
        month, ur = item
        got = _read(product, ur, ce, token)
        if got is None:
            return month, np.nan
        v = np.ma.masked_greater(np.asarray(got[product["var"]]).squeeze(), 1e20)
        return month, (np.nan if np.ma.is_masked(v) or not np.isfinite(float(v)) else float(v) * product["scale"])

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(one, grans))
    s = pd.Series(dict(rows), name=code).sort_index().dropna()
    if len(s) < 24:
        raise PodaacError("Too few valid ocean months at this place.")
    off = math.hypot(clat - lat, (clon - lon + 180) % 360 - 180)
    note = (f"Ocean grid cell centred at {clat:.2f}°, {clon:.2f}°" +
            (f" (about {off * 111:.0f} km from the place you chose)." if off > 0.3 else "."))
    return s, note


if __name__ == "__main__":
    from planets import config
    tok = config.env("EARTHDATA_TOKEN")
    for code, lat, lon in (("POD:SST", 21.3, -157.8), ("POD:SSH", 21.3, -157.8), ("POD:SST", 25.76, -80.19)):
        t = time.time()
        s, note = fetch(code, lat, lon, 2015, 2024, tok)
        print(code, f"{len(s)} months in {time.time() - t:.0f}s", f"{s.iloc[0]:.2f} .. {s.iloc[-1]:.2f}", note)
