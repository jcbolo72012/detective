"""Live point time series from NASA's "Giovanni in the Cloud: Time Series" API, using the USER'S OWN Earthdata token.

    GET https://api.giovanni.earthdata.nasa.gov/timeseries?data=<variable id>&location=[lat,lon]&time=<start>/<end>
    Authorization: Bearer <Earthdata Login token>      -> CSV

Facts confirmed from NASA forum/tutorial pages: endpoint, parameter names, `location` as "[lat,lon]", the Bearer
header, CSV output, point-only (no area averages), only some collections supported.
NOT confirmed (docs are JavaScript-only): exact `time` format and exact CSV layout. So this module tries three
time formats and parses the CSV tolerantly, and on any failure keeps the raw response head for the UI to show.

The token is never cached, logged or put in an error message. It lives in st.session_state only.
"""
import csv
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ENDPOINT = "https://api.giovanni.earthdata.nasa.gov/timeseries"
QUOTA_URL = "https://api.giovanni.earthdata.nasa.gov/quota"
TOKEN_URL = "https://urs.earthdata.nasa.gov/profile"
TIME_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d")

# id -> (name, units, definition, verified)
PRESETS = {
    "GLDAS_NOAH025_3H_2_1_Tair_f_inst": (
        "Air temperature (GLDAS land model, 3-hourly)", "K",
        "Near-surface air temperature from the GLDAS Noah land surface model, every 3 hours. Kelvin: a change of "
        "1 K equals a change of 1 °C.", True),
    "M2TMNXSLV_5_12_4_T2M": (
        "Air temperature at 2 m (MERRA-2 monthly)", "K",
        "Temperature 2 metres above the surface from the MERRA-2 reanalysis, monthly means.", False),
    "M2TMNXSLV_5_12_4_TO3": (
        "Total column ozone (MERRA-2 monthly)", "DU",
        "Total ozone in a column of air, in Dobson units, from the MERRA-2 reanalysis.", False),
    "GPM_3IMERGM_07_precipitation": (
        "Precipitation (GPM IMERG monthly)", "mm/hr",
        "Satellite-based rainfall estimate from NASA's GPM mission, monthly mean rate.", False),
}
CUSTOM_ID = "Other variable ID…"


class GiovanniError(Exception):
    """kind: auth | request | empty | network | parse"""
    def __init__(self, message, kind="request", raw=""):
        super().__init__(message)
        self.kind, self.raw = kind, raw


def clean_token(raw):
    t = (raw or "").strip().strip('"').strip("'")
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    return t


def _redact(text, token):
    return text.replace(token, "[token]") if token else text


def parse_timeseries_csv(text):
    """Tolerant parser. Skips metadata lines; the first row whose first cell is a date and second cell a number
    starts the data. Returns (raw_series, meta). Raises GiovanniError('parse'|'empty') with the raw head."""
    head = text[:600]
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    meta = {"fill": None, "units": None, "header": None, "title": None}
    first = None
    for i, r in enumerate(rows):
        if len(r) < 2:
            continue
        cell = r[0].strip()
        low = cell.lower().rstrip(":")
        if low.startswith("fill") and len(r) > 1:
            try:
                meta["fill"] = float(r[-1])
            except ValueError:
                pass
        if low.startswith("unit") and len(r) > 1:
            meta["units"] = r[-1].strip() or None
        if low in ("title", "name", "param_name") and len(r) > 1:
            meta["title"] = r[1].strip()
        try:
            float(r[1])
            ok_num = True
        except ValueError:
            ok_num = False
        if ok_num and cell and not pd.isna(pd.to_datetime(cell, errors="coerce", utc=True)):
            first = i
            break
    if first is None:
        raise GiovanniError("NASA answered, but I couldn't find a date,value table in the response.", "parse", head)
    if first > 0 and len(rows[first - 1]) >= 2:
        meta["header"] = [c.strip() for c in rows[first - 1]]
    times, vals = [], []
    for r in rows[first:]:
        if len(r) < 2:
            continue
        try:
            v = float(r[1])
        except ValueError:
            v = np.nan
        times.append(r[0].strip())
        vals.append(v)
    t = pd.to_datetime(pd.Series(times), errors="coerce", utc=True).dt.tz_localize(None)
    s = pd.Series(vals, index=t).dropna(how="all")
    s = s[~s.index.isna()]
    if meta["fill"] is not None:
        s[np.isclose(s.values, meta["fill"])] = np.nan
    s[np.abs(s.values) > 1e30] = np.nan
    s[s.values == -9999] = np.nan
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated()]
    if s.empty:
        raise GiovanniError("NASA returned a table with no valid values at this point and time range.", "empty", head)
    return s, meta


def to_monthly(s):
    """Daily / sub-daily / weekly -> monthly means (drops months that are less than 80% covered).
    Already-monthly -> month start. Returns (series, note)."""
    if len(s) < 3:
        raise GiovanniError("Too few values came back to build a monthly series.", "empty")
    step = float(np.median(np.diff(s.index.values).astype("timedelta64[s]").astype(float))) / 86400.0
    if 25 <= step <= 35:
        m = s.copy()
        m.index = m.index.to_period("M").to_timestamp()
        m = m.groupby(level=0).mean()
        return m, "Monthly values used as returned."
    if step > 35:
        raise GiovanniError("This variable is reported less often than monthly; the trend tools need monthly data.",
                            "request")
    per_day = round(1.0 / step) if step < 1 else 1.0 / step
    g = s.groupby(s.index.to_period("M"))
    mean, count = g.mean(), g.count()
    expected = np.array([p.days_in_month for p in mean.index]) * per_day
    keep = count.values >= 0.8 * expected
    m = mean[keep]
    m.index = m.index.to_timestamp()
    kind = "hourly" if step < 0.1 else "sub-daily" if step < 0.9 else "daily" if step < 2 else "weekly"
    return m, f"{kind} values averaged into monthly means ({int((~keep).sum())} partial months dropped)."


# Years per request by native time step, so each request stays near NASA's tutorial size (~60k values).
CHUNK_YEARS = {"half-hourly": 3, "hourly": 6, "3-hourly": 20}
CHUNK_WORKERS = 3  # pieces of one record downloaded at the same time


def _chunks(y0, y1, step):
    n = CHUNK_YEARS.get(step)
    if not n:
        return [(y0, y1)]
    return [(a, min(a + n - 1, y1)) for a in range(y0, y1 + 1, n)]


def _request(http, params, headers, timeout, token):
    try:
        return http.get(ENDPOINT, params=params, headers=headers, timeout=timeout)
    except requests.Timeout:
        raise GiovanniError("NASA took too long to answer. Try fewer years, or try again.", "network")
    except requests.RequestException as e:
        raise GiovanniError(_redact(f"Couldn't reach NASA Giovanni ({type(e).__name__}).", token), "network")


def _fetch_span(http, data_id, lat, lon, y0, y1, headers, timeout, token, formats, depth=0):
    """One span -> (csv text, time format). Tries time formats only on 4xx; splits the span in half on 5xx."""
    t0, t1 = pd.Timestamp(f"{y0}-01-01 00:00:00"), pd.Timestamp(f"{y1}-12-31 23:59:59")
    last = None
    for fmt in formats:
        params = {"data": data_id, "location": f"[{round(float(lat), 4)},{round(float(lon), 4)}]",
                  "time": f"{t0.strftime(fmt)}/{t1.strftime(fmt)}"}
        r = _request(http, params, headers, timeout, token)
        body = r.text or ""
        if r.status_code in (401, 403):
            raise GiovanniError(
                "NASA rejected the token (expired, mistyped, or this Earthdata account hasn't authorized the "
                "'NASA GESDISC DATA ARCHIVE' application). Tokens expire; generate a fresh one.",
                "auth", _redact(body[:600], token))
        if r.status_code == 200 and body.strip() and not body.lstrip().startswith("{"):
            return body, fmt
        if r.status_code >= 500:
            if y1 > y0 and depth < 3:  # server fell over: usually too much data in one go, so halve it
                mid = (y0 + y1) // 2
                a, fa = _fetch_span(http, data_id, lat, lon, y0, mid, headers, timeout, token, [fmt], depth + 1)
                b, _ = _fetch_span(http, data_id, lat, lon, mid + 1, y1, headers, timeout, token, [fa], depth + 1)
                return a + "\n" + b, fa
            raise GiovanniError(f"NASA's server failed on this request (HTTP {r.status_code}), even for a single "
                                "year. The service may be down; try again later or pick a monthly variable.",
                                "request", _redact(body[:600], token))
        last = (r.status_code, _redact(body[:600], token))  # 4xx / empty / JSON error: try next time format
    code, head = last
    hint = (" The variable ID may be wrong, or not supported by the time series service." if code in (400, 404, 422)
            else " The response was empty: this point may have no data (for example ocean-only or regional "
                 "datasets), or the years are outside the record.")
    raise GiovanniError(f"NASA didn't return data (HTTP {code}).{hint}", "request", head)


def fetch(data_id, lat, lon, start_year, end_year, token, session=None, timeout=90, step=None):
    """Returns dict(series=monthly Series, raw=native Series, meta, note, time_format, raw_head, requests).
    Long sub-daily records are fetched in several smaller requests. Raises GiovanniError."""
    token = clean_token(token)
    if not token:
        raise GiovanniError("Paste your Earthdata token first.", "auth")
    if not data_id or not data_id.strip():
        raise GiovanniError("Enter a Giovanni variable ID.", "request")
    http = session or requests
    headers = {"Authorization": f"Bearer {token}"}
    spans = _chunks(int(start_year), int(end_year), step)
    data_id = data_id.strip()
    # first piece alone: it settles which time format NASA accepts; the rest then go CHUNK_WORKERS at a time
    first_body, fmt = _fetch_span(http, data_id, lat, lon, *spans[0], headers, timeout, token, list(TIME_FORMATS))
    bodies = [first_body]
    if len(spans) > 1:
        def one(span):
            return _fetch_span(http, data_id, lat, lon, *span, headers, timeout, token, [fmt])[0]
        with ThreadPoolExecutor(max_workers=min(CHUNK_WORKERS, len(spans) - 1)) as ex:
            bodies += list(ex.map(one, spans[1:]))  # map keeps order; the first error is re-raised here
    n_req = len(spans)
    head = _redact(first_body[:600], token)
    parts, meta = [], None
    for body in bodies:
        try:  # a split (5xx) answer is two CSVs glued together; the parser skips the second header
            raw_part, m = parse_timeseries_csv(body)
        except GiovanniError as e:
            if e.kind == "empty":  # an empty piece is fine (e.g. the record starts later)
                continue
            e.raw = _redact(e.raw, token)
            raise
        parts.append(raw_part)
        meta = meta or m
    if not parts:
        raise GiovanniError("NASA returned no valid values at this point and time range (it may be outside the "
                            "dataset's region, e.g. ocean-only or North America-only data).", "empty", head)
    raw = pd.concat(parts).sort_index()
    raw = raw[~raw.index.duplicated()]
    monthly, note = to_monthly(raw)
    units = meta.get("units") or ""
    if units.strip().lower() in ("k", "kelvin", "degk", "deg k"):
        monthly, raw = monthly - 273.15, raw - 273.15
        note += " Converted from kelvin to °C."
        units = "°C"
    meta["display_units"] = units
    if n_req > 1:
        note += f" Fetched in {n_req} requests to keep each one small."
    return dict(series=monthly, raw=raw, meta=meta, note=note, time_format=fmt, raw_head=head, requests=n_req)


def check_token(token, session=None, timeout=20):
    """Validate a token against Giovanni's quota endpoint. Returns fraction of this month's quota used (0-1).
    Raises GiovanniError('auth'|'network'|'request')."""
    token = clean_token(token)
    if not token:
        raise GiovanniError("Paste your Earthdata token first.", "auth")
    http = session or requests
    try:
        r = http.get(QUOTA_URL, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    except requests.RequestException as e:
        raise GiovanniError(f"Couldn't reach NASA Giovanni ({type(e).__name__}).", "network")
    if r.status_code in (401, 403):
        raise GiovanniError("NASA rejected this token (expired or mistyped). Generate a fresh one.", "auth")
    if r.status_code != 200:
        raise GiovanniError(f"NASA's quota check answered HTTP {r.status_code}.", "request", _redact((r.text or "")[:300], token))
    try:
        return float(r.json().get("quota"))
    except (ValueError, TypeError, AttributeError):
        raise GiovanniError("NASA's quota check answered in an unexpected format.", "request",
                            _redact((r.text or "")[:300], token))
