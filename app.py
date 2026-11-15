"""PLaNETS: Plain-Language NASA Earth Trend Sleuth. The student is the detective;
the Planeteer is their sidekick that fetches NASA evidence and runs the tests.
Run:   pip install streamlit pandas numpy scipy matplotlib requests   (+ anthropic or openai, + mlflow: optional)
       streamlit run app.py
Flow:  question -> [AI or keywords] parameters -> NASA POWER / Giovanni CSV -> analyze() in code -> [AI or template] explanation
"""
import datetime as dt
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Streamlit Cloud reruns app.py when a new commit lands but keeps already-imported modules in memory, so app.py and
# planets/ can end up from different commits (e.g. "analyze() takes 2 positional arguments but 3 were given").
# Re-import the planets package whenever any of its files changed since it was loaded.
import sys
from pathlib import Path

_PLANETS_STAMP = max(f.stat().st_mtime for f in (Path(__file__).resolve().parent / "planets").rglob("*.py"))
if getattr(sys.modules.get("planets"), "_stamp", _PLANETS_STAMP) != _PLANETS_STAMP:  # loaded by an older run
    for _mod in [m for m in sys.modules if m == "planets" or m.startswith("planets.")]:
        del sys.modules[_mod]
import planets  # noqa: E402

planets._stamp = _PLANETS_STAMP

try:  # lets worker threads share this session's Streamlit context (cache, no "missing ScriptRunContext" noise)
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:  # older Streamlit or test stub
    add_script_run_ctx = get_script_run_ctx = None

from planets.analysis import compare
from planets.ui import explorer
from planets.forecast import forecast_lstm
from planets.analysis import emergence
from planets.analysis import quality
from planets.ui import share
from planets.ui import glossary
from planets.ui import look, palette
from planets.query import geo
from planets.sources import case_files
from planets.sources import giovanni_catalog
from planets.sources import giovanni_live
from planets.query import retrieve
from planets.query import nl
from planets.sources import power
from planets.sources import providers
from planets.ui import space_view
from planets.query.catalog import DEMO_QUESTIONS, PLACES, VARIABLES
from planets.query.retrieve import ALL, search
from planets.analysis.trend import analyze, load_giovanni_csv

try:
    from planets.sources import podaac
except ImportError:  # netCDF4 not installed: the PO.DAAC ocean variables are simply not offered
    podaac = None

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

st.set_page_config(page_title="PLaNETS: Plain-Language NASA Earth Trend Sleuth", page_icon="🌍", layout="wide")
look.apply()
MODE_LABELS = {"Ask a question": "🔎 Ask the Planeteer", "Compare two trends": "⚖️ Compare two suspects",
               "Pick from menus": "🧭 Pick from menus", "Case files (Giovanni CSV)": "🗂️ Case files (Giovanni CSV)",
               "Explore variables": "🔭 Explore the evidence locker"}
SEASON_LABELS = {None: "All months", "DJF": "Dec–Feb", "MAM": "Mar–May",
                 "JJA": "Jun–Aug", "SON": "Sep–Nov"}


def keep_cycle():
    """Sidebar toggle: feed the seasonal cycle into the trend tests (all-months analyses only)."""
    return bool(st.session_state.get("keep_cycle"))


def season_text(season):
    if season:
        return SEASON_LABELS[season]
    return "All months (seasonal cycle kept)" if keep_cycle() else "All months (seasonal cycle removed)"


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _power_series(param, lat, lon, start, end):
    return power.fetch_point(param, lat, lon, start, end)


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=200)
def _giovanni_series(gid, lat, lon, start, end, step, _token):
    """_token starts with an underscore so Streamlit never hashes or stores it as part of the cache key."""
    got = giovanni_live.fetch(gid, lat, lon, start, end, _token, step=step)
    return (got["series"], got["meta"].get("display_units") or "", got["note"],
            dict(doi=got["meta"].get("doi"), product=got["meta"].get("product")))


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=100)
def _podaac_series(code, lat, lon, start, end, _token):
    """One request per month of record, so results are cached for a day. _token is not part of the cache key."""
    s, note = podaac.fetch(code, lat, lon, start, end, _token)
    return s, ALL[code].get("units") or "", note


def gio_token():
    return giovanni_live.clean_token(st.session_state.get("gio_token", ""))


FETCH_WORKERS = 5  # requests in flight at once: a point + its 4 neighbours together, still polite to NASA


def run_parallel(jobs, workers=FETCH_WORKERS):
    """Run zero-argument callables at the same time. Returns [(result, exception)] in the same order;
    one failure never cancels the others. Worker threads must not draw Streamlit elements."""
    ctx = get_script_run_ctx() if get_script_run_ctx else None

    def call(fn):
        if ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        try:
            return fn(), None
        except Exception as e:  # handed back to the caller, which decides whether it matters
            return None, e

    if len(jobs) <= 1:
        return [call(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs)), thread_name_prefix="fetch") as ex:
        return list(ex.map(call, jobs))


def _fetch(param, lat, lon, start, end, tok):
    """Fetch one series from whichever source owns `param`. Safe in a worker thread: no Streamlit UI or
    session_state here. Returns (series, giovanni_meta or None)."""
    if retrieve.is_giovanni(param):
        if not tok:
            raise ValueError(f"'{ALL[param]['short_name']}' comes from NASA Giovanni, which needs your free "
                             "Earthdata token. Add it under 'NASA Giovanni' in the sidebar.")
        try:
            s, units, note, cite = _giovanni_series(param[len(giovanni_catalog.PREFIX):], lat, lon, start, end,
                                                    ALL[param].get("step") or "", tok)
        except giovanni_live.GiovanniError as e:
            msg = f"NASA Giovanni: {e}"
            if e.raw and e.kind in ("parse", "request"):
                msg += f"\n\nWhat NASA sent back: `{e.raw[:200]}`"
            raise ValueError(msg)
        return s, dict(units=units, note=note, **cite)
    if providers.provider_of(param) == "podaac":
        if not tok:
            raise ValueError(f"'{ALL[param]['short_name']}' comes from NASA PO.DAAC, which needs your free "
                             "Earthdata token. Add it under 'NASA Giovanni' in the sidebar.")
        try:
            s, units, note = _podaac_series(param, lat, lon, start, end, tok)
        except podaac.PodaacError as e:
            raise ValueError(f"NASA PO.DAAC: {e}")
        return s, dict(units=units, note=note)
    return _power_series(param, lat, lon, start, end), None


def get_many(reqs):
    """reqs: [(param, lat, lon, start, end), ...] -> [(series or None, exception or None), ...], fetched in
    parallel. Cached series come back instantly; only the missing ones go to NASA."""
    tok = gio_token()
    n_gio = sum(retrieve.is_giovanni(r[0]) for r in reqs)
    n_pod = sum(providers.provider_of(r[0]) == "podaac" for r in reqs)
    label = (f"🛰️ The Planeteer is fetching {len(reqs)} series from NASA at once…" if len(reqs) > 1 else
             "Asking NASA PO.DAAC (one small request per month, about a minute)…" if n_pod else
             "Asking NASA Giovanni (long hourly records come in several pieces)…" if n_gio else
             "🛰️ The Planeteer is fetching NASA data…")
    with st.spinner(label):
        out = run_parallel([partial(_fetch, *r, tok) for r in reqs])
    meta = st.session_state.setdefault("gio_meta", {})
    results = []
    for r, (res, err) in zip(reqs, out):
        if res is not None and res[1]:
            meta[r[0]] = res[1]
        results.append((res[0] if res is not None else None, err))
    return results


def get_series(param, lat, lon, start, end):
    """One series from any source (NASA POWER codes as-is; Giovanni codes start with 'GIO:')."""
    (s, err), = get_many([(param, lat, lon, start, end)])
    if err is not None:
        raise err
    return s


def var_info(code):
    """Catalog entry, with units learned from NASA's answer filled in when the catalog didn't know them."""
    v = dict(ALL[code])
    learned = (st.session_state.get("gio_meta") or {}).get(code)
    if learned and learned.get("units") and not v.get("units"):
        v["units"] = learned["units"]
    return v


def usable_codes(curated_first=True):
    """Variables offered in the menus: the featured POWER ones, plus all Giovanni ones once a token is in."""
    codes = list(VARIABLES)
    if gio_token():
        codes += sorted((k for k in ALL if retrieve.is_external(k)), key=lambda k: ALL[k]["name"])
    return codes


# ---- AI calls are cached so Streamlit reruns (any click) don't call the API again
class _NoCache(Exception):
    """Carries a fallback result out of a cached function so it is NOT cached."""
    def __init__(self, value):
        self.value = value


@st.cache_data(show_spinner=False)
def _ai_parse_cached(q, ai_on, gio_on=False):
    return nl.parse_question(q)


def _own_key_cached(kind, fn, q, gio_on):
    """Answers computed with a visitor's own Claude key are kept in that visitor's session only (so a click does not
    call Claude again), never in the shared cache, and are redone if the key changes."""
    store = st.session_state.setdefault("_own_key_parses", {})
    k = (kind, q, bool(gio_on), hash(nl.USER_API_KEY.get()))
    if k not in store:
        store[k] = fn(q)
    return store[k]


def ai_parse(q, ai_on, gio_on=False):
    if ai_on == "user":
        return _own_key_cached("parse", nl.parse_question, q, gio_on)
    return _ai_parse_cached(q, ai_on, gio_on)


@st.cache_data(show_spinner=False)
def question_pair(q, gio_on=False):
    return nl.find_pair(q)


@st.cache_data(show_spinner=False)
def _explain_cached(ctx_json, ai_on):
    out = nl.explain(json.loads(ctx_json))
    if ai_on and not out[1]:  # AI was expected but failed: show the fallback, don't cache it
        raise _NoCache(out)
    return out


@st.cache_data(show_spinner=False)
def _clues_cached(ctx_json, ai_on):
    out = nl.suggest_clues(json.loads(ctx_json))
    if ai_on and not out[1]:
        raise _NoCache(out)
    return out


@st.cache_data(show_spinner=False)
def _ai_parse_compare_cached(q, ai_on, gio_on=False):
    return nl.parse_compare(q)


def ai_parse_compare(q, ai_on, gio_on=False):
    if ai_on == "user":
        return _own_key_cached("compare", nl.parse_compare, q, gio_on)
    return _ai_parse_compare_cached(q, ai_on, gio_on)


@st.cache_data(show_spinner=False)
def _explain_compare_cached(ctx_json, ai_on):
    out = nl.explain_compare(json.loads(ctx_json))
    if ai_on and not out[1]:
        raise _NoCache(out)
    return out


def _cached(fn, ctx):
    try:
        return fn(json.dumps(ctx, sort_keys=True, default=str), nl.ai_mode())
    except _NoCache as e:
        return e.value


@st.cache_data(show_spinner=False)
def variable_table(n_vars):
    return explorer.build_table(ALL, VARIABLES)


ANY_PLACE, CUSTOM = "Type any place…", "Custom lat/lon"


def place_picker(col, label, key, index=0):
    """Preset places, any city/country/US state typed by name (offline lookup), or raw coordinates.
    Returns (name, lat, lon) or None while the user hasn't finished choosing."""
    opts = list(PLACES) + ([ANY_PLACE] if geo.AVAILABLE else []) + [CUSTOM]
    choice = col.selectbox(label, opts, index=min(index, len(PLACES) - 1), key=key)
    if choice == ANY_PLACE:
        txt = col.text_input("City, country or US state", key=f"{key}_txt", placeholder="Chicago · Kenya · Paris, Texas")
        if not txt:
            return None
        m = geo.lookup(txt)
        if not m:
            col.warning(f"No place called '{txt}' in the list of cities over 15,000 people.")
            return None
        col.caption(f"Using **{m['label']}** ({m['lat']:.2f}, {m['lon']:.2f})")
        return m["label"], m["lat"], m["lon"]
    if choice == CUSTOM:
        lat = col.number_input("Latitude", -90.0, 90.0, 40.0, key=f"{key}_lat")
        lon = col.number_input("Longitude", -180.0, 180.0, -100.0, key=f"{key}_lon")
        return f"{lat:.2f}, {lon:.2f}", lat, lon
    return choice, *PLACES[choice]


def _set_override(q, fields):
    """'Did you mean' buttons: remember a correction for this exact question."""
    ov = st.session_state.get("ask_override")
    if not ov or ov.get("q") != q:
        ov = {"q": q, "fields": {}}
    ov["fields"].update(fields)
    st.session_state["ask_override"] = ov


# Streamlit forgets a widget's value on any run where the widget isn't drawn, and the clue view hides the
# mode's widgets. So their values are copied aside when a clue opens and put back when the user returns.
KEEP_WIDGETS = ("q", "cmp_q", "ex_q", "ex_group", "ex_src", "ex_hourly", "ex_season", "menu_place", "cmp_va",
                "cmp_vb", "cmp_season", "gio_place", "gio_season")


def _push_followup(case):
    if not st.session_state.get("followup"):
        st.session_state["_kept"] = {k: st.session_state[k] for k in KEEP_WIDGETS if k in st.session_state}
    st.session_state.setdefault("followup", []).append(case)


def _restore_widgets():
    """Call before drawing the mode's widgets (no clue open)."""
    for k, v in (st.session_state.pop("_kept", None) or {}).items():
        if k not in st.session_state:
            st.session_state[k] = v


def _pop_followup():
    if st.session_state.get("followup"):
        st.session_state["followup"].pop()


def _clear_followup():
    st.session_state["followup"] = []


def trend_figure(res, title, units, season):
    s, anom, fit, m = res["series"], res["anom"], res["fit"], res["metrics"]
    mk = "o-" if season else "-"
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].plot(s.index, s.values, mk, lw=1, ms=3, color=palette.OCEAN)
    ax[0].set_ylabel(units)
    ax[0].set_title(title + (f" ({SEASON_LABELS[season]} mean each year)" if season else " (monthly)"), fontsize=11)
    kept = m.get("seasonal_cycle_kept")
    ax[1].plot(anom.index, anom.values, mk, lw=1, ms=3, color=palette.MUTED,
               label=("difference from the overall average (seasonal cycle kept)" if kept else
                      "difference from normal" + ("" if season else " for that month")))
    ax[1].plot(fit.index, fit.values, lw=2.5, color=palette.CORAL if m["significant"] else palette.SUN,
               label=f"trend {m['ols_slope_per_decade']:+.3g} {units}/decade")
    ax[1].axhline(0, color=palette.GRID, lw=0.6)
    ax[1].set_ylabel(f"anomaly ({units})")
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def neighbor_check(p, own_slope, series=None):
    """Trend at the 4 surrounding points (~100 km). `series`: already-fetched neighbour series (any may be None);
    if not given, they are fetched here, in parallel."""
    if series is None:
        pts = power.neighbor_points(p["lat"], p["lon"])
        series = [s for s, _ in get_many([(p["variable"], la, lo, p["start_year"], p["end_year"]) for la, lo in pts])]
    out = []
    for s in series:
        if s is None:
            continue
        try:
            s, _ = data_check(p, var_info(p["variable"]), s)
            out.append(analyze(s, p["season"], keep_cycle())["metrics"])
        except Exception:
            pass
    if not out:
        return None
    return dict(n=len(out), same_direction=sum((m["sen_slope_per_decade"] > 0) == (own_slope > 0) for m in out),
                significant=sum(m["significant"] for m in out),
                slopes_per_decade=[round(m["sen_slope_per_decade"], 4) for m in out])


MAX_FORECASTS_PER_SESSION = 5


@st.cache_data(show_spinner=False, max_entries=100)
def _lstm_forecast(values, start, horizon, epochs):
    """Cached by the series itself, so reruns, Back and repeat clicks are instant."""
    return forecast_lstm.run(dict(values=list(values), start=start, horizon=horizon, epochs=epochs, members=3))


def render_forecast(series, var, p):
    """LSTM forecast, trained here in NumPy (forecast_lstm.py). Code computes everything; the AI is not involved."""
    key = "fc_" + "|".join(str(p.get(k)) for k in ("variable", "place", "start_year", "end_year"))
    with st.expander("🔮 Peek into the future with an LSTM (experimental)"):
        st.caption("The straight line handles the trend and the seasons; a small LSTM, trained right now on this "
                   "series, learns what the line misses. Both are tested by forecasting 2 years from 4 earlier "
                   "dates and comparing with what actually happened. Check those tests before trusting a projection.")
        c1, c2 = st.columns(2)
        horizon = c1.slider("Months ahead", 12, 120, 60, 12, key=key + "_h")
        epochs = c2.slider("Training epochs (most runs stop early)", 40, 300, 120, 20, key=key + "_e")
        used = st.session_state.get("fc_used", 0)
        clicked = st.button("Train and forecast", key=key + "_b", disabled=used >= MAX_FORECASTS_PER_SESSION)
        if clicked and used < MAX_FORECASTS_PER_SESSION:
            st.session_state["fc_used"] = used + 1
            s = series.dropna()
            try:
                with st.spinner("The Planeteer is training LSTMs and testing them on past years (a few seconds)…"):
                    st.session_state[key] = _lstm_forecast(tuple(float(v) for v in s.values),
                                                           str(s.index[0].to_period("M")), horizon, epochs)
            except ValueError as e:  # gaps, too short, too long
                st.session_state.pop(key, None)
                st.warning(str(e))
        if used >= MAX_FORECASTS_PER_SESSION:
            st.caption("Forecast limit for this session reached. Reload the page to reset.")
        res = st.session_state.get(key)
        if not res:
            return
        n_short = res["short_months"]
        rows = [{"Model": m, "Error, next 2 years": round(v["rmse"], 3),
                 f"Error, next {n_short} months": round(v["rmse_short"], 3),
                 "Skill vs repeating last year": f"{v['skill']:+.0%}"} for m, v in res["scores"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config=glossary.column_config(**{"Error, next 2 years": "rmse",
                                                             f"Error, next {n_short} months": "rmse",
                                                             "Skill vs repeating last year": "skill"}))
        st.info(res["verdict"])
        st.caption(f"Line + LSTM beat the line alone in {res['wins']} of {res['n_folds']} backtests over two years, "
                   f"and in {res['wins_short']} of {res['n_folds']} over the first {n_short} months. "
                   f"Errors are typical misses (RMSE) in {var['units']}.")
        fig = forecast_lstm.figure(series, res, f"{var['name']}, {p['place']}", var["units"])
        st.pyplot(fig)
        plt.close(fig)
        st.caption("A forecast extends past patterns; it does not explain them, and it knows nothing about future "
                   f"emissions or policy. Trained in {res['seconds']} s with NumPy (no PyTorch).")


POWER_REFERENCE = ("The data was obtained from National Aeronautics and Space Administration (NASA) Langley Research "
                   "Center's Prediction Of Worldwide Energy Resources (POWER) project funded through the NASA Earth "
                   "Science Division.")
PODAAC_SOURCES = {  # dataset pages and citations as PO.DAAC publishes them
    "POD:SST": dict(title="NOAA Extended Reconstructed Sea Surface Temperature (ERSST) v5, distributed by NASA PO.DAAC",
                    about="https://podaac.jpl.nasa.gov/dataset/REYNOLDS_NCDC_L4_MONTHLY_V5", doi="10.5067/ERSST-L4N50",
                    cite="Smith, T., Reynolds, R. 2015. NCDC Smith & Reynolds Historical Reconstructed Sea Surface "
                         "Temperature Data Sets. Ver. 5. PO.DAAC, CA, USA. Dataset accessed {date} at "
                         "https://doi.org/10.5067/ERSST-L4N50"),
    "POD:SSH": dict(title="NASA-SSH Simple Gridded Sea Surface Height, distributed by NASA PO.DAAC",
                    about="https://podaac.jpl.nasa.gov/NASA-SSH", doi=None,
                    cite="Willis, J. K., S. Fournier, K. Marlis, E. Killett and J. Sanchez. NASA-SSH Simple Gridded Sea "
                         "Surface Height from Standardized Reference Missions. PO.DAAC, CA, USA. Dataset accessed {date}."),
}


def _where(p):
    lat, lon = float(p["lat"]), float(p["lon"])
    return (f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}")


def source_info(p, var, series):
    """Where one series came from: title, citation text and links (no tokens ever go into these links)."""
    code, today = p["variable"], dt.date.today().isoformat()
    span = (f"{series.index[0]:%b %Y}–{series.index[-1]:%b %Y}" if series is not None and len(series)
            else f"{p['start_year']}–{p['end_year']}")
    detail = f"Monthly values at {_where(p)} ({p['place']}), {span}."
    meta = (st.session_state.get("gio_meta") or {}).get(code, {})
    if retrieve.is_giovanni(code):
        gid = code[len(giovanni_catalog.PREFIX):]
        product = meta.get("product")
        about = (f"https://disc.gsfc.nasa.gov/datasets/{product.replace('.', '_')}/summary" if product
                 else f"https://disc.gsfc.nasa.gov/datasets?keywords={gid}")
        doi = meta.get("doi")
        cite = (f"{var['name']}. Product {product or gid}, NASA Goddard Earth Sciences Data and Information Services "
                f"Center (GES DISC)" + (f", https://doi.org/{doi}" if doi else "") +
                f". Accessed {today} via the NASA Giovanni time-series API.")
        return dict(title=f"NASA GES DISC via Giovanni · {var.get('source', '').split('·')[-1].strip()}",
                    detail=detail + (f" {meta['note']}" if meta.get("note") else ""), cite=cite,
                    view=about, view_label="Open the dataset at NASA GES DISC",
                    about=f"https://doi.org/{doi}" if doi else None, about_label="Dataset DOI",
                    view_note="Downloading from NASA needs your own free Earthdata login.")
    if providers.provider_of(code) == "podaac":
        info = PODAAC_SOURCES.get(code, {})
        concept = podaac.PRODUCTS[code[len(podaac.PREFIX):]]["concept"]
        return dict(title=info.get("title", var.get("source", "NASA PO.DAAC")),
                    detail=detail + (f" {meta['note']}" if meta.get("note") else ""),
                    cite=info.get("cite", "NASA PO.DAAC. Dataset accessed {date}.").format(date=today),
                    view=f"https://search.earthdata.nasa.gov/search/granules?p={concept}",
                    view_label="Browse the files in NASA Earthdata Search",
                    about=info.get("about"), about_label="About this dataset",
                    view_note="Downloading from NASA needs your own free Earthdata login.")
    # NASA POWER: the exact request the app made, as a CSV anyone can open (no login)
    start = max(int(p["start_year"]), 1984 if code.startswith("ALLSKY") else 1981)
    end = min(int(p["end_year"]), power.LAST_FULL_YEAR)
    view = (f"{power.URL}?parameters={code}&community=AG&latitude={round(float(p['lat']), 3)}"
            f"&longitude={round(float(p['lon']), 3)}&start={start}&end={end}&format=CSV")
    underlying = var.get("source", "").replace("NASA POWER", "").strip(" ()")
    underlying = "" if underlying in ("SOURCE", "POWER") else underlying
    return dict(title="NASA POWER (Prediction Of Worldwide Energy Resources)" + (f", from {underlying}" if underlying else ""),
                detail=detail + f" Parameter `{code}`.",
                cite=f"{POWER_REFERENCE} POWER monthly point API, parameter {code}, accessed {today}.",
                view=view, view_label="View this exact data at NASA (CSV)",
                about="https://power.larc.nasa.gov/docs/", about_label="About NASA POWER", view_note=None)


def check_years(p, var):
    """The years this variable can actually cover inside the requested span (what the data check expects)."""
    code, y0, y1 = p["variable"], int(p["start_year"]), int(p["end_year"])
    if not retrieve.is_external(code):  # NASA POWER: complete years only; solar from 1984
        y0 = max(y0, 1984 if code.startswith("ALLSKY") else 1981)
    elif var.get("first_year"):
        y0 = max(y0, int(var["first_year"]))
    if var.get("last_year"):
        y1 = min(y1, int(var["last_year"]))
    return y0, min(y1, power.LAST_FULL_YEAR)


def data_check(p, var, series):
    """Range-check and completeness report for one series -> (clean_series, report)."""
    return quality.check(series, var.get("units"), *check_years(p, var))


def render_check(qc):
    color, icon, word = {"green": ("green", "✅", "Data check passed"),
                         "amber": ("orange", "⚠️", "Data check: worth a look"),
                         "red": ("red", "⛔", "Data check: incomplete record")}[qc["level"]]
    st.badge(word, icon=icon, color=color)
    st.caption(quality.headline(qc) + ".")
    if qc["removed"]:
        shown = ", ".join(f"{r['month']} ({r['value']:g})" for r in qc["removed"][:6])
        st.caption(f"Removed as physically impossible (allowed {qc['bounds'][0]:g} to {qc['bounds'][1]:g}): "
                   + shown + (" …" if len(qc["removed"]) > 6 else ""))


def source_box(p, var, series, tag, heading=None, qc=None):
    """Prominent citation for one series, with a direct link to the source data and a download."""
    info = source_info(p, var, series)
    with st.container(border=True):
        st.markdown(f"##### 📚 Source{': ' + heading if heading else ''}")
        if qc:
            render_check(qc)
        st.markdown(f"**{info['title']}**")
        st.caption(info["detail"])
        st.caption("Cite as:")
        st.code(info["cite"], language=None, wrap_lines=True)
        b1, b2, b3 = st.columns(3)
        short = heading is not None  # narrow column in compare mode: short labels
        b1.link_button("🔗 View at NASA" if short else f"🔗 {info['view_label']}", info["view"], width="stretch",
                       help=info["view_label"])
        if info.get("about"):
            b2.link_button("About" if short else info["about_label"], info["about"], width="stretch",
                           help=info["about_label"])
        if series is not None:
            b3.download_button("⬇️ CSV" if short else "⬇️ Download what we used (CSV)", series.rename(p["variable"]).to_csv(),
                               f"{p['variable']}_{p['start_year']}-{p['end_year']}.csv", mime="text/csv",
                               key=f"src_dl_{tag}_{p['variable']}_{p['lat']}_{p['lon']}", width="stretch")
        if info.get("view_note"):
            st.caption(info["view_note"])


def investigate(p, question=None, series=None, var=None):
    """p = validated params. `series`/`var` are passed for Giovanni CSV case files."""
    want_nb = (series is None and st.session_state.get("do_neighbors")
               and not retrieve.is_external(p["variable"]))  # Giovanni/PO.DAAC: each point costs quota or time
    nb_series = None
    if series is None:  # the point itself and (optionally) its 4 neighbours, all fetched at the same time
        reqs = [(p["variable"], p["lat"], p["lon"], p["start_year"], p["end_year"])]
        if want_nb:
            reqs += [(p["variable"], la, lo, p["start_year"], p["end_year"])
                     for la, lo in power.neighbor_points(p["lat"], p["lon"])]
        got = get_many(reqs)
        s, err = got[0]
        if err is not None:
            raise err
        nb_series = [x for x, _ in got[1:]]
    else:
        s = series
    var = var or var_info(p["variable"])
    s, qc = data_check(p, var, s)  # range check + completeness, before any statistics
    res = analyze(s, p["season"], keep_cycle())
    m = res["metrics"]
    nb = neighbor_check(p, m["sen_slope_per_decade"], nb_series) if want_nb else None

    st.subheader(f"{var['name']} · {p['place']}")
    st.caption(f"**What is it?** {var['definition']}  ·  Source: {var['source']}")
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(trend_figure(res, var["name"], var["units"], p["season"]))
    with right:
        if qc["level"] == "red":
            st.error(f"**Data check: incomplete record.** {quality.headline(qc)}. Treat the verdict below with "
                     "caution.")
        if m["significant"]:
            st.success(f"**Verdict: a significant {m['direction']} trend.** The evidence holds up, Detective.")
        elif m["tests_agree"]:
            st.warning("**Verdict: no significant trend (yet).** Not enough evidence to call it.")
        else:
            st.warning("**Verdict: borderline.** The two tests disagree, so the case stays open.")
        c1, c2 = st.columns(2)
        c1.metric("Change per decade", f"{m['sen_slope_per_decade']:+.3g} {var['units']}",
                  help=glossary.tip("change_per_decade"))
        c2.metric("95% range (OLS)", f"{m['ols_ci95_low']:+.2g} to {m['ols_ci95_high']:+.2g}",
                  help=glossary.tip("ci95"))
        c1.metric("Mann-Kendall p", f"{m['mk_p_value']:.3g}", help=glossary.tip("mk_p"))
        c2.metric("t-test p", f"{m['ols_p_value']:.3g}", help=glossary.tip("ols_p"))
        st.caption(f"{m['n_points']} points, {m['start'][:4]}–{m['end'][:4]} · p < 0.05 counts as significant")
        em = emergence.years_to_significance(res)
        em_value = {"significant": f"Detectable since ~{em['detectable_from']}",
                    "on_track": f"~{em['more_years']:.0f} more years" if em["more_years"] is not None else "—",
                    "too_slow": "Centuries", "flat": "No trend to detect"}[em["status"]]
        st.metric("⏳ Years until significant", em_value, help=glossary.tip("years_until"))
        st.caption(emergence.describe(em, var["units"], m["direction"]))
        if nb:
            st.info(f"**Canvassing the neighbours:** {nb['same_direction']} of {nb['n']} surrounding points (~100 km) trend "
                    f"the same way; {nb['significant']} significantly.")
    ctx = dict(question=question, place=p["place"], season=p["season"], metrics=m, neighbors=nb, emergence=em,
               data_check={k: qc[k] for k in ("level", "coverage", "longest_gap", "issues")},
               variable={k: var[k] for k in ("name", "units", "definition")}, variable_id=p["variable"])
    text, used_ai = _cached(_explain_cached, ctx)
    if m.get("seasonal_cycle_kept"):
        st.info("**Seasonal cycle kept.** Each month is compared with the overall average, so the swing between "
                "summer and winter counts as noise in the tests. That makes a real trend harder to detect, and the "
                "tests' assumptions hold less well. Turn off *Keep the seasonal cycle* in the sidebar for the "
                "standard analysis.")
    elif keep_cycle() and p.get("season"):
        st.caption("The seasonal-cycle toggle has no effect here: one season gives one value per year, so there is "
                   "no cycle to keep.")
    if series is None:
        source_box(p, var, s, "main", qc=qc)
    else:
        render_check(qc)
    st.markdown("#### 🛰️ The Planeteer's report")
    st.markdown(text)
    st.caption(("Wording by AI from the computed numbers. " if used_ai else "") +
               "All statistics are calculated in code (SciPy), never by the AI.")

    if series is None:  # next clues (not for Giovanni CSV case files: no live data to follow up with)
        clues, clue_ai = _cached(_clues_cached, ctx)
        if clues:
            st.markdown("#### 🔍 Leads worth following, Detective")
            for i, c in enumerate(clues):
                cv = ALL[c["variable"]]
                a, b = st.columns([3, 2])
                a.markdown(f"**{cv['name']}**: {c['why']}")
                case = {**{k: p[k] for k in ("place", "lat", "lon", "season", "start_year", "end_year")},
                        "variable": c["variable"], "came_from": var["name"], "question": question}
                b.button(f"Follow the lead: {cv['name']}", key=f"clue_{p['variable']}_{c['variable']}_{i}",
                         on_click=_push_followup, args=(case,), width="stretch")
            st.caption(("Suggested by AI. " if clue_ai else "") +
                       "These are hunches to check, not explanations. Same place, months and years.")
    if series is None:  # live data only: case-file CSVs have no real place to look at
        space_view.render(p, p["variable"], var["name"])
    with st.expander("🗂️ Case file: exact numbers and data"):
        st.json(m)
        st.json({"data_check": qc})
        st.download_button("Download series (CSV)", res["series"].to_csv(), "series.csv")

    if not p.get("season"):
        render_forecast(res["series"], var, p)

    case_key = json.dumps({k: p.get(k) for k in ("variable", "place", "season", "start_year", "end_year")}, default=str)
    logged = st.session_state.setdefault("logged_cases", [])
    if HAS_MLFLOW and st.session_state.get("do_mlflow") and case_key not in logged:
        logged.append(case_key)
        mlflow.set_experiment("earth-trend-detective")
        with mlflow.start_run(run_name=f"{var['name']} · {p['place']}"):
            mlflow.log_params({k: p[k] for k in ("variable", "place", "lat", "lon", "season", "start_year", "end_year")})
            mlflow.log_metrics({k: float(v) for k, v in m.items() if isinstance(v, (int, float, bool))})


COMPARE_EXAMPLES = ["Is Svalbard warming faster than London?",  # NASA POWER only; checked against the data
                    "Is Phoenix warming faster than Seattle in summer?",
                    "Is Boston warming faster than London?"]


def _label(p, other):
    """Short label for one side: the place if the variable is shared, else variable + place."""
    name = ALL[p["variable"]]["name"]
    if p["variable"] == other["variable"]:
        return p["place"]
    return name if p["place"] == other["place"] else f"{name} ({p['place']})"


def render_compare(case, question=None):
    pa, pb = case["a"], case["b"]
    (sa, ea), (sb, eb) = get_many([(pa["variable"], pa["lat"], pa["lon"], pa["start_year"], pa["end_year"]),
                                   (pb["variable"], pb["lat"], pb["lon"], pb["start_year"], pb["end_year"])])
    if ea is not None or eb is not None:
        raise ea if ea is not None else eb
    va, vb = var_info(pa["variable"]), var_info(pb["variable"])
    sa, qa = data_check(pa, va, sa)
    sb, qb = data_check(pb, vb, sb)
    same_units = pa["variable"] == pb["variable"] or (va["units"] == vb["units"] and bool(va["units"]))
    c = compare.compare(sa, sb, pa["season"], same_units, keep_cycle())
    la, lb = _label(pa, pb), _label(pb, pa)
    text, kind = compare.headline(c, la, lb)
    if pa["variable"] == pb["variable"]:
        st.subheader(f"{va['name']} · {la} vs {lb}")
    elif pa["place"] == pb["place"]:
        st.subheader(f"{la} vs {lb} · {pa['place']}")
    else:
        st.subheader(f"{la} vs {lb}")
    st.caption(f"{season_text(pa['season'])} · {pa['start_year']}–{pa['end_year']}")
    if c["a"]["metrics"].get("seasonal_cycle_kept"):
        st.info("**Seasonal cycle kept.** The summer-winter swing stays in both series, so it counts as noise in the "
                "trend tests and can make the two look more alike than they really are. Turn off *Keep the seasonal "
                "cycle* in the sidebar for the standard analysis.")
    {"success": st.success, "warning": st.warning, "info": st.info}[kind](f"**{text}**")
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(compare.overlay_figure(c, la, lb, va["units"], vb["units"], pa["season"]))
    with right:
        rows = []
        for lab, m, u in ((f"A: {la}", c["a"]["metrics"], va["units"]), (f"B: {lb}", c["b"]["metrics"], vb["units"])) + \
                         (((("A − B (difference)", c["diff"]["metrics"], va["units"]),)) if c["diff"] else ()):
            rows.append({"": lab, "Change per decade": f"{m['sen_slope_per_decade']:+.3g} {u}",
                         "95% range": f"{m['ols_ci95_low']:+.2g} to {m['ols_ci95_high']:+.2g}",
                         "Mann-Kendall p": f"{m['mk_p_value']:.3g}",
                         "Significant?": "yes" if m["significant"] else "no"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config=glossary.column_config(**{"Change per decade": "change_per_decade",
                                                             "95% range": "ci95", "Mann-Kendall p": "mk_p",
                                                             "Significant?": "significant"}))
        if c["diff"]:
            st.caption("The **A − B** row tests the gap between them directly. If it is significant, one really "
                       "is changing faster than the other.")
        else:
            st.caption("Different units, so the lines are drawn in standard scores and there is no A − B test.")
    cr = c["corr"]
    if cr:
        st.markdown("#### 🤝 Are they in cahoots? How closely they move together")
        d, w = cr["detrended"], cr["with_trends"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Correlation (trends removed)", f"{d['r']:+.2f}",
                  help=glossary.tip("correlation"))
        m2.metric("95% range", f"{d['ci_low']:+.2f} to {d['ci_high']:+.2f}", help=glossary.tip("corr_range"))
        m3.metric("p-value", f"{d['p']:.2g}", help=glossary.tip("corr_p") + f" Here: effective sample size {d['n_eff']} of {d['n']}.")
        if cr["significant"]:
            st.success(f"A {cr['strength']} relationship that is statistically significant.")
        else:
            st.info(f"No statistically significant relationship (strength: {cr['strength']}).")
        if cr["shared_trend"]:
            st.warning(f"Including the trends the correlation looks much stronger ({w['r']:+.2f}), but that is just both "
                       "drifting over time. It is not evidence they are connected.")
        sc, lg = st.columns([2, 3])
        with sc:
            fig = compare.scatter_figure(c, la, lb, va["units"], vb["units"], pa["season"])
            st.pyplot(fig)
            plt.close(fig)
        with lg:
            if cr.get("lags"):
                st.caption("Does one lead the other? Correlation at each delay (exploratory: 25 delays were tried, so "
                           "treat a standout as a hint, not a finding).")
                fig = compare.lag_figure(c, la, lb)
                st.pyplot(fig)
                plt.close(fig)
                bl = cr["best_lag"]
                if bl["clear"]:
                    st.caption(f"Strongest at a {abs(bl['lag'])}-month delay (r = {bl['r']:+.2f} vs {bl['r_at_zero']:+.2f} "
                               "with no delay).")
            else:
                st.caption("Delay analysis needs a monthly record of at least 4 years, so it is skipped in season mode.")
        st.caption(f"Significance accounts for neighbouring months not being independent (effective sample "
                   f"{d['n_eff']} of {d['n']}); ordinary tests would overstate it. Rank-based (Spearman) check: "
                   f"{cr['spearman_detrended']:+.2f}.")
    else:
        st.caption("Not enough overlapping data to measure correlation.")
    ctx = dict(question=question, season=pa["season"], headline=text,
               a=dict(label=la, variable=va["name"], units=va["units"], metrics=c["a_brief"]),
               b=dict(label=lb, variable=vb["name"], units=vb["units"], metrics=c["b_brief"]),
               comparison=c["summary"])
    expl, used_ai = _cached(_explain_compare_cached, ctx)
    st.markdown("#### 🛰️ The Planeteer's report")
    st.markdown(expl)
    st.caption(("Wording by AI from the computed numbers. " if used_ai else "") +
               "All statistics are calculated in code (SciPy), never by the AI.")
    b1, b2 = st.columns(2)
    for col, side, lab in ((b1, pa, la), (b2, pb, lb)):
        solo = {**{k: side[k] for k in ("variable", "place", "lat", "lon", "season", "start_year", "end_year")},
                "came_from": "the comparison", "origin": "compare", "question": None}
        col.button(f"Open {lab} on its own", key=f"cmp_solo_{side['variable']}_{side['place']}_{lab}",
                   on_click=_push_followup, args=(solo,), width="stretch")
    s1, s2 = st.columns(2)
    with s1:
        source_box(pa, va, sa, "a", heading=f"A: {la}", qc=qa)
    with s2:
        source_box(pb, vb, sb, "b", heading=f"B: {lb}", qc=qb)
    with st.expander("🗂️ Case file: exact numbers"):
        st.json({"A": c["a_brief"], "B": c["b_brief"], "comparison": c["summary"]})



def _check_gio_token():
    """Runs when a token is pasted (or 'Check again'): ask NASA's quota endpoint whether it works."""
    tok = gio_token()
    if not tok:
        st.session_state.pop("gio_status", None)
        return
    try:
        q = giovanni_live.check_token(tok)
        st.session_state["gio_status"] = ("ok", f"Token works. {q:.0%} of this month's Giovanni quota used.")
    except giovanni_live.GiovanniError as e:
        kind = "bad" if e.kind == "auth" else "unknown"  # NASA unreachable isn't the token's fault
        msg = str(e) if kind == "bad" else f"Couldn't check the token right now ({e}). It may still work."
        st.session_state["gio_status"] = (kind, msg)


def _forget_gio_token():
    st.session_state["gio_token"] = ""
    for k in ("gio_status", "gio_meta"):
        st.session_state.pop(k, None)


# ------------------------------------------------------------------ UI
st.title("🌍 PLaNETS: Plain-Language NASA Earth Trend Sleuth")
st.caption("**You're the detective.** Ask what's changing on Earth, and your sidekick, the Planeteer, fetches "
           "the NASA evidence and puts it on trial. Statistics decide what's real.")

if not st.session_state.get("_link_loaded"):  # a shared link: reproduce what it describes
    st.session_state["_link_loaded"] = True
    try:
        opened = share.read(st.query_params.to_dict(), nl.validate)
    except ValueError as e:
        opened = None
        st.warning(str(e))
    if opened:
        kind, val = opened
        if st.query_params.get("sc") == "1":
            st.session_state["keep_cycle"] = True
        if kind == "ask":
            st.session_state.update(mode="Ask a question", q=val)
        elif kind == "compare_q":
            st.session_state.update(mode="Compare two trends", cmp_q=val)
        elif kind == "case":
            st.session_state["followup"] = [{**val, "came_from": "a shared link", "origin": "link", "question": None}]
        elif kind == "pair":
            st.session_state.update(mode="Compare two trends", cmp_q_done="",
                                    cmp_case={**val, "question": None, "used_ai": False})

with st.sidebar:
    mode = st.radio("How shall we investigate, Detective?", ["Ask a question", "Compare two trends", "Pick from menus", "Case files (Giovanni CSV)",
                     "Explore variables"],
                    on_change=_clear_followup, key="mode", format_func=lambda m: MODE_LABELS.get(m, m))
    share_slot = st.empty()  # filled at the end of the run with a link to whatever is on screen
    stt = st.session_state.get("gio_status")
    gio_label = ("add your login" if not gio_token() else
                 "token rejected ⚠️" if stt and stt[0] == "bad" else
                 "token added (not checked)" if stt and stt[0] == "unknown" else "connected ✅")
    with st.expander("🛰️ NASA Giovanni: " + gio_label, expanded=bool(stt and stt[0] == "bad")):
        st.caption("Adds satellite, model and ocean variables (dust, ozone, snow, soil moisture, pollution, sea "
                   "temperature, sea level…) to every mode, fetched live with **your own** free Earthdata token. "
                   "Kept in this browser session only: never saved, logged or put in links, and sent only to NASA.")
        if not gio_token():
            st.markdown(
                f"1. Sign in at **[Earthdata Login]({giovanni_live.TOKEN_URL})** (or *Register*: free, about 2 minutes).\n"
                "2. Open the **Generate Token** tab, click **Generate Token**, then copy it (the long text "
                "starting with `ey…`).\n"
                "3. Paste it below. It's checked with NASA straight away.\n\n"
                "First time using Giovanni? Under **Applications → Authorized Apps**, approve "
                "**NASA GESDISC DATA ARCHIVE**, or NASA will refuse the data.")
        st.text_input("Earthdata token", type="password", key="gio_token", placeholder="paste token",
                      on_change=_check_gio_token)
        if stt and gio_token():
            {"ok": st.success, "bad": st.error, "unknown": st.warning}[stt[0]](stt[1])
        if gio_token():
            g1, g2 = st.columns(2)
            g1.button("Check again", key="gio_check", on_click=_check_gio_token, width="stretch")
            g2.button("Forget token", key="gio_forget", on_click=_forget_gio_token, width="stretch")
        n_gio = sum(1 for k in ALL if retrieve.is_giovanni(k))
        st.caption(f"{n_gio} Giovanni variables available" +
                   ("" if giovanni_catalog.is_built() else " (starter list; run `python -m planets.sources.giovanni_catalog --build` for "
                                                          "NASA's full list)"))
    retrieve.GIOVANNI_ON.set(bool(gio_token()))
    with st.expander("🤖 Use your own Claude key (optional)"):
        st.caption("Optional: the app works without it. With a key, Claude words the explanations and picks the "
                   "variable from your question. It's billed to **your** Anthropic account, kept in this browser "
                   "session only (never saved, logged or put in links), and sent only to Anthropic. All statistics "
                   "are still computed in code.")
        st.text_input("Anthropic API key", type="password", key="user_claude_key", placeholder="sk-ant-…")
        own_key = nl.set_user_key(st.session_state.get("user_claude_key"))
        if st.session_state.get("user_claude_key") and not own_key:
            st.warning("That doesn't look like an Anthropic key (they start with sk-ant-).")
        if own_key:
            k1, k2 = st.columns(2)
            if k1.button("Test key", key="claude_test", width="stretch"):
                ok, msg = nl.check_user_key()
                (st.success if ok else st.error)(msg)
            k2.button("Forget key", key="claude_forget", width="stretch",
                      on_click=lambda: st.session_state.update(user_claude_key=""))
    st.checkbox("Compare with nearby points (slower)", key="do_neighbors")
    st.toggle("Keep the seasonal cycle in the trend tests", key="keep_cycle",
              help="Off (standard): each month is compared with its own normal, so only real change over the years "
                   "is tested. On: the summer-winter swing stays in the data and counts as noise, which shows why "
                   "scientists usually remove it. Applies to all-months analyses, not single seasons.")
    if HAS_MLFLOW:
        st.checkbox("Log each case to MLflow", key="do_mlflow")
    st.divider()
    st.caption("AI: " + {"user": "using your Claude key ✅", "shared": "connected ✅",
                         "": "no API key, using keyword matching + template text"}[nl.ai_mode()])
    st.caption({"ready": "Search: keywords + meaning (embeddings)",
                "loading": "Search: keywords only while the meaning model loads",
                "failed": "Search: keywords only (meaning model unavailable; retrying)",
                "off": "Search: keywords only"}[retrieve.model_status()])
    st.caption(f"Catalog: {sum(1 for k in ALL if not retrieve.is_external(k))} NASA POWER variables"
               + (f" + {sum(1 for k in ALL if retrieve.is_giovanni(k))} Giovanni" if gio_token() else "")
               + (f" + {sum(1 for k in ALL if providers.provider_of(k) == 'podaac')} PO.DAAC ocean" if gio_token() else ""))
    st.caption("Places: any city over 15,000 people, country or US state, no AI needed" if geo.AVAILABLE else
               "Places: 14 presets only without an AI key. Run `uv sync` to add the offline city list.")
    with st.expander("📖 Detective's glossary: what the numbers mean"):
        for label, text in glossary.TERMS.values():
            st.markdown(f"**{label}**: {text}")
    with st.expander("⭐ Featured variables"):
        for v in VARIABLES.values():
            st.markdown(f"**{v['name']}** ({v['units']}): {v['definition']}")

try:
    trail = st.session_state.get("followup") or []
    if not trail:
        _restore_widgets()
    if trail:  # following a clue: show it instead of the normal mode content
        fu = trail[-1]
        b1, b2 = st.columns([1, 6])
        b1.button("← Back", on_click=_pop_followup)
        if fu.get("origin") in ("explorer", "compare", "link"):
            b2.caption(f"Opened from **{fu['came_from']}**")
        else:
            b2.caption(f"Following a lead from **{fu['came_from']}** · same place, months and years"
                       + (f" · lead {len(trail)} in this trail" if len(trail) > 1 else ""))
        investigate(fu, question=fu.get("question"))

    elif mode == "Ask a question":
        cols = st.columns(len(DEMO_QUESTIONS))
        for c, dq in zip(cols, DEMO_QUESTIONS):
            if c.button(dq, width="stretch"):
                st.session_state["q"] = dq
        q = st.text_input("What's your hunch, Detective?", key="q", placeholder="Is Phoenix getting hotter in summer?")
        pair = question_pair(q, bool(gio_token())) if q else None
        one_only = bool(pair) and st.checkbox("Investigate just one variable instead", key=f"ask_one_{q}")
        if pair and not one_only:  # "hotter and drier": test the two variables against each other
            st.info("Your question names more than one variable, so they are tested against each other. "
                    "Tick the box above to investigate just one.")
            st.caption(f"Understood as: **{ALL[pair['a']['variable']]['name']}** at **{pair['a']['place']}** "
                       f"and **{ALL[pair['b']['variable']]['name']}** at **{pair['b']['place']}** · "
                       f"{season_text(pair['season'])} · {pair['start_year']}–{pair['end_year']} · keyword match")
            render_compare({**pair, "question": q}, question=q)
            if pair.get("others"):
                st.caption("Your question names more variables than a comparison can hold (two), so "
                           + ", ".join(ALL[v]["name"] for v in pair["others"]) + " was left out. Ask about it separately.")
        elif q:
            p0, used_ai = ai_parse(q, nl.ai_mode(), bool(gio_token()))
            p = dict(p0)
            ov = st.session_state.get("ask_override")
            if ov and ov.get("q") == q:  # a "did you mean" correction for this question
                p = {**nl.validate({**p0, **ov["fields"]}), "candidates": p0["candidates"],
                     "place_alts": p0.get("place_alts", [])}
                st.session_state["_link_case"] = p
            st.caption(f"The Planeteer heard: **{ALL[p['variable']]['name']}** at **{p['place']}** "
                       f"({p['lat']:.2f}, {p['lon']:.2f}) · {season_text(p['season'])} · "
                       f"{p['start_year']}–{p['end_year']}" + ("" if used_ai else " · keyword match"))
            other_vars = [cid for cid, _ in p0["candidates"] if cid != p["variable"]][:3]
            places = [dict(label=p0["place"], lat=p0["lat"], lon=p0["lon"])] + list(p0.get("place_alts", []))
            other_places = [x for x in places if x["label"] != p["place"]][:3]
            if other_vars or other_places:
                st.caption("Wrong suspect? Pick another:")
                btn_cols = st.columns(max(1, len(other_vars) + len(other_places)))
                for col, cid in zip(btn_cols, other_vars):
                    col.button(ALL[cid]["name"], key=f"dym_var_{cid}", width="stretch",
                               on_click=_set_override, args=(q, {"variable": cid}))
                for col, x in zip(btn_cols[len(other_vars):], other_places):
                    col.button(f"📍 {x['label']}", key=f"dym_place_{x['label']}", width="stretch",
                               on_click=_set_override, args=(q, {"place": x["label"], "lat": x["lat"], "lon": x["lon"]}))
            with st.expander(f"How the Planeteer picked the variable: searched {len(ALL)} catalog entries"):
                for cid, sc in p["candidates"]:
                    st.markdown(("✅ " if cid == p["variable"] else "· ") +
                                f"**{ALL[cid]['name']}** (`{cid}`), match score {sc}")
            if nl.looks_like_comparison(q):
                st.info("This looks like a comparison. 'Compare two trends' in the sidebar tests two places or "
                        "variables against each other directly.")
            st.map(pd.DataFrame({"lat": [p["lat"]], "lon": [p["lon"]]}), zoom=3, height=220)
            investigate(p, question=q)

    elif mode == "Compare two trends":
        st.caption("Line up two suspects: one variable in two places, or two variables in one place.")
        cols = st.columns(len(COMPARE_EXAMPLES))
        for col, ex in zip(cols, COMPARE_EXAMPLES):
            if col.button(ex, width="stretch", key=f"cmp_ex_{ex}"):
                st.session_state["cmp_q"] = ex
        cq = st.text_input("Which two trends should go head to head?", key="cmp_q",
                           placeholder="Is Svalbard warming faster than London?")
        if cq and cq != st.session_state.get("cmp_q_done"):
            st.session_state["cmp_q_done"] = cq
            st.session_state.pop("cmp_case", None)
            case, used_ai = ai_parse_compare(cq, nl.ai_mode(), bool(gio_token()))
            st.session_state["cmp_case"] = {**case, "question": cq, "used_ai": used_ai}
        with st.expander("Or pick from menus", expanded=not st.session_state.get("cmp_case")):
            m1, m2 = st.columns(2)
            va_id = m1.selectbox("A: variable", usable_codes(), format_func=lambda k: ALL[k]["name"], key="cmp_va")
            pa_pick = place_picker(m1, "A: place", key="cmp_pa")
            vb_id = m2.selectbox("B: variable", ["(same as A)"] + usable_codes(), key="cmp_vb",
                                 format_func=lambda k: k if k.startswith("(") else ALL[k]["name"])
            pb_pick = place_picker(m2, "B: place", key="cmp_pb", index=1)
            season = m1.selectbox("Months", list(SEASON_LABELS), format_func=SEASON_LABELS.get, key="cmp_season")
            y0, y1 = st.slider("Years", 1981, power.LAST_FULL_YEAR, (1981, power.LAST_FULL_YEAR), key="cmp_years")
            if st.button("Compare", type="primary"):
                vb_real = va_id if vb_id.startswith("(") else vb_id
                if not pa_pick or not pb_pick:
                    st.warning("Finish choosing both places first.")
                elif vb_real == va_id and pa_pick[0] == pb_pick[0]:
                    st.warning("A and B are identical. Change a place or a variable.")
                else:
                    mk = lambda v, pk: nl.validate(dict(variable=v, place=pk[0], lat=pk[1], lon=pk[2],
                                                        season=season, start_year=y0, end_year=y1))
                    st.session_state["cmp_case"] = dict(a=mk(va_id, pa_pick), b=mk(vb_real, pb_pick),
                                                        question=None, used_ai=False)
                    st.session_state["cmp_q_done"] = st.session_state.get("cmp_q", "")
        case = st.session_state.get("cmp_case")
        if case:
            if case.get("question"):
                st.caption(f"Understood as: **{ALL[case['a']['variable']]['name']}** at **{case['a']['place']}** "
                           f"vs **{ALL[case['b']['variable']]['name']}** at **{case['b']['place']}**"
                           + ("" if case.get("used_ai") else " · keyword match"))
            render_compare(case, question=case.get("question"))

    elif mode == "Pick from menus":
        c1, c2, c3 = st.columns(3)
        vid = c1.selectbox("Variable", usable_codes(), format_func=lambda k: ALL[k]["name"])
        pick = place_picker(c2, "Place", key="menu_place")
        season = c3.selectbox("Months", list(SEASON_LABELS), format_func=SEASON_LABELS.get)
        fy = int(ALL[vid].get("first_year") or 1981)
        ly = int(ALL[vid].get("last_year") or power.LAST_FULL_YEAR)
        d0 = max(fy, ly - 19) if ALL[vid].get("step") in ("hourly", "half-hourly") else fy
        y0, y1 = st.slider("Years", min(fy, 1981), ly, (d0, ly))
        if ALL[vid].get("step") in ("hourly", "half-hourly"):
            st.caption("Hourly data: long ranges are downloaded in several requests and can take a minute or two.")
        if st.button("Investigate", type="primary", disabled=pick is None):  # kept in session so it survives reruns
            st.session_state["menu_case"] = nl.validate(dict(variable=vid, place=pick[0], lat=pick[1], lon=pick[2],
                                                             season=season, start_year=y0, end_year=y1))
        if st.session_state.get("menu_case"):
            investigate(st.session_state["menu_case"])

    elif mode == "Explore variables":
        df = variable_table(len(ALL))
        cases = case_files.case_files()
        shown = explorer.filter_table(df)
        st.subheader("🔭 The evidence locker: what can you investigate?")
        st.markdown(
            f"A **variable** is one thing NASA measures about the Earth and tracks over time. This app knows "
            f"**{int(df['Featured'].sum())} featured** variables with plain-language definitions, "
            f"**{int((shown['Source'] == 'NASA POWER').sum()) - int(df['Featured'].sum())} more** from the NASA POWER catalog (any point on Earth, "
            f"1981 to last year), **{int((df['Source'] == 'NASA Giovanni').sum())} live NASA Giovanni** satellite and "
            f"model variables (with your Earthdata token), and **{len(cases)} satellite case "
            f"file{'s' if len(cases) != 1 else ''}**.")
        c1, c2 = st.columns([3, 2])
        text = c1.text_input("Search in plain words", key="ex_q", placeholder="cloudy, frost, wind, soil…")
        group = c2.selectbox("Group", ["All", "Featured"] + [g for g, _ in explorer.GROUPS] + ["Other"], key="ex_group")
        d1, d2 = st.columns([2, 3])
        src = d1.selectbox("Source", ["All sources", "NASA POWER", "NASA Giovanni", "NASA PO.DAAC"], key="ex_src")
        hourly = d2.checkbox("Include hour-by-hour POWER variants (about 1,100 near-duplicates)", key="ex_hourly")
        ranked = [c for c, _ in search(text, k=60, giovanni=True)] if text else None
        view = explorer.filter_table(df, group=group, text=text, show_hourly=hourly, ranked_codes=ranked or None,
                                     source=None if src == "All sources" else src)
        if src not in ("NASA POWER",) and not gio_token():
            st.info("Giovanni and PO.DAAC variables can be browsed here; to investigate one, add your Earthdata "
                    "token under 'NASA Giovanni' in the sidebar.")
        st.caption(f"{len(view)} variable{'s' if len(view) != 1 else ''}" + (" · best matches first" if ranked else " · featured first"))
        st.dataframe(view[["Variable", "Source", "Units", "Group", "Definition", "Code"]], hide_index=True,
                     width="stretch", height=320)
        if len(view):
            code = st.selectbox("Look closer at", list(view["Code"]), format_func=lambda c: f"{ALL[c]['name']} ({c})")
            v = ALL[code]
            st.markdown(f"#### {v['name']}")
            st.markdown(f"{v['definition']}\n\n**Units:** {v['units'] or 'none'} · **Group:** "
                        f"{explorer.group_of(v['name'], v['definition'])} · **Code:** `{code}`")
            if retrieve.is_giovanni(code):
                st.caption(f"From NASA Giovanni · {v.get('step') or 'native'} data, averaged to months · "
                           f"available from about {v.get('first_year') or '?'}" +
                           ("" if v.get("verified") else " · ID not yet confirmed by NASA's list"))
            elif providers.provider_of(code) == "podaac":
                st.caption("From NASA PO.DAAC · ocean only, averaged to months · "
                           f"available from about {v.get('first_year') or '?'} · needs your Earthdata token")
            elif code not in VARIABLES:
                st.caption("From the full NASA POWER catalog: not every variable has monthly data at every place.")
            a, b, c = st.columns(3)
            pick = place_picker(a, "Place", key="ex_place")
            season = b.selectbox("Months", list(SEASON_LABELS), format_func=SEASON_LABELS.get, key="ex_season")
            if pick:
                case = {**nl.validate(dict(variable=code, place=pick[0], lat=pick[1], lon=pick[2], season=season)),
                        "came_from": "the variable explorer", "origin": "explorer", "question": None}
                needs_tok = providers.needs_token(code) and not gio_token()
                c.button("Investigate this here", type="primary", on_click=_push_followup, args=(case,),
                         width="stretch", disabled=needs_tok)
                if needs_tok:
                    c.caption("Needs your Earthdata token (sidebar).")
        st.markdown("#### Satellite case files (NASA Giovanni)")
        if cases:
            st.dataframe(pd.DataFrame([dict(Variable=cv["name"], Units=cv["units"], Group=cv["group"],
                                            Region=cv["region"], Definition=cv["definition"], Source=cv["source"])
                                       for cv in cases]), hide_index=True, width="stretch")
        st.caption("These cover what POWER lacks (ozone, NO₂, haze, groundwater). Open them in "
                   "'Case files (Giovanni CSV)' mode, where you can also get more.")
        with st.expander("Where the data comes from"):
            st.markdown(
                "- **[NASA POWER](https://power.larc.nasa.gov)** (Prediction Of Worldwide Energy Resources): weather "
                "variables come from **MERRA-2**, NASA's reconstruction of global weather "
                "since 1980; sunlight and cloud variables come from the **CERES** satellite instruments.\n"
                "- **NASA Giovanni** (GES DISC): live point time series from models and satellites such as "
                "**GLDAS** (land), **MERRA-2** (aerosols, ozone), **GPM** (rain) and **OMI** (NO₂), using your own "
                "Earthdata token; plus area-average case files.\n"
                "- **NASA PO.DAAC**: sea surface temperature (NOAA ERSST v5) and sea level (NASA-SSH altimetry), ocean "
                "only, read live with your Earthdata token.\n"
                f"- {geo.ATTRIBUTION}. A country or state is represented by its largest city.")

    else:
        cases = case_files.case_files()
        if not cases:
            st.info("No case files yet. Use the links below, then save each CSV into the `data/` folder.")
        else:
            i = st.selectbox("Case file", range(len(cases)),
                             format_func=lambda i: f"{cases[i]['name']} · {cases[i]['region']}")
            cv = cases[i]
            if cv["synthetic"]:
                st.warning("This file is synthetic test data, not real measurements. Replace it before a demo.")
            st.caption(f"**What is it?** {cv['definition']}")
            seasons = list(SEASON_LABELS)
            c1, c2 = st.columns(2)
            season = c1.selectbox("Months", seasons, index=seasons.index(cv["season"]) if cv["season"] in seasons else 0,
                                  format_func=SEASON_LABELS.get, key=f"csv_season_{os.path.basename(cv['file'])}")
            c2.caption(f"Units: {cv['units']} · Region: {cv['region']} · File: `{os.path.basename(cv['file'])}`")
            if st.button("Investigate", type="primary"):  # keep the case in session so it survives reruns
                st.session_state["csv_case"] = dict(file=cv["file"], season=season)
            case = st.session_state.get("csv_case")
            if case and case["file"] == cv["file"]:
                investigate(dict(variable=cv["name"], place=cv["region"], lat=0, lon=0,
                                 season=case["season"], start_year=0, end_year=0),
                            series=load_giovanni_csv(cv["file"]), var=cv)
        with st.expander("Get more case files from NASA Giovanni", expanded=not cases):
            st.markdown("Each link opens Giovanni with the request filled in. Sign in with a free Earthdata login, "
                        "click **Plot Data**, then **Downloads → CSV**, and save the file into the `data/` folder. "
                        "It appears here on the next refresh, described automatically.")
            for r in case_files.RECIPES:
                st.markdown(f"- **[{r['title']}]({r['url']})**: {r['why']} "
                            f"If the variable isn't preselected, search Giovanni for `{r['keyword']}`.")
except Exception as e:
    st.error(f"**Dead end.** {e}")


# ------------------------------------------------------------------ shareable link for what's on screen
def _link_params():
    trail = st.session_state.get("followup") or []
    if trail:
        return share.case_params(trail[-1])
    if mode == "Ask a question" and st.session_state.get("q"):
        lc = st.session_state.get("_link_case")
        ov = st.session_state.get("ask_override")
        if lc and ov and ov.get("q") == st.session_state["q"]:
            return share.case_params(lc)
        return {"q": st.session_state["q"]}
    if mode == "Compare two trends" and st.session_state.get("cmp_case"):
        c = st.session_state["cmp_case"]
        return {"cq": c["question"]} if c.get("question") else share.pair_params(c)
    if mode == "Pick from menus" and st.session_state.get("menu_case"):
        return share.case_params(st.session_state["menu_case"])
    return {}


_lp = _link_params()
if _lp and keep_cycle():
    _lp["sc"] = "1"
if _lp != st.query_params.to_dict():
    st.query_params.from_dict(_lp)
if _lp:
    with share_slot.container():
        with st.expander("🔗 Share this case"):
            st.code(share.full_url(_lp, getattr(st.context, "url", None)), language=None, wrap_lines=True)
            st.caption("Opens exactly this result. Giovanni and PO.DAAC data still need the visitor's own "
                       "Earthdata token; no keys or tokens are ever put in links.")
