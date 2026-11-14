"""PLaNETS (Plain-Language NASA Earth Trend Sleuth) - interactive app.
Run:   pip install streamlit pandas numpy scipy matplotlib requests   (+ anthropic or openai, + mlflow: optional)
       streamlit run app.py
Flow:  question -> [AI or keywords] parameters -> NASA POWER / Giovanni CSV -> analyze() in code -> [AI or template] explanation
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:  # lets worker threads share this session's Streamlit context (cache, no "missing ScriptRunContext" noise)
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:  # older Streamlit or test stub
    add_script_run_ctx = get_script_run_ctx = None

import compare
import explorer
import forecast_client
import glossary
import geo
import giovanni
import giovanni_catalog
import giovanni_live
import retrieve
import nl
import power
from catalog import DEMO_QUESTIONS, PLACES, VARIABLES
from retrieve import ALL, search
from trend_detective import analyze, load_giovanni_csv

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

st.set_page_config(page_title="PLaNETS: Plain-Language NASA Earth Trend Sleuth", page_icon="🔎", layout="wide")
SEASON_LABELS = {None: "All months (seasonal cycle removed)", "DJF": "Dec–Feb", "MAM": "Mar–May",
                 "JJA": "Jun–Aug", "SON": "Sep–Nov"}


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _power_series(param, lat, lon, start, end):
    return power.fetch_point(param, lat, lon, start, end)


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=200)
def _giovanni_series(gid, lat, lon, start, end, step, _token):
    """_token starts with an underscore so Streamlit never hashes or stores it as part of the cache key."""
    got = giovanni_live.fetch(gid, lat, lon, start, end, _token, step=step)
    return got["series"], got["meta"].get("display_units") or "", got["note"]


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
            s, units, note = _giovanni_series(param[len(giovanni_catalog.PREFIX):], lat, lon, start, end,
                                              ALL[param].get("step") or "", tok)
        except giovanni_live.GiovanniError as e:
            msg = f"NASA Giovanni: {e}"
            if e.raw and e.kind in ("parse", "request"):
                msg += f"\n\nWhat NASA sent back: `{e.raw[:200]}`"
            raise ValueError(msg)
        return s, dict(units=units, note=note)
    return _power_series(param, lat, lon, start, end), None


def get_many(reqs):
    """reqs: [(param, lat, lon, start, end), ...] -> [(series or None, exception or None), ...], fetched in
    parallel. Cached series come back instantly; only the missing ones go to NASA."""
    tok = gio_token()
    n_gio = sum(retrieve.is_giovanni(r[0]) for r in reqs)
    label = (f"Fetching {len(reqs)} series from NASA at once…" if len(reqs) > 1 else
             "Asking NASA Giovanni (long hourly records come in several pieces)…" if n_gio else "Fetching NASA data…")
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
        codes += sorted((k for k in ALL if retrieve.is_giovanni(k)), key=lambda k: ALL[k]["name"])
    return codes


# ---- AI calls are cached so Streamlit reruns (any click) don't call the API again
class _NoCache(Exception):
    """Carries a fallback result out of a cached function so it is NOT cached."""
    def __init__(self, value):
        self.value = value


@st.cache_data(show_spinner=False)
def ai_parse(q, ai_on, gio_on=False):
    return nl.parse_question(q)


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
def ai_parse_compare(q, ai_on, gio_on=False):
    return nl.parse_compare(q)


@st.cache_data(show_spinner=False)
def _explain_compare_cached(ctx_json, ai_on):
    out = nl.explain_compare(json.loads(ctx_json))
    if ai_on and not out[1]:
        raise _NoCache(out)
    return out


def _cached(fn, ctx):
    try:
        return fn(json.dumps(ctx, sort_keys=True, default=str), nl.llm_available())
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
    ax[0].plot(s.index, s.values, mk, lw=1, ms=3, color="#33658a")
    ax[0].set_ylabel(units)
    ax[0].set_title(title + (f" ({SEASON_LABELS[season]} mean each year)" if season else " (monthly)"), fontsize=11)
    ax[1].plot(anom.index, anom.values, mk, lw=1, ms=3, color="#9aa0a6",
               label="difference from normal" + ("" if season else " for that month"))
    ax[1].plot(fit.index, fit.values, lw=2.5, color="#d1495b" if m["significant"] else "#e0a030",
               label=f"trend {m['ols_slope_per_decade']:+.3g} {units}/decade")
    ax[1].axhline(0, color="k", lw=0.5)
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
            out.append(analyze(s, p["season"])["metrics"])
        except Exception:
            pass
    if not out:
        return None
    return dict(n=len(out), same_direction=sum((m["sen_slope_per_decade"] > 0) == (own_slope > 0) for m in out),
                significant=sum(m["significant"] for m in out),
                slopes_per_decade=[round(m["sen_slope_per_decade"], 4) for m in out])


MAX_FORECASTS_PER_SESSION = 5


def render_forecast(series, var, p):
    """Optional LSTM forecast. Code computes the comparison and verdict; the AI is not involved."""
    mode = forecast_client.backend()
    key = "fc_" + "|".join(str(p.get(k)) for k in ("variable", "place", "start_year", "end_year"))
    with st.expander("🔮 Forecast with an LSTM (experimental)"):
        if mode is None:
            st.caption("Not enabled on this server (needs the forecast service or PyTorch).")
            return
        st.caption("Trains a small LSTM on the first ~80% of the series, tests it on the last ~20% against two "
                   "simple baselines, then refits on everything and projects forward. Check the held-out errors "
                   "before trusting the projection.")
        c1, c2 = st.columns(2)
        horizon = c1.slider("Months ahead", 12, 120, 60, 12, key=key + "_h")
        epochs = c2.slider("Training epochs", 50, 300, 150, 50, key=key + "_e")
        used = st.session_state.get("fc_used", 0)
        clicked = st.button("Train and forecast", key=key + "_b", disabled=used >= MAX_FORECASTS_PER_SESSION)
        if clicked and used < MAX_FORECASTS_PER_SESSION:
            st.session_state["fc_used"] = used + 1
            try:
                with st.spinner("Training the LSTM (about 10-30 seconds)…"):
                    st.session_state[key] = forecast_client.forecast(series, horizon, epochs)
            except forecast_client.Unavailable as e:
                st.session_state.pop(key, None)
                st.warning(str(e))
        if used >= MAX_FORECASTS_PER_SESSION:
            st.caption("Forecast limit for this session reached. Reload the page to reset.")
        res = st.session_state.get(key)
        if res:
            rows = [{"Model": m, "Test RMSE": round(v["rmse"], 3), "Test MAE": round(v["mae"], 3)}
                    for m, v in res["scores"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config=glossary.column_config(**{"Test RMSE": "rmse", "Test MAE": "mae"}))
            st.info(res["verdict"])
            fig = forecast_client.figure(series, res, f"{var['name']}, {p['place']}", var["units"])
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Test window: last {res['n_test']} months. Lower error is better. "
                       "A forecast extends past patterns; it does not explain them.")


def investigate(p, question=None, series=None, var=None):
    """p = validated params. `series`/`var` are passed for Giovanni CSV case files."""
    want_nb = (series is None and st.session_state.get("do_neighbors")
               and not retrieve.is_giovanni(p["variable"]))  # Giovanni: each point costs the user's quota
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
    res = analyze(s, p["season"])
    m = res["metrics"]
    nb = neighbor_check(p, m["sen_slope_per_decade"], nb_series) if want_nb else None

    st.subheader(f"{var['name']} · {p['place']}")
    st.caption(f"**What is it?** {var['definition']}  ·  Source: {var['source']}")
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(trend_figure(res, var["name"], var["units"], p["season"]))
    with right:
        if m["significant"]:
            st.success(f"**Verdict: significant {m['direction']} trend**")
        elif m["tests_agree"]:
            st.warning("**Verdict: no significant trend (yet)**")
        else:
            st.warning("**Verdict: borderline (the two tests disagree)**")
        c1, c2 = st.columns(2)
        c1.metric("Change per decade", f"{m['sen_slope_per_decade']:+.3g} {var['units']}",
                  help=glossary.tip("change_per_decade"))
        c2.metric("95% range (OLS)", f"{m['ols_ci95_low']:+.2g} to {m['ols_ci95_high']:+.2g}",
                  help=glossary.tip("ci95"))
        c1.metric("Mann-Kendall p", f"{m['mk_p_value']:.3g}", help=glossary.tip("mk_p"))
        c2.metric("t-test p", f"{m['ols_p_value']:.3g}", help=glossary.tip("ols_p"))
        st.caption(f"{m['n_points']} points, {m['start'][:4]}–{m['end'][:4]} · p < 0.05 counts as significant")
        if nb:
            st.info(f"**Nearby check:** {nb['same_direction']} of {nb['n']} surrounding points (~100 km) trend "
                    f"the same way; {nb['significant']} significantly.")
    ctx = dict(question=question, place=p["place"], season=p["season"], metrics=m, neighbors=nb,
               variable={k: var[k] for k in ("name", "units", "definition")}, variable_id=p["variable"])
    text, used_ai = _cached(_explain_cached, ctx)
    if retrieve.is_giovanni(p["variable"]):
        gm = (st.session_state.get("gio_meta") or {}).get(p["variable"], {})
        st.caption(f"Live from NASA Giovanni (`{p['variable'][len(giovanni_catalog.PREFIX):]}`) with your Earthdata "
                   f"token. {gm.get('note', '')} Data: NASA GES DISC; credit NASA when you share results.")
    st.markdown("#### What the detective found")
    st.markdown(text)
    st.caption(("Wording by AI from the computed numbers. " if used_ai else "") +
               "All statistics are calculated in code (SciPy), never by the AI.")

    if series is None:  # next clues (not for Giovanni CSV case files: no live data to follow up with)
        clues, clue_ai = _cached(_clues_cached, ctx)
        if clues:
            st.markdown("#### Next clues")
            for i, c in enumerate(clues):
                cv = ALL[c["variable"]]
                a, b = st.columns([3, 2])
                a.markdown(f"**{cv['name']}**: {c['why']}")
                case = {**{k: p[k] for k in ("place", "lat", "lon", "season", "start_year", "end_year")},
                        "variable": c["variable"], "came_from": var["name"], "question": question}
                b.button(f"Investigate {cv['name']} here", key=f"clue_{p['variable']}_{c['variable']}_{i}",
                         on_click=_push_followup, args=(case,), width="stretch")
            st.caption(("Suggested by AI. " if clue_ai else "") +
                       "These are hypotheses to check, not explanations. Same place, months and years.")
    with st.expander("Case file: exact numbers and data"):
        st.json(m)
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


COMPARE_EXAMPLES = ["Is Svalbard warming faster than London?",
                    "Is the Sahel getting wetter while Cape Town gets drier?",
                    "Is Delhi warming faster than Phoenix in summer?"]


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
    same_units = pa["variable"] == pb["variable"] or (va["units"] == vb["units"] and bool(va["units"]))
    c = compare.compare(sa, sb, pa["season"], same_units)
    la, lb = _label(pa, pb), _label(pb, pa)
    text, kind = compare.headline(c, la, lb)
    if pa["variable"] == pb["variable"]:
        st.subheader(f"{va['name']} · {la} vs {lb}")
    elif pa["place"] == pb["place"]:
        st.subheader(f"{la} vs {lb} · {pa['place']}")
    else:
        st.subheader(f"{la} vs {lb}")
    st.caption(f"{SEASON_LABELS[pa['season']]} · {pa['start_year']}–{pa['end_year']}")
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
        st.markdown("#### How closely do they move together?")
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
    st.markdown("#### What the detective found")
    st.markdown(expl)
    st.caption(("Wording by AI from the computed numbers. " if used_ai else "") +
               "All statistics are calculated in code (SciPy), never by the AI.")
    b1, b2 = st.columns(2)
    for col, side, lab in ((b1, pa, la), (b2, pb, lb)):
        solo = {**{k: side[k] for k in ("variable", "place", "lat", "lon", "season", "start_year", "end_year")},
                "came_from": "the comparison", "origin": "compare", "question": None}
        col.button(f"Open {lab} on its own", key=f"cmp_solo_{side['variable']}_{side['place']}_{lab}",
                   on_click=_push_followup, args=(solo,), width="stretch")
    with st.expander("Case file: exact numbers"):
        st.json({"A": c["a_brief"], "B": c["b_brief"], "comparison": c["summary"]})



# ------------------------------------------------------------------ UI
st.title("🔎 PLaNETS")
st.caption("**Pl**ain-**La**nguage **N**ASA **E**arth **T**rend **S**leuth: ask what is changing, where. "
           "NASA data answers; statistics say whether it's real.")

with st.sidebar:
    mode = st.radio("How do you want to investigate?", ["Ask a question", "Compare two trends", "Pick from menus", "Case files (Giovanni CSV)",
                     "Explore variables"],
                    on_change=_clear_followup)
    with st.expander("🛰️ NASA Giovanni: " + ("connected" if gio_token() else "add your login"),
                     expanded=False):
        st.caption("Adds satellite and model variables (dust, ozone, snow, soil moisture, pollution…) to every mode, "
                   "fetched live with **your own** free Earthdata token. Kept in this browser session only: never "
                   "saved or logged, and sent only to NASA.")
        st.markdown(f"[Get a token]({giovanni_live.TOKEN_URL}) → *Generate Token*, copy, paste here.")
        st.text_input("Earthdata token", type="password", key="gio_token", placeholder="paste token")
        if gio_token():
            if st.button("Check token", key="gio_check"):
                try:
                    q = giovanni_live.check_token(gio_token())
                    st.session_state["gio_status"] = ("ok", f"Token works. {q:.0%} of this month's Giovanni quota used.")
                except giovanni_live.GiovanniError as e:
                    st.session_state["gio_status"] = ("bad", str(e))
            stt = st.session_state.get("gio_status")
            if stt:
                (st.success if stt[0] == "ok" else st.error)(stt[1])
        n_gio = sum(1 for k in ALL if retrieve.is_giovanni(k))
        st.caption(f"{n_gio} Giovanni variables available" +
                   ("" if giovanni_catalog.is_built() else " (starter list; run `giovanni_catalog.py --build` for "
                                                          "NASA's full list)"))
    retrieve.GIOVANNI_ON.set(bool(gio_token()))
    st.checkbox("Compare with nearby points (slower)", key="do_neighbors")
    if HAS_MLFLOW:
        st.checkbox("Log each case to MLflow", key="do_mlflow")
    st.divider()
    st.caption("AI: " + ("connected ✅" if nl.llm_available() else "no API key, using keyword matching + template text"))
    st.caption(f"Catalog: {sum(1 for k in ALL if not retrieve.is_giovanni(k))} NASA POWER variables"
               + (f" + {sum(1 for k in ALL if retrieve.is_giovanni(k))} Giovanni" if gio_token() else ""))
    st.caption("Places: any city over 15,000 people, country or US state, no AI needed" if geo.AVAILABLE else
               "Places: 14 presets only without an AI key. Run `uv sync` to add the offline city list.")
    with st.expander("📖 Glossary: what the numbers mean"):
        for label, text in glossary.TERMS.values():
            st.markdown(f"**{label}**: {text}")
    with st.expander("Featured variables"):
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
        if fu.get("origin") in ("explorer", "compare"):
            b2.caption(f"Opened from **{fu['came_from']}**")
        else:
            b2.caption(f"Following a clue from **{fu['came_from']}** · same place, months and years"
                       + (f" · clue {len(trail)} in this trail" if len(trail) > 1 else ""))
        investigate(fu, question=fu.get("question"))

    elif mode == "Ask a question":
        cols = st.columns(len(DEMO_QUESTIONS))
        for c, dq in zip(cols, DEMO_QUESTIONS):
            if c.button(dq, width="stretch"):
                st.session_state["q"] = dq
        q = st.text_input("Your question", key="q", placeholder="Is Phoenix getting hotter in summer?")
        pair = question_pair(q, bool(gio_token())) if q else None
        one_only = bool(pair) and st.checkbox("Investigate just one variable instead", key=f"ask_one_{q}")
        if pair and not one_only:  # "hotter and drier": test the two variables against each other
            st.info("Your question names more than one variable, so they are tested against each other. "
                    "Tick the box above to investigate just one.")
            st.caption(f"Understood as: **{ALL[pair['a']['variable']]['name']}** at **{pair['a']['place']}** "
                       f"and **{ALL[pair['b']['variable']]['name']}** at **{pair['b']['place']}** · "
                       f"{SEASON_LABELS[pair['season']]} · {pair['start_year']}–{pair['end_year']} · keyword match")
            render_compare({**pair, "question": q}, question=q)
            if pair.get("others"):
                st.caption("Your question names more variables than a comparison can hold (two), so "
                           + ", ".join(ALL[v]["name"] for v in pair["others"]) + " was left out. Ask about it separately.")
        elif q:
            p0, used_ai = ai_parse(q, nl.llm_available(), bool(gio_token()))
            p = dict(p0)
            ov = st.session_state.get("ask_override")
            if ov and ov.get("q") == q:  # a "did you mean" correction for this question
                p = {**nl.validate({**p0, **ov["fields"]}), "candidates": p0["candidates"],
                     "place_alts": p0.get("place_alts", [])}
            st.caption(f"Understood as: **{ALL[p['variable']]['name']}** at **{p['place']}** "
                       f"({p['lat']:.2f}, {p['lon']:.2f}) · {SEASON_LABELS[p['season']]} · "
                       f"{p['start_year']}–{p['end_year']}" + ("" if used_ai else " · keyword match"))
            other_vars = [cid for cid, _ in p0["candidates"] if cid != p["variable"]][:3]
            places = [dict(label=p0["place"], lat=p0["lat"], lon=p0["lon"])] + list(p0.get("place_alts", []))
            other_places = [x for x in places if x["label"] != p["place"]][:3]
            if other_vars or other_places:
                st.caption("Not what you meant? Pick another:")
                btn_cols = st.columns(max(1, len(other_vars) + len(other_places)))
                for col, cid in zip(btn_cols, other_vars):
                    col.button(ALL[cid]["name"], key=f"dym_var_{cid}", width="stretch",
                               on_click=_set_override, args=(q, {"variable": cid}))
                for col, x in zip(btn_cols[len(other_vars):], other_places):
                    col.button(f"📍 {x['label']}", key=f"dym_place_{x['label']}", width="stretch",
                               on_click=_set_override, args=(q, {"place": x["label"], "lat": x["lat"], "lon": x["lon"]}))
            with st.expander(f"How I chose the variable: searched {len(ALL)} catalog entries"):
                for cid, sc in p["candidates"]:
                    st.markdown(("✅ " if cid == p["variable"] else "· ") +
                                f"**{ALL[cid]['name']}** (`{cid}`), match score {sc}")
            if nl.looks_like_comparison(q):
                st.info("This looks like a comparison. 'Compare two trends' in the sidebar tests two places or "
                        "variables against each other directly.")
            st.map(pd.DataFrame({"lat": [p["lat"]], "lon": [p["lon"]]}), zoom=3, height=220)
            investigate(p, question=q)

    elif mode == "Compare two trends":
        st.caption("Compare one variable in two places, or two variables in one place.")
        cols = st.columns(len(COMPARE_EXAMPLES))
        for col, ex in zip(cols, COMPARE_EXAMPLES):
            if col.button(ex, width="stretch", key=f"cmp_ex_{ex}"):
                st.session_state["cmp_q"] = ex
        cq = st.text_input("Your comparison question", key="cmp_q",
                           placeholder="Is Svalbard warming faster than London?")
        if cq and cq != st.session_state.get("cmp_q_done"):
            st.session_state["cmp_q_done"] = cq
            st.session_state.pop("cmp_case", None)
            case, used_ai = ai_parse_compare(cq, nl.llm_available(), bool(gio_token()))
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
        cases = giovanni.case_files()
        shown = explorer.filter_table(df)
        st.subheader("What can I investigate?")
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
        src = d1.selectbox("Source", ["All sources", "NASA POWER", "NASA Giovanni"], key="ex_src")
        hourly = d2.checkbox("Include hour-by-hour POWER variants (about 1,100 near-duplicates)", key="ex_hourly")
        ranked = [c for c, _ in search(text, k=60, giovanni=True)] if text else None
        view = explorer.filter_table(df, group=group, text=text, show_hourly=hourly, ranked_codes=ranked or None,
                                     source=None if src == "All sources" else src)
        if src != "NASA POWER" and not gio_token():
            st.info("Giovanni variables can be browsed here; to investigate one, add your Earthdata token under "
                    "'NASA Giovanni' in the sidebar.")
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
            elif code not in VARIABLES:
                st.caption("From the full NASA POWER catalog: not every variable has monthly data at every place.")
            a, b, c = st.columns(3)
            pick = place_picker(a, "Place", key="ex_place")
            season = b.selectbox("Months", list(SEASON_LABELS), format_func=SEASON_LABELS.get, key="ex_season")
            if pick:
                case = {**nl.validate(dict(variable=code, place=pick[0], lat=pick[1], lon=pick[2], season=season)),
                        "came_from": "the variable explorer", "origin": "explorer", "question": None}
                needs_tok = retrieve.is_giovanni(code) and not gio_token()
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
                f"- {geo.ATTRIBUTION}. A country or state is represented by its largest city.")

    else:
        cases = giovanni.case_files()
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
            for r in giovanni.RECIPES:
                st.markdown(f"- **[{r['title']}]({r['url']})**: {r['why']} "
                            f"If the variable isn't preselected, search Giovanni for `{r['keyword']}`.")
except Exception as e:
    st.error(str(e))
