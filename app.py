"""Earth Trend Detective - interactive app.
Run:   pip install streamlit pandas numpy scipy matplotlib requests   (+ anthropic or openai, + mlflow: optional)
       streamlit run app.py
Flow:  question -> [AI or keywords] parameters -> NASA POWER / Giovanni CSV -> analyze() in code -> [AI or template] explanation
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import nl
import power
from catalog import DEMO_QUESTIONS, PLACES, VARIABLES
from retrieve import ALL
from trend_detective import analyze, load_giovanni_csv

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

st.set_page_config(page_title="Earth Trend Detective", page_icon="🔎", layout="wide")
SEASON_LABELS = {None: "All months (seasonal cycle removed)", "DJF": "Dec–Feb", "MAM": "Mar–May",
                 "JJA": "Jun–Aug", "SON": "Sep–Nov"}


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_series(param, lat, lon, start, end):
    return power.fetch_point(param, lat, lon, start, end)


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


def neighbor_check(p, own_slope):
    out = []
    for la, lo in power.neighbor_points(p["lat"], p["lon"]):
        try:
            m = analyze(get_series(p["variable"], la, lo, p["start_year"], p["end_year"]), p["season"])["metrics"]
            out.append(m)
        except Exception:
            pass
    if not out:
        return None
    return dict(n=len(out), same_direction=sum((m["sen_slope_per_decade"] > 0) == (own_slope > 0) for m in out),
                significant=sum(m["significant"] for m in out),
                slopes_per_decade=[round(m["sen_slope_per_decade"], 4) for m in out])


def investigate(p, question=None, series=None, var=None):
    """p = validated params. `series`/`var` are passed for Giovanni CSV case files."""
    var = var or ALL[p["variable"]]
    with st.spinner("Fetching NASA data…"):
        s = series if series is not None else get_series(p["variable"], p["lat"], p["lon"], p["start_year"], p["end_year"])
    res = analyze(s, p["season"])
    m = res["metrics"]
    nb = None
    if series is None and st.session_state.get("do_neighbors"):
        with st.spinner("Checking 4 nearby points…"):
            nb = neighbor_check(p, m["sen_slope_per_decade"])

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
        c1.metric("Change per decade", f"{m['sen_slope_per_decade']:+.3g} {var['units']}")
        c2.metric("95% range (OLS)", f"{m['ols_ci95_low']:+.2g} to {m['ols_ci95_high']:+.2g}")
        c1.metric("Mann-Kendall p", f"{m['mk_p_value']:.3g}")
        c2.metric("t-test p", f"{m['ols_p_value']:.3g}")
        st.caption(f"{m['n_points']} points, {m['start'][:4]}–{m['end'][:4]} · p < 0.05 counts as significant")
        if nb:
            st.info(f"**Nearby check:** {nb['same_direction']} of {nb['n']} surrounding points (~100 km) trend "
                    f"the same way; {nb['significant']} significantly.")
    ctx = dict(question=question, place=p["place"], season=p["season"], metrics=m, neighbors=nb,
               variable={k: var[k] for k in ("name", "units", "definition")})
    text, used_ai = nl.explain(ctx)
    st.markdown("#### What the detective found")
    st.markdown(text)
    st.caption(("Wording by AI from the computed numbers. " if used_ai else "") +
               "All statistics are calculated in code (SciPy), never by the AI.")
    with st.expander("Case file: exact numbers and data"):
        st.json(m)
        st.download_button("Download series (CSV)", res["series"].to_csv(), "series.csv")

    if HAS_MLFLOW and st.session_state.get("do_mlflow"):
        mlflow.set_experiment("earth-trend-detective")
        with mlflow.start_run(run_name=f"{var['name']} · {p['place']}"):
            mlflow.log_params({k: p[k] for k in ("variable", "place", "lat", "lon", "season", "start_year", "end_year")})
            mlflow.log_metrics({k: float(v) for k, v in m.items() if isinstance(v, (int, float, bool))})


# ------------------------------------------------------------------ UI
st.title("🔎 Earth Trend Detective")
st.caption("Ask what is changing, where. NASA data answers; statistics say whether it's real.")

with st.sidebar:
    mode = st.radio("How do you want to investigate?", ["Ask a question", "Pick from menus", "Case files (Giovanni CSV)"])
    st.checkbox("Compare with nearby points (slower)", key="do_neighbors")
    if HAS_MLFLOW:
        st.checkbox("Log each case to MLflow", key="do_mlflow")
    st.divider()
    st.caption("AI: " + ("connected ✅" if nl.llm_available() else "no API key, using keyword matching + template text"))
    st.caption(f"Catalog: {len(ALL)} variables searchable")
    with st.expander("Featured variables"):
        for v in VARIABLES.values():
            st.markdown(f"**{v['name']}** ({v['units']}): {v['definition']}")

try:
    if mode == "Ask a question":
        cols = st.columns(len(DEMO_QUESTIONS))
        for c, dq in zip(cols, DEMO_QUESTIONS):
            if c.button(dq, use_container_width=True):
                st.session_state["q"] = dq
        q = st.text_input("Your question", key="q", placeholder="Is Phoenix getting hotter in summer?")
        if q:
            p, used_ai = nl.parse_question(q)
            st.caption(f"Understood as: **{ALL[p['variable']]['name']}** at **{p['place']}** "
                       f"({p['lat']:.2f}, {p['lon']:.2f}) · {SEASON_LABELS[p['season']]} · "
                       f"{p['start_year']}–{p['end_year']}" + ("" if used_ai else " · keyword match"))
            with st.expander(f"How I chose the variable: searched {len(ALL)} catalog entries"):
                for cid, sc in p["candidates"]:
                    st.markdown(("✅ " if cid == p["variable"] else "· ") +
                                f"**{ALL[cid]['name']}** (`{cid}`), match score {sc}")
            st.map(pd.DataFrame({"lat": [p["lat"]], "lon": [p["lon"]]}), zoom=3, height=220)
            investigate(p, question=q)

    elif mode == "Pick from menus":
        c1, c2, c3 = st.columns(3)
        vid = c1.selectbox("Variable", list(VARIABLES), format_func=lambda k: VARIABLES[k]["name"])
        place = c2.selectbox("Place", list(PLACES) + ["Custom lat/lon"])
        season = c3.selectbox("Months", list(SEASON_LABELS), format_func=SEASON_LABELS.get)
        if place == "Custom lat/lon":
            lat = c1.number_input("Latitude", -90.0, 90.0, 40.0)
            lon = c2.number_input("Longitude", -180.0, 180.0, -100.0)
            place = f"{lat:.2f}, {lon:.2f}"
        else:
            lat, lon = PLACES[place]
        y0, y1 = st.slider("Years", 1981, power.LAST_FULL_YEAR, (1981, power.LAST_FULL_YEAR))
        if st.button("Investigate", type="primary"):
            investigate(nl.validate(dict(variable=vid, place=place, lat=lat, lon=lon, season=season,
                                         start_year=y0, end_year=y1)))

    else:
        files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.csv")))
        if not files:
            st.info("Drop Giovanni 'Time Series, Area-Averaged' CSVs into the `data/` folder and they appear here.")
        else:
            f = st.selectbox("Case file", files, format_func=os.path.basename)
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Variable name", os.path.splitext(os.path.basename(f))[0].replace("_", " "))
            units = c2.text_input("Units", "units")
            season = c3.selectbox("Months", list(SEASON_LABELS), format_func=SEASON_LABELS.get)
            definition = st.text_input("Plain-language definition", "")
            if st.button("Investigate", type="primary"):
                var = dict(name=name, units=units, definition=definition or "—", source="NASA Giovanni (GES DISC)")
                investigate(dict(variable=name, place="Giovanni region", lat=0, lon=0, season=season,
                                 start_year=0, end_year=0), series=load_giovanni_csv(f), var=var)
except Exception as e:
    st.error(str(e))
