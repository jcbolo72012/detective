# The Planeteer (PLaNETS: Plain-Language NASA Earth Trend Sleuthing)

Ask a plain-language question about environmental change ("Is Svalbard getting warmer?", "Is Oslo warming faster than Cairo in winter?") and get a statistically tested answer with a plain-language explanation.

Built for the NASA Space Apps Challenge, *Be an Earth Trend Detective*.

## How it works
1. **Understand the question.** Retrieval over NASA POWER's parameter list plus an offline place index; an LLM (optional) picks among the retrieved candidates. Without an LLM, hybrid keyword + embedding search (with a hand-written phrase lexicon in `planets/query/lexicon.py`) picks the variable, and a question that names two variables ("hotter and drier in Texas") is tested as a comparison of the two. Check retrieval quality with `python eval/eval_retrieval.py`; after editing `planets/query/lexicon.py` or the catalog run `python -m planets.query.retrieve --build-embeddings`.
2. **Get the data.** Monthly NASA POWER data for the place (live), or a NASA Giovanni time-series CSV (case files).
3. **Test the trend in code.** Seasonal cycle removed; OLS slope per decade with 95% CI and p-value; Mann-Kendall and Sen's slope; lag-1 autocorrelation caveat. A trend is called significant only when both tests agree.
4. **Explain.** The AI only writes the explanation from numbers the code computed. Without a key, template text is used.

All statistics are computed by deterministic code, never by the model.

## Project layout
```
app.py                    Streamlit app (entry point)
planets/
  config.py               paths and .env settings, used everywhere
  analysis/               trend tests (trend.py), comparison, correlation, years until significant
  forecast/               NumPy LSTM and the backtested forecast
  sources/                NASA POWER, Giovanni (live + case files + catalog), PO.DAAC, provider routing
  query/                  question -> variable and place: catalog, lexicon, hybrid search, places, AI/keyword parsing
  ui/                     glossary, share links, variable explorer table, "see it from space"
data/                     Giovanni case-file CSVs; data/catalog/ holds catalogs and embedding vectors
eval/                     retrieval test questions and eval_retrieval.py
tests/                    python tests/test_forecast_lstm.py, python tests/test_emergence.py
docs/                     notes: adding data sources, retrieval metrics, global LSTM spec
```

## Run locally
    python -m uv sync
    python -m uv run streamlit run app.py

Copy keys into `.env` (optional; the app works without them).

Visitors can also paste **their own Claude key** under "Use your own Claude key" in the sidebar. It is kept in that browser session only (`nl.USER_API_KEY`, a per-session context variable: never in `os.environ`, a cache key, a log or an error message), is billed to the visitor, and bypasses the shared `AI_DAILY_CAP`. If it fails, the app falls back to keyword matching and template text.

## Deploy (Streamlit Community Cloud)
Push to GitHub, create the app with main file `app.py`, and paste `.streamlit/secrets.toml.example` into Settings -> Secrets with a real key. Set a spend limit on the key. `AI_DAILY_CAP` limits AI calls per day.

## Data and credits
- **NASA POWER** (Prediction Of Worldwide Energy Resources), https://power.larc.nasa.gov, NASA Langley Research Center. Data are open; please credit NASA.
- **NASA GES DISC Giovanni**, https://giovanni.gsfc.nasa.gov, for the case-file time series.
- **NASA PO.DAAC**, https://podaac.jpl.nasa.gov, for ocean variables: sea surface temperature (NOAA ERSST v5, distributed by PO.DAAC) and sea level anomaly (NASA-SSH). Needs your Earthdata token.
- **NASA GIBS / Worldview**, https://worldview.earthdata.nasa.gov, for the "See it from space" images and links. No login.
- **GeoNames** city, country and region data (via the `geonamescache` package), https://www.geonames.org, licensed CC BY 4.0.
- `data/demo_ozone_SYNTHETIC.csv` is synthetic and labeled as such in the app.

## Limits
Single grid points (about 50 km) stand in for places; countries and states use their largest city. Monthly records are short for slow climate signals, and significance does not imply cause.

## NASA Giovanni (optional, your own login)
Add a free Earthdata token under "NASA Giovanni" in the sidebar and Giovanni's satellite and model variables (dust, ozone, snow, soil moisture, pollution...) join NASA POWER in every mode: questions, comparisons, menus and the variable explorer. Data come live from NASA's Giovanni in the Cloud time series service. The token stays in the browser session only: never stored, cached, or logged, and sent only to NASA. Data: NASA GES DISC; credit NASA when sharing results.

Variable list: `python -m uv run python -m planets.sources.giovanni_catalog --build` downloads NASA's list of configured variables into `data/catalog/giovanni_vars.json` (commit it). Without it a small starter list is used.

The same Earthdata token also unlocks two ocean variables from **NASA PO.DAAC** (`planets/sources/podaac.py`): sea surface temperature and sea level. They are read straight from PO.DAAC's cloud OPeNDAP server, one small request per month of record, so the first look at a place takes about a minute (20 years by default) and is then cached for a day. They are ocean-only: a coastal place uses the nearest open-ocean grid cell, and says so. Sea level is one 5-day satellite map per month (nearest the 15th), not a monthly mean.

Each investigation also has a **See it from space** panel (`planets/ui/space_view.py`): a NASA Worldview link and first-year vs last-year GIBS images for the place. These need no login and are context only, not evidence for the trend.
