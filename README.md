# PLaNETS: Plain-Language NASA Earth Trend Sleuth

Ask a plain-language question about environmental change ("Is Svalbard getting warmer?", "Is Oslo warming faster than Cairo in winter?") and get a statistically tested answer with a plain-language explanation.

Built for the NASA Space Apps Challenge, *Be an Earth Trend Detective*.

## How it works
1. **Understand the question.** Retrieval over NASA POWER's parameter list plus an offline place index; an LLM (optional) picks among the retrieved candidates. Without an LLM, hybrid keyword + embedding search (with a hand-written phrase lexicon in `lexicon.py`) picks the variable, and a question that names two variables ("hotter and drier in Texas") is tested as a comparison of the two. Check retrieval quality with `python eval_retrieval.py`; after editing `lexicon.py` or the catalog run `python retrieve.py --build-embeddings`.
2. **Get the data.** Monthly NASA POWER data for the place (live), or a NASA Giovanni time-series CSV (case files).
3. **Test the trend in code.** Seasonal cycle removed; OLS slope per decade with 95% CI and p-value; Mann-Kendall and Sen's slope; lag-1 autocorrelation caveat. A trend is called significant only when both tests agree.
4. **Explain.** The AI only writes the explanation from numbers the code computed. Without a key, template text is used.

All statistics are computed by deterministic code, never by the model.

## Run locally
    python -m uv sync
    python -m uv run streamlit run app.py

Copy keys into `.env` (optional; the app works without them).

## Deploy (Streamlit Community Cloud)
Push to GitHub, create the app with main file `app.py`, and paste `.streamlit/secrets.toml.example` into Settings -> Secrets with a real key. Set a spend limit on the key. `AI_DAILY_CAP` limits AI calls per day.

## Data and credits
- **NASA POWER** (Prediction Of Worldwide Energy Resources), https://power.larc.nasa.gov, NASA Langley Research Center. Data are open; please credit NASA.
- **NASA GES DISC Giovanni**, https://giovanni.gsfc.nasa.gov, for the case-file time series.
- **GeoNames** city, country and region data (via the `geonamescache` package), https://www.geonames.org, licensed CC BY 4.0.
- `data/demo_ozone_SYNTHETIC.csv` is synthetic and labeled as such in the app.

## Limits
Single grid points (about 50 km) stand in for places; countries and states use their largest city. Monthly records are short for slow climate signals, and significance does not imply cause.

## NASA Giovanni (optional, your own login)
Add a free Earthdata token under "NASA Giovanni" in the sidebar and Giovanni's satellite and model variables (dust, ozone, snow, soil moisture, pollution...) join NASA POWER in every mode: questions, comparisons, menus and the variable explorer. Data come live from NASA's Giovanni in the Cloud time series service. The token stays in the browser session only: never stored, cached, or logged, and sent only to NASA. Data: NASA GES DISC; credit NASA when sharing results.

Variable list: `python -m uv run python giovanni_catalog.py --build` downloads NASA's list of configured variables into `giovanni_vars.json` (commit it). Without it a small starter list is used.
