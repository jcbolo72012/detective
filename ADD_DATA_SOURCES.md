# Instructions: add four data sources to PLaNETS

> **Status (executed Nov 14, 2026):** Phase 0, Phase 1 (`space_view.py`) and Phase 3 (`podaac.py`) are done and tested
> live. **Phase 2 (AppEEARS) is blocked:** AppEEARS returns 403 for a valid Earthdata *token* (only username/password
> login is documented), LP DAAC's MODIS monthly collections have no Harmony or OPeNDAP subsetting, and the ORNL DAAC
> MODIS API is too slow (60 s per metadata call) and has no monthly NDVI. Options for a decision: AppEEARS with a
> server-side Earthdata username/password secret, or another vegetation source. **Phase 4 (FIRMS) was deliberately
> skipped.** Phase 3 deviates from the plan: no Harmony; it reads PO.DAAC's Hyrax OPeNDAP (DAP4 `.dap.nc4`) one tiny
> request per month, because Harmony jobs took about 45 s per year of 5-day granules.

Hand this file to the agent doing the work. It adds, in this order:

1. **Worldview deep links + GIBS image previews** (NASA imagery, no login)
2. **LP DAAC via AppEEARS** (vegetation greenness and land surface temperature time series, Earthdata token)
3. **PO.DAAC** (sea surface temperature and sea level time series, Earthdata token)
4. **FIRMS wildfire detections** (free `MAP_KEY`) **LOWER PRIORITY. Do this last, and only if 1 to 3 are finished and stable.**

Nothing here has been run against the live services. Endpoint names, parameters, product ids and auth behaviour below
come from documentation and memory. **Start every section with a small throwaway smoke test (a short script, not
committed) that proves the call works with a real credential, and trust what it returns over this file.**

## Ground rules (read first)

- **Another agent may be editing this repo.** Before touching a shared file (`app.py`, `retrieve.py`, `nl.py`,
  `catalog.py`, `README.md`), look at its current contents and make small, localized edits. Put new logic in **new
  modules** and keep the edits to shared files to a few lines each. Do not reformat or reorganize existing code.
- Use the project venv: `.\.venv\Scripts\python.exe` (the system Python lacks `geonamescache`). Add dependencies with
  `python -m uv add <pkg>`, which updates `pyproject.toml` and `uv.lock`; also add the package to `requirements.txt`
  (Streamlit Cloud reads it).
- **Credentials:** Earthdata token = the one the user already pastes in the sidebar (`gio_token()` in `app.py`, stored
  in `st.session_state["gio_token"]`). Reuse it; do not add a second login. Never log, cache-key, print or persist a
  token or key. Follow the existing pattern: an `_token` parameter whose name starts with an underscore keeps
  `st.cache_data` from hashing it (see `_giovanni_series` in `app.py`). Redact keys from exception text (FIRMS puts the
  key in the URL).
- **Threads:** fetch functions run in worker threads via `run_parallel` / `get_many`. They must not call `st.*` or
  touch `st.session_state`.
- **All statistics stay in code.** New sources only supply a monthly `pandas.Series` (DatetimeIndex at month starts,
  float values, NaNs dropped); `trend_detective.analyze()` does the rest. Match what `power.fetch_point` returns.
- **Catalog entries** use the shape produced by `giovanni_catalog.make_entry`: `name, short_name, units, definition,
  source, keywords, provider, step, first_year, last_year, region` (plus source-specific ids). Keys get a source
  prefix so they never clash (`GIO:` exists; use `LPD:`, `POD:`, `FIRMS:`).
- After adding or changing catalog entries or `lexicon.py`: `python retrieve.py --build-embeddings` and commit the
  rebuilt `catalog_embeddings.npz`. Then run `python eval_retrieval.py` and `python eval_retrieval.py --heldout`; scores
  must not drop (see "Retrieval hygiene").
- Credit NASA in the UI caption for every new source, and add a line to the README "Data and credits" section.
- Test through the real app: `streamlit.testing.v1.AppTest` (example at the end).

## Phase 0: small shared groundwork (do before the sources)

The code assumes two providers: POWER (no prefix) and Giovanni (`GIO:` prefix, `retrieve.is_giovanni`). Generalize
with the smallest change that works.

1. **Provider lookup.** Add `retrieve.provider_of(code)` returning `"power" | "giovanni" | "appeears" | "podaac" |
   "firms"` from the prefix, and keep `is_giovanni` as is. Put prefixes in one place (a small `providers.py` is fine).
2. **Credentials gate in ranking.** `retrieve._weight` returns 0 for Giovanni entries when there is no token. Give the
   new Earthdata-token sources the same behavior (0 without a token), and FIRMS 0 without a key. Keep the existing
   weights for POWER and Giovanni unchanged. New sources should rank slightly below POWER (about 0.7, like Giovanni) so
   free, login-free POWER wins ties.
3. **Fetch dispatch.** `app._fetch` currently branches on `retrieve.is_giovanni`. Add a branch per new provider that
   calls the new module and returns `(series, meta)` the same way. Keep the friendly "needs your Earthdata token" error.
4. **Menus.** `app.usable_codes()` appends Giovanni entries once a token exists; append the new token-gated entries the
   same way (FIRMS only when a key is available). Update `explorer.py`'s "Source" column if it assumes two sources.
5. **Validation.** `nl.validate` clamps years per provider (`provider == "giovanni"` branch). Make new providers use
   their own `first_year` / `last_year` the same way, with a sensible default window. The locked-variable hint
   (`nl._giovanni_hint`) should name whichever login the variable needs.
6. **Async job helper** (needed by AppEEARS and PO.DAAC): a tiny function that submits a job, polls with backoff, and
   returns the result, with a hard timeout (about 3 minutes) and a clear error. It must work in a worker thread (no `st`
   calls). A job that finishes should be cached (`st.cache_data`, ttl 24 h) by (source, variable, lat, lon, start, end).

Acceptance: with no new source enabled, behavior and `eval_retrieval.py` scores are identical to before.

## Phase 1: Worldview deep links and GIBS previews (no login)

**Goal:** next to each investigation, a "See it from space" panel: a link to NASA Worldview centred on the place with
a layer matching the variable, and two GIBS images of the place (first year vs last year, same season).

**Auth:** none. **Effort:** small. **Do not** add a map library; links and `st.image` are enough.

**New module `space_view.py`:**
- `LAYER_FOR`: a dict from catalog id (or a variable family, such as temperature, precipitation, snow, vegetation,
  fire, aerosol, sea ice, sea surface temperature) to a GIBS layer id, with a default of MODIS Terra true colour.
  **Look the layer ids up in GIBS GetCapabilities** (`https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities`);
  record each layer's start date so you never ask for a date it does not cover. Likely candidates (verify them):
  `MODIS_Terra_CorrectedReflectance_TrueColor`, `VIIRS_SNPP_CorrectedReflectance_TrueColor`,
  `MODIS_Terra_Land_Surface_Temp_Day`, `IMERG_Precipitation_Rate`, `MODIS_Terra_NDSI_Snow_Cover`,
  `MODIS_Terra_NDVI_8Day`, `MODIS_Terra_Aerosol`, `AMSRU2_Sea_Ice_Concentration_12km`,
  `VIIRS_SNPP_Thermal_Anomalies_375m_All`, plus the sea surface temperature layer from Phase 3.
- `worldview_url(lat, lon, date, layer_ids)`: Worldview takes its state in the query string (`v` extent as
  `west,south,east,north`, `t` time, `l` comma-separated layers, and a way to drop a place marker). **Build one by hand
  in a browser, copy the resulting URL, and match its exact parameter names**; do not guess them. Include reference
  layers (coastlines, borders) so the image is readable.
- `gibs_image_url(layer_id, lat, lon, date, half_width_deg=2.0, size=512)`: a WMS `GetMap` URL on
  `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi` (EPSG:4326, `FORMAT=image/jpeg`, `TIME=YYYY-MM-DD`).
  Keep the bounding box square in pixels (widen longitude by `cos(lat)`), and clamp lat/lon to valid ranges.
- Dates: use the middle of the season in the first and last full year of the analysis (for "all months" use July 15
  in the northern hemisphere and January 15 in the southern). If the layer does not exist that far back, show only the
  most recent image.

**UI (small insertion in `app.py`, in `investigate()` after the metrics block):** an `st.expander("See it from
space")` with two `st.image` columns (first year, last year), the Worldview link as `st.link_button`, and a caption
crediting NASA GIBS / Worldview. Images that fail to load must not break the page (GIBS can return a blank tile for
cloudy or missing days); show a short note instead.

**Pitfalls:** the images are satellite pictures, not the statistic. Caption them as context ("what this place looked
like in 2003 vs 2024"), never as evidence of the trend. Do not fetch images server-side on every rerun; pass the URL
to `st.image`.

**Acceptance:** for three places (an Arctic, a tropical and a desert place) the expander shows two images and a
working Worldview link; a layer with a late start date falls back to one image; no exceptions in `AppTest`.

## Phase 2: LP DAAC via AppEEARS (vegetation and land surface temperature)

**Goal:** new variables the catalog lacks: **vegetation greenness (NDVI, EVI)** and **land surface temperature**, as
monthly point series, so questions like "is the Sahel getting greener?" work.

**Auth:** Earthdata Login. AppEEARS is at `https://appeears.earthdatacloud.nasa.gov/api/`. Its documented login is a
`POST /login` with the user's username and password (Basic auth), which returns a bearer token. **Do not ask users for
their password.** First check whether AppEEARS accepts the user's existing Earthdata *token* as
`Authorization: Bearer <token>`. If it does, use that. If it does not, stop and report back rather than collecting
passwords; the fallback is to skip AppEEARS and read the monthly NDVI product through Harmony or OPeNDAP as in Phase 3.

**Why AppEEARS:** it samples a product at a lat/lon for a date range and returns a small CSV, so you never download
global rasters. Jobs are asynchronous (seconds to minutes).

**Flow (verify each against the AppEEARS API docs):**
1. `POST /task` with `task_type: "point"`, `params` containing `dates` (MM-DD-YYYY), `layers`
   (`[{"product": "...", "layer": "..."}]`) and `coordinates` (`[{"latitude":..,"longitude":..,"id":"p1","category":"x"}]`).
2. Poll `GET /status/{task_id}` until `done` (use the Phase 0 helper; fail fast on `error`).
3. `GET /bundle/{task_id}`, find the results CSV in the file list, download it with `GET /bundle/{task_id}/{file_id}`.
4. Parse to a monthly `Series`; apply the scale factor and drop fill values **yourself if AppEEARS did not**.
   Verify by sampling a place you know (a dense forest should give NDVI around 0.7 to 0.9, a desert around 0.05).

**Start with these products (confirm ids and layer names in the AppEEARS product list):**

| Catalog id | Product and layer | Notes |
|---|---|---|
| `LPD:NDVI` | `MOD13A3.061` monthly 1 km, NDVI layer | record from 2000; a Terra-only record is more consistent |
| `LPD:EVI` | same product, EVI layer | |
| `LPD:LST_DAY` | `MOD11C3.061` monthly 0.05 degree, daytime land surface temperature | Kelvin: convert to °C like `giovanni_catalog.to_celsius_if_kelvin` |

Use the product's quality layer where one exists and drop low-quality months rather than interpolating.

**Integration:**
- New module `appeears.py` with `fetch(product, layer, lat, lon, start_year, end_year, token) -> pd.Series`, a
  catalog builder (a small static dict of the entries above is fine, no remote catalog needed) and `load()` returning
  entries with `provider: "appeears"`, `first_year: 2000`, `step: "monthly"`.
- Merge them in `retrieve.load_all()` the same way `giovanni_catalog.load()` is merged, with the `LPD:` prefix.
- Wire `app._fetch` per Phase 0. Because a job takes a while, make the spinner text say so
  ("Asking NASA AppEEARS: this can take a minute"), and for **neighbour checks** (`want_nb`, 5 points) do not fire 5
  jobs automatically for these variables; disable the neighbour option for AppEEARS variables, as Giovanni already
  does for cost (see the `not retrieve.is_giovanni(...)` condition near the neighbour checkbox in `investigate`).
- Add `lexicon.py` entries for each id: greener, greening, browning, vegetation, plant growth, forest, crops healthy,
  "leaf cover", "greenness", "land surface temperature", "ground heat", "urban heat". Then rebuild embeddings.
- Add `LAST_FULL_YEAR`-aware limits (the product lags a few weeks; ask for completed years only).

**Pitfalls:** the first sample point is the 1 km cell containing the (often city-centre) coordinate, so large countries
are represented by their largest city, exactly as for the other sources; say so in the caption. NDVI saturates over
dense forest and has a strong seasonal cycle, which `analyze()` removes, but state that cloud screening varies by
season. Never present NDVI change as proof of cause.

**Acceptance:** "Is the Sahel getting greener?" and "Is Phoenix land surface temperature rising?" return a verdict
with a plausible series (no all-NaN, no scale-factor values like 3000), with a token; without a token the variable is
hidden from search and the locked-variable hint names the token. Cached repeat requests return instantly.

## Phase 3: PO.DAAC (sea surface temperature, sea level)

**Goal:** ocean variables for coastal and ocean points: **sea surface temperature** and **sea surface height**.

**Auth:** Earthdata Login token (same as above). Data are served through the cloud with Harmony (subsetting service)
and OPeNDAP, both of which accept the user's bearer token.

**Pick the access route by testing, in this order:**
1. **Harmony** (`https://harmony.earthdata.nasa.gov`, python client `harmony-py`): request a spatial box around the
   point and a time range for one collection and variable; it returns a small NetCDF/CSV. Needs the collection's
   concept id (find it with the CMR collection search, `https://cmr.earthdata.nasa.gov/search/collections.json?short_name=...`,
   which needs no login) and that the collection has a Harmony-capable subsetting service. Jobs are async: use the
   Phase 0 helper.
2. **OPeNDAP via `xarray` + `pydap`/`netCDF4`** with the token, if Harmony cannot subset the collection. Open one
   granule per time step lazily and slice to the point. Only practical for products with few, large granules.

Do **not** loop over thousands of daily files. Choose products with a **monthly** or at most 5-day cadence.

**Start with these (confirm short names and concept ids in CMR; confirm lat/lon grid and time range):**

| Catalog id | Candidate product | Notes |
|---|---|---|
| `POD:SST` | A monthly L3/L4 sea surface temperature product (for example a monthly GHRSST or MODIS Aqua SST product); if only daily products fit, average to months in code | Kelvin to °C; ocean points only |
| `POD:SSH` | A gridded sea surface height product (for example NASA-SSH or the JPL gridded SSH, roughly 5-day) averaged to monthly means | metres; ocean points only |

If a candidate has no subsetting route, pick another from the same family and record the decision in a short comment at
the top of `podaac.py`. Report any product you could not use and why.

**Integration:** new module `podaac.py` mirroring `appeears.py` (`fetch(...) -> pd.Series`, a static catalog dict,
`provider: "podaac"`, prefix `POD:`). Land points: when the sample cell is land (all-NaN), raise a friendly
`ValueError` ("This is an ocean variable and this place is on land; try a coastal city or an ocean point") rather than
a stack trace. Consider offering the nearest ocean cell within about 100 km as an explicit option, never silently.
Add GIBS layer ids for these variables to `space_view.LAYER_FOR` from Phase 1.

**Lexicon:** sea surface temperature, SST, ocean temperature, warming oceans, marine heatwave, sea level, sea level
rise, rising seas, tides, ocean height, coral bleaching.

**Pitfalls:** SST and sea level have trends and strong interannual swings (El Nino); the existing short-record caveats
apply. Sea surface height from satellite altimetry is not valid right at the coast or under sea ice.

**Acceptance:** "Is the sea level rising near Honolulu?" and "Is the ocean warming off Perth?" return plausible series
(SST in the right °C range for the place, sea level rising on the order of millimetres per year) and a clear message
for an inland place.

## Phase 4: FIRMS wildfire detections (LOWER PRIORITY, do last)

**Do this only after phases 1 to 3 are working and merged.** It is a new kind of question ("more fires near here?"), but
it is the most work for the least certain statistical value: a long monthly history needs many small API calls, and
fire counts are noisy, zero-heavy and sensor-dependent. If time is short, skip it or ship only the "recent fires map
link" via Worldview (Phase 1 already has the fire layer).

**Auth:** a free `MAP_KEY` (requested with an email address at `https://firms.modaps.eosdis.nasa.gov/api/map_key`).
Treat it as a server secret read from an environment variable `FIRMS_MAP_KEY` (Streamlit secrets, same mechanism as
`ANTHROPIC_API_KEY`; add a commented example line to `.streamlit/secrets.toml.example`). Optionally let a user paste
their own key in the sidebar, handled like the Earthdata token. The key is part of the request URL, so redact it from
every error message and never log URLs.

**API (verified in the docs index; confirm parameters):**
`https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{START_DATE}`
with `SOURCE` of `MODIS_SP` (standard processing, from late 2000) or `VIIRS_SNPP_SP` (from 2012), `DAY_RANGE` at most
10, and the `data_availability` endpoint to learn the date coverage. Check the documented rate limit (a per-key
transaction cap per 10 minutes) and stay well under it.

**Design:**
- Sample a box around the place (about 0.5 degree each side, or a radius filter in code with haversine) and count
  detections per month, optionally also total fire radiative power. **Treat months with data and zero detections as real
  zeros** (do not drop them), but never invent months the sensor did not cover.
- A full record is hundreds of 10-day calls. Default the window to the last 10 years, fetch in parallel with a modest
  worker count, cache aggressively (`st.cache_data`, 24 h+), show progress text, and cap total calls.
- Do not splice MODIS and VIIRS into one series; they differ in sensitivity. Use one sensor per analysis and name it.
- `analyze()` assumes a mostly continuous series. For places with very few fire months, return a friendly "too few fires
  here to test a trend" instead of a verdict.
- New module `firms.py` (`fetch(lat, lon, start_year, end_year, key) -> pd.Series`), catalog entries with prefix `FIRMS:`
  (`FIRMS:COUNT_MODIS` "Fire detections per month, MODIS"; optionally `FIRMS:FRP_MODIS`), `provider: "firms"`,
  `first_year: 2001`. Gate ranking on a key being available.
- Lexicon: wildfire, fires, burning, burn season, fire season, blaze, smoke, hotspots, "more fires".
- Add a glossary note: a detection is a satellite hotspot, not a measure of area burned.

**Pitfalls:** detection counts depend on overpass times and cloud, so a change in count is a change in *detections*;
say that in the explanation caption. Never claim a cause.

**Acceptance:** with a key, "Are there more fires in California?" returns a verdict or the "too few fires" message in a
reasonable time (under about a minute for a 10-year window, then instant when cached); with no key the variable is
hidden and the hint says how to get one.

## Retrieval hygiene (after any catalog or lexicon change)

1. `python retrieve.py --build-embeddings`, commit `catalog_embeddings.npz`.
2. Add 3 to 5 questions per new variable to `eval/retrieval_cases.json` (and a few to `eval/retrieval_cases_heldout.json`
   written **before** you tune anything), with `"gio": true` style flags extended so token-gated entries are included
   (the eval's `giovanni=` argument controls this; extend it for the new providers).
3. `python eval_retrieval.py` and `python eval_retrieval.py --heldout`. Existing scores (hybrid top-1 about 90 percent,
   cov@3 about 95 percent) must not regress. If a new source steals questions from POWER, lower its weight rather
   than editing the lexicon of old variables.

## Documentation and housekeeping

- README: add each source to "Data and credits" (NASA LP DAAC / AppEEARS, NASA PO.DAAC, NASA GIBS / Worldview, NASA
  FIRMS) and note which need a token or key. Update the Giovanni token help text to say the same Earthdata token also
  unlocks AppEEARS and PO.DAAC.
- Streamlit Cloud: any new package goes in `requirements.txt`; `FIRMS_MAP_KEY` goes in the app's Secrets (never committed).
- Clean up smoke-test scripts and any downloaded sample files before finishing. Do not commit tokens, keys or data dumps.

## End-to-end check (run before declaring done)

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=300).run()
at.text_input(key="gio_token").set_value("<an Earthdata token>")   # sidebar token
at.text_input(key="q").set_value("Is the Sahel getting greener?").run()
print([e.value for e in at.exception], [e.value for e in at.error])
print([s.value for s in at.subheader])
```

Repeat for one question per new source (with and without the token or key), and confirm there are no exceptions, no
token in any rendered text or log, and that the original POWER and Giovanni questions still work.

## What to report back

For each phase: what was built, the exact product ids and endpoints that were confirmed to work, anything that did not
match this file, eval scores before and after, and any source you skipped or changed and why.
