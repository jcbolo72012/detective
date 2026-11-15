"""'See it from space': NASA Worldview links and NASA GIBS image previews for a place. No login, no API key.

GIBS (Global Imagery Browse Services) serves satellite imagery by URL. For an investigated place this module builds
  * a link that opens NASA Worldview centred on the place with a layer that matches the variable, and
  * two image URLs: the first year and the last year of the analysis, same time of year.
The pictures are context ("what this place looked like then and now"), never evidence for the statistic.

Layer ids and their record dates come from GIBS GetCapabilities
(https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities);
Worldview's URL parameters are documented at github.com/nasa-gibs/worldview/blob/main/doc/url_parameters.md."""
import datetime as dt
import math
import re
import urllib.parse

import requests
import streamlit as st

WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
WORLDVIEW = "https://worldview.earthdata.nasa.gov/"
BASE = "BlueMarble_ShadedRelief_Bathymetry"   # static backdrop under transparent data layers
OVERLAY = "Coastlines_15m"
WV_REFERENCE = ["Reference_Labels_15m", "Reference_Features_15m", "Coastlines_15m"]

# name -> (GIBS layer id, first date, last date or None if ongoing, "D" daily / "M" monthly, label)
LAYERS = {
    "truecolor": ("MODIS_Terra_CorrectedReflectance_TrueColor", "2000-02-24", None, "D", "natural-colour satellite view"),
    "land_temp": ("MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day", "2000-03-01", None, "M", "daytime land surface temperature"),
    "snow": ("MODIS_Terra_L3_Snow_Cover_Monthly_Average_Pct", "2000-03-01", None, "M", "snow cover"),
    "ndvi": ("MODIS_Terra_L3_NDVI_Monthly", "2000-03-01", None, "M", "vegetation greenness (NDVI)"),
    "aerosol": ("MERRA2_Aerosol_Optical_Depth_Analysis_Monthly", "1980-01-01", "2026-08-01", "M", "aerosol haze (MERRA-2)"),
    "dust": ("MERRA2_Dust_Surface_Mass_Concentration_Monthly", "1980-01-01", "2026-08-01", "M", "surface dust (MERRA-2)"),
    "sea_ice": ("AMSRU2_Sea_Ice_Concentration_12km", "2012-07-02", "2025-09-01", "D", "sea ice concentration"),
    "sst": ("GHRSST_L4_MUR_Sea_Surface_Temperature", "2002-06-01", None, "D", "sea surface temperature"),
}
SEASON_MONTH = {None: None, "DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}

# (regex on catalog id or name, layer key); first match wins; anything else gets the natural-colour view
RULES = [
    (r"^POD:SST", "sst"),
    (r"^FRSEAICE|sea.?ice", "sea_ice"),
    (r"dust", "dust"),
    (r"^AOD_|aerosol|haze|smok|soot|black carbon|organic carbon|optical thickness", "aerosol"),
    (r"^(SNODP|FRSNO|PRECSNO)|snow", "snow"),
    (r"^(GWET|RZMC|SFMC|PRMC|EV)|soil|evapor|vegetation|canopy|leaf", "ndvi"),
    (r"^(T2M|T10M|TS|TSURF|TSOIL)|land surface temp|skin temp", "land_temp"),
]


def layer_for(code, name=""):
    """(key, layer tuple) for the variable; natural colour when nothing more specific fits."""
    text = f"{code} {name}"
    key = next((k for pat, k in RULES if re.search(pat, text, re.I)), "truecolor")
    return key, LAYERS[key]


def _iso(d):
    return dt.date.fromisoformat(d)


def pick_date(layer, year, month, day=15):
    """The date to ask GIBS for, or None if the layer has no data for that time."""
    _, first, last, step, _ = layer
    month = month or 7
    d = dt.date(int(year), month, 1 if step == "M" else min(day, 28))
    if d < _iso(first):
        return None
    if last and d > _iso(last):
        return None
    return d


def _bbox(lat, lon, half_deg):
    half_lon = min(half_deg / max(math.cos(math.radians(lat)), 0.25), 60)
    la0, la1 = max(-90.0, lat - half_deg), min(90.0, lat + half_deg)
    return lon - half_lon, la0, lon + half_lon, la1


def gibs_image_url(layer_key, lat, lon, date, half_deg=2.0, size=480):
    """WMS GetMap URL (WMS 1.1.1: x is longitude, y is latitude). Data layers sit on a Blue Marble backdrop."""
    layer_id = LAYERS[layer_key][0]
    layers = [layer_id] if layer_key == "truecolor" else [BASE, layer_id]
    x0, y0, x1, y1 = _bbox(lat, lon, half_deg)
    q = dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.1.1", LAYERS=",".join(layers + [OVERLAY]),
             SRS="EPSG:4326", BBOX=f"{x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f}", WIDTH=size, HEIGHT=size,  # square on the ground
             FORMAT="image/jpeg", TRANSPARENT="FALSE", TIME=date.isoformat())
    return f"{WMS}?{urllib.parse.urlencode(q, safe=',:')}"


def worldview_url(layer_key, lat, lon, date, half_deg=8.0):
    """Opens Worldview on the place: layers (top first), day, map extent and a location marker (lon,lat)."""
    layer_id = LAYERS[layer_key][0]
    data, backdrop = ([], layer_id) if layer_key == "truecolor" else ([layer_id], BASE)
    layers = WV_REFERENCE + data + [backdrop]
    x0, y0, x1, y1 = _bbox(lat, lon, half_deg)
    q = dict(v=f"{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}", l=",".join(layers), t=f"{date.isoformat()}-T00:00:00Z",
             s=f"{lon:.4f},{lat:.4f}")
    return f"{WORLDVIEW}?{urllib.parse.urlencode(q, safe=',:')}"


def plan(code, name, lat, lon, start_year, end_year, season):
    """Everything the UI needs: {label, worldview, images: [(caption, url)]} (images may be 0, 1 or 2)."""
    key, layer = layer_for(code, name)
    month = SEASON_MONTH.get(season) or (7 if lat >= 0 else 1)
    # the layer may not reach back to the first year (satellite records start around 2000): use its first usable year
    first = next((d for y in range(start_year, end_year - 3) if (d := pick_date(layer, y, month))), None)
    last = next((d for y in range(end_year, start_year - 1, -1) if (d := pick_date(layer, y, month))), None)
    if first and last and first.year >= last.year - 3:
        first = None  # too close together to show a change
    when = last or first
    images = [(f"{d.year}", gibs_image_url(key, lat, lon, d)) for d in (first, last) if d]
    if first is None and last is None:  # no data for those years: fall back to the layer's newest available date
        newest = _iso(layer[2]) if layer[2] else dt.date.today() - dt.timedelta(days=30)
        when = newest
        images = [(newest.strftime("%Y-%m"), gibs_image_url(key, lat, lon, newest))]
    return dict(label=layer[4], worldview=worldview_url(key, lat, lon, when or dt.date.today()), images=images)


def render(p, code, name):
    """Streamlit panel for one investigated case `p` (validated params). Imagery loads only when ticked."""
    pl = plan(code, name, p["lat"], p["lon"], p["start_year"], p["end_year"], p.get("season"))
    with st.expander("🛰️ See it from space (NASA GIBS imagery)"):
        st.link_button("Open this place in NASA Worldview", pl["worldview"])
        if st.checkbox("Show satellite images for this place", key=f"sv_{code}_{p['lat']:.2f}_{p['lon']:.2f}"):
            cols = st.columns(max(1, len(pl["images"])))
            for col, (cap, url) in zip(cols, pl["images"]):
                img = _get_image(url)
                if img is None:
                    col.caption(f"{cap}: NASA has no usable image for this day. Try Worldview.")
                else:
                    col.image(img, caption=f"{cap} · {pl['label']}", width="stretch")
            st.caption("Single-day or single-month pictures for context, from NASA GIBS; colours show relative values "
                       "(Worldview has the legend). Clouds can hide the surface, and a picture is not evidence for "
                       "the trend above.")


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=200)
def _get_image(url):
    """Image bytes, or None when GIBS has nothing usable (error, wrong type, or a blank tile)."""
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException:
        return None
    ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") and len(r.content) > 3000
    return r.content if ok else None
