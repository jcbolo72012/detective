"""Giovanni case files: describe any 'Time Series, Area-Averaged' CSV dropped into ./data automatically.

Reads the CSV's own header (title, units, region), then matches the variable against KNOWN for a
plain-language name and definition. Unknown variables still work: they use the header's own wording."""
from planets import config
import glob
import os
import re

DATA_DIR = config.DATA_DIR

# (pattern matched against the CSV's column name + title, description). First match wins.
KNOWN = [
    (r"TO3|ColumnAmountO3|ozone", dict(
        name="Total column ozone", units="DU", group="Atmosphere", season="SON",
        instrument="OMI (Aura satellite) or MERRA-2 model",
        definition="All the ozone in the air above a place, which shields us from UV. Measured in Dobson Units: "
                   "300 DU squeezed to sea level would be a layer 3 mm thick. Over Antarctica it dips every "
                   "Sep–Nov (the 'ozone hole'), so analyse those months.")),
    (r"NO2|nitrogen dioxide", dict(
        name="Nitrogen dioxide (NO₂) in the lower atmosphere", units="molecules/cm²", group="Air quality", season=None,
        instrument="OMI (Aura satellite)",
        definition="A pollution gas made when fuel burns (cars, power plants, industry). The satellite counts how "
                   "much is in the column of air above each spot, so it tracks a city's emissions.")),
    (r"AOD|Aerosol_Optical|TOTEXTTAU|aerosol", dict(
        name="Aerosol optical depth (haze)", units="unitless", group="Air quality", season=None,
        instrument="MODIS (Terra/Aqua satellites) or MERRA-2 model",
        definition="How much tiny particles in the air (smoke, dust, pollution haze) block sunlight. "
                   "About 0.1 is a clear day; 1 is very hazy.")),
    (r"GWS|groundwater|TWS|GRACE|water storage", dict(
        name="Groundwater / water storage", units="mm", group="Water", season=None,
        instrument="GRACE satellites (via GLDAS)",
        definition="Water held underground and in soil, measured from space by tiny changes in Earth's gravity. "
                   "A falling trend means more water is being pumped or lost than rain puts back.")),
    (r"chlor|chl_", dict(
        name="Ocean chlorophyll", units="mg/m³", group="Ocean", season=None, instrument="MODIS (Aqua satellite)",
        definition="The green pigment of microscopic ocean plants, seen as ocean colour. More chlorophyll means more "
                   "plant life at the base of the ocean food chain.")),
    (r"MOPITT|carbon monoxide|_CO_", dict(
        name="Carbon monoxide", units="ppbv", group="Air quality", season=None, instrument="MOPITT (Terra satellite)",
        definition="A gas from fires and incomplete burning of fuel. It travels far, so it traces smoke and pollution plumes.")),
    (r"NDVI|EVI|vegetation", dict(
        name="Vegetation greenness", units="index (0-1)", group="Land", season=None, instrument="MODIS",
        definition="How green and leafy the land is, from the light plants reflect. Higher means denser, healthier vegetation.")),
    (r"soil ?moi|SoilMoi|GWET|soil wetness", dict(
        name="Soil moisture", units="kg/m²", group="Water", season=None, instrument="GLDAS model / SMAP",
        definition="How much water the soil holds. Falling soil moisture is an early sign of drought.")),
    (r"precip|IMERG|TRMM|rain", dict(
        name="Precipitation", units="mm/month", group="Water", season=None, instrument="GPM IMERG (satellites)",
        definition="Rain plus melted snow reaching the ground, measured by a constellation of satellites.")),
    (r"T2M|temperature|Tair|SurfAirTemp|SurfSkinTemp", dict(
        name="Temperature", units="K", group="Temperature", season=None, instrument="AIRS (Aqua) or MERRA-2 model",
        definition="Air or surface temperature. Giovanni usually reports it in kelvin (K); a change of 1 K equals 1 °C.")),
]


def read_meta(path):
    """Header of a Giovanni CSV -> {title, long_name, product, units, region, column}. Missing parts are ''."""
    meta = dict(title="", long_name="", product="", units="", region="", column="")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            low = line.lower()
            if low.startswith("title"):
                meta["title"] = line.split(",", 1)[1].strip() if "," in line else ""
            elif low.startswith("time"):
                cols = line.strip().split(",")
                meta["column"] = cols[1] if len(cols) > 1 else ""
                break
    t = meta["title"]
    m = re.search(r"\bof\s+(.+?)\s*\[(.+?)\]\s*(.*?)\s+over\s", t)
    if m:
        meta["long_name"], meta["product"], meta["units"] = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    else:
        m = re.search(r"\bof\s+(.+)$", t)
        meta["long_name"] = m.group(1).strip() if m else ""
    m = re.search(r"Region\s+(.+)$", t)
    if m:
        meta["region"] = m.group(1).strip().strip('"')
    return meta


def describe(path):
    """Variable card for a case file: {name, units, definition, source, group, season, region, file, synthetic}."""
    meta = read_meta(path)
    base = os.path.basename(path)
    hay = f"{meta['column']} {meta['title']} {base}"
    known = next((d for pat, d in KNOWN if re.search(pat, hay, re.I)), None)
    pretty = os.path.splitext(base)[0].replace("_", " ")
    synthetic = "synthetic" in base.lower()
    d = dict(
        name=(known or {}).get("name") or meta["long_name"] or pretty,
        units=meta["units"] or (known or {}).get("units") or "units",
        definition=(known or {}).get("definition") or meta["long_name"] or "No description available for this file.",
        source="SYNTHETIC test data (not real measurements)" if synthetic else
               "NASA Giovanni" + (f" · {meta['product']}" if meta["product"] else
                                  f" · {known['instrument']}" if known else ""),
        group=(known or {}).get("group", "Other"), season=(known or {}).get("season"),
        region=meta["region"] or "Giovanni region", file=path, synthetic=synthetic, keywords=[])
    return d


def case_files():
    return [describe(p) for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))]


def _link(data, bbox, start, end="2025-12-31"):
    return (f"https://giovanni.gsfc.nasa.gov/giovanni/#service=ArAvTs&starttime={start}T00:00:00Z"
            f"&endtime={end}T23:59:59Z&bbox={bbox}&data={data}")


# One-click Giovanni requests (need a free Earthdata login). After the plot appears: Downloads > CSV > save into data/
RECIPES = [
    dict(title="NO₂ over New York City, 2005–2025 (OMI)", keyword="OMNO2d tropospheric",
         url=_link("OMNO2d_003_ColumnAmountNO2TropCloudScreened", "-74.556,40.2182,-72.5894,41.2894", "2005-01-01"),
         why="Air-pollution trend over a city. Daily data, so the request can take several minutes."),
    dict(title="NO₂ over Delhi, 2005–2025 (OMI)", keyword="OMNO2d tropospheric",
         url=_link("OMNO2d_003_ColumnAmountNO2TropCloudScreened", "76.8,28.3,77.6,28.9", "2005-01-01"),
         why="Pair with New York to look for opposite trends in the same pollutant."),
    dict(title="Total ozone over Antarctica, 1980–2025 (MERRA-2 monthly)", keyword="MERRA-2 total column ozone monthly",
         url=_link("M2TMNXSLV_5_12_4_TO3", "-180,-90,180,-60", "1980-01-01"),
         why="The ozone hole and its slow recovery. Analyse Sep–Nov."),
    dict(title="Aerosol haze over eastern China, 2000–2025 (MODIS Terra monthly)", keyword="MOD08_M3 aerosol optical depth",
         url=_link("MOD08_M3_6_1_Aerosol_Optical_Depth_Land_Ocean_Mean_Mean", "110,25,122,40", "2000-03-01"),
         why="Haze rose, then fell after clean-air policies: a trend that changes direction."),
]
