"""Catalog of NASA Giovanni variables the app can fetch with the user's own Earthdata token.

One-time build (run on a computer that can reach NASA; commit the result so the hosted app has it):

    python -m uv run python giovanni_catalog.py --build

It downloads NASA's list of variables configured for Giovanni's cloud services
(https://api.giovanni.earthdata.nasa.gov/configured-variables/), turns each ID into a readable name, and saves
giovanni_vars.json next to this file (plus giovanni_vars_raw.json, NASA's untouched answer, for debugging).
If NASA wants a login for the list, it uses EARTHDATA_TOKEN from .env or asks for it. The token is never saved.

Until the file exists, a small built-in starter list (STARTER) is used.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(HERE, "giovanni_vars.json")
RAW_FILE = os.path.join(HERE, "giovanni_vars_raw.json")
LIST_URL = "https://api.giovanni.earthdata.nasa.gov/configured-variables/"
PREFIX = "GIO:"  # catalog keys are "GIO:<giovanni id>" so they never clash with NASA POWER codes

# ---------------------------------------------------------------------------------------------- naming
# (regex on the Giovanni ID, product label, time step, first year, last year or None, region note).
# Group "v" is the variable part. First match wins. Built from NASA's configured-variables list (Nov 2026).
PRODUCTS = [
    (r"^AIRS3STD_(?:006|7_0)_(?P<v>.+)$", "AIRS satellite (Aqua)", "daily", 2003, None, ""),
    (r"^AIRS3STM_(?:006|7_0)_(?P<v>.+)$", "AIRS satellite (Aqua)", "monthly", 2003, None, ""),
    (r"^AIRX3STD_(?:006|7_0)_(?P<v>.+)$", "AIRS+AMSU satellites (Aqua)", "daily", 2003, None, ""),
    (r"^AIRX3STM_(?:006|7_0)_(?P<v>.+)$", "AIRS+AMSU satellites (Aqua)", "monthly", 2003, None, ""),
    (r"^FLDAS_NOAH01_CP?_GL_M_001_(?P<v>.+)$", "FLDAS land model, global", "monthly", 1982, None, ""),
    (r"^FLDAS_NOAHMP001_G_CA_D_001_(?P<v>.+)$", "FLDAS land model, Central Asia", "daily", 2001, None, "Central Asia only"),
    (r"^GLDAS_CLSM025_DA1_D_2_2_(?P<v>.+)$", "GLDAS Catchment model v2.2 (with GRACE)", "daily", 2003, None, ""),
    (r"^GLDAS_CLSM025_D_2_0_(?P<v>.+)$", "GLDAS Catchment model v2.0", "daily", 1948, 2014, ""),
    (r"^GLDAS_NOAH025_3H_2_0_(?P<v>.+)$", "GLDAS Noah land model v2.0, 0.25°", "3-hourly", 1948, 2014, ""),
    (r"^GLDAS_NOAH025_3H_2_1_(?P<v>.+)$", "GLDAS Noah land model v2.1, 0.25°", "3-hourly", 2000, None, ""),
    (r"^GLDAS_NOAH025_M_2_0_(?P<v>.+)$", "GLDAS Noah land model v2.0, 0.25°", "monthly", 1948, 2014, ""),
    (r"^GLDAS_NOAH025_M_2_1_(?P<v>.+)$", "GLDAS Noah land model v2.1, 0.25°", "monthly", 2000, None, ""),
    (r"^GLDAS_NOAH10_M_2_0_(?P<v>.+)$", "GLDAS Noah land model v2.0, 1°", "monthly", 1948, 2014, ""),
    (r"^GLDAS_NOAH10_M_2_1_(?P<v>.+)$", "GLDAS Noah land model v2.1, 1°", "monthly", 2000, None, ""),
    (r"^GPCPDAY_3_3_(?P<v>.+)$", "GPCP v3.3 precipitation", "daily", 2001, None, ""),
    (r"^GPM_3IMERGM_0?\d+_(?P<v>.+)$", "GPM IMERG satellite precipitation", "monthly", 2001, None, ""),
    (r"^GPM_3IMERGD[EFL]_0?\d+_(?P<v>.+)$", "GPM IMERG satellite precipitation", "daily", 2001, None, ""),
    (r"^GPM_3IMERGHH[EL]?_0?\d+_(?P<v>.+)$", "GPM IMERG satellite precipitation", "half-hourly", 2001, None, ""),
    (r"^GRACEDADM_CLSM0125US_7D_4_0_(?P<v>.+)$", "GRACE drought model, USA", "weekly", 2003, None, "USA only"),
    (r"^GRACEDADM_CLSM025GL_7D_3_0_(?P<v>.+)$", "GRACE drought model, global", "weekly", 2003, None, ""),
    (r"^M2T1NX(?P<c>[A-Z]{3})_5_12_4_(?P<v>.+)$", "MERRA-2 {c}", "hourly", 1980, None, ""),
    (r"^M2I1NX(?P<c>[A-Z]{3})_5_12_4_(?P<v>.+)$", "MERRA-2 {c}", "hourly", 1980, None, ""),
    (r"^M2[TI]MN[XP](?P<c>[A-Z]{3})_5_12_4_(?P<v>.+)$", "MERRA-2 {c}", "monthly", 1980, None, ""),
    (r"^M2SDNX(?P<c>[A-Z]{3})_5_12_4_(?P<v>.+)$", "MERRA-2 {c}", "daily", 1980, None, ""),
    (r"^NCALDAS_NOAH0125_D_2_0_(?P<v>.+)$", "NCA-LDAS land model, USA", "daily", 1979, None, "USA only"),
    (r"^NLDAS_(?P<m>FORA|NOAH|MOS|VIC)0125_M_2_0_(?P<v>.+)$", "NLDAS-2 {m}, North America", "monthly", 1979, None,
     "North America only"),
    (r"^NLDAS_(?P<m>FORA|NOAH|MOS|VIC)0125_H_2_0_(?P<v>.+)$", "NLDAS-2 {m}, North America", "hourly", 1979, None,
     "North America only"),
    (r"^NOBM_DAY_R2017_(?P<v>.+)$", "NASA Ocean Biogeochemical Model", "daily", 1998, None, "oceans only"),
    (r"^OCO2_GEOS_L3CO2_DAY_10r_(?P<v>.+)$", "OCO-2 satellite (GEOS assimilation)", "daily", 2015, None, ""),
    (r"^OM[A-Z0-9]+[de]?_0*\d+_(?P<v>.+)$", "Aura OMI satellite", "daily", 2005, None, ""),
    (r"^SMERGE_RZSM0_40CM_2_0_(?P<v>.+)$", "SMERGE merged soil moisture, USA", "daily", 1979, None, "USA only"),
    (r"^TRMM_3B42_Daily_7_(?P<v>.+)$", "TRMM satellite precipitation", "daily", 1998, 2019, ""),
    (r"^TRMM_3B43_7_(?P<v>.+)$", "TRMM satellite precipitation", "monthly", 1998, 2019, ""),
]
MERRA_COLL = {"SLV": "weather", "AER": "aerosols", "FLX": "surface fluxes", "RAD": "radiation", "LND": "land",
              "OCN": "ocean", "CHM": "chemistry", "ADG": "aerosol diagnostics", "GLC": "land ice", "LFO": "land forcing",
              "ASM": "weather", "INT": "vertical integrals", "CSP": "clouds"}
NLDAS_MODEL = {"FORA": "forcing", "NOAH": "Noah model", "MOS": "Mosaic model", "VIC": "VIC model"}

_T, _P, _S, _A, _H = ("temperature hot warm heat cold", "rain precipitation wet dry drought storm",
                      "snow snowpack winter", "aerosol air quality pollution haze", "humidity humid muggy moisture")
# variable short name (suffixes _inst/_tavg/_acc/_f and AIRS _A/_D removed) -> (name, units or "", keywords)
VARS = {
    # land models (GLDAS, FLDAS, NLDAS, NCA-LDAS)
    "Tair": ("Near-surface air temperature", "K", _T),
    "Tair_f_max": ("Daily maximum air temperature", "K", _T + " heatwave"),
    "Tair_f_min": ("Daily minimum air temperature", "K", _T + " frost night"), "AvgSurfT": ("Land surface temperature", "K", _T),
    "RadT": ("Surface radiative temperature", "K", _T), "SnowT": ("Snow temperature", "K", _S),
    "Rainf": ("Rainfall rate", "", _P), "CRainf": ("Convective rainfall", "", _P),
    "CRainf_frac": ("Fraction of rain that is convective", "", _P), "Snowf": ("Snowfall rate", "", _S),
    "SWE": ("Snow water equivalent", "kg m-2", _S), "SnowDepth": ("Snow depth", "m", _S),
    "SnowCover": ("Snow cover", "", _S), "Snowcover": ("Snow cover", "", _S), "SnowFrac": ("Snow cover fraction", "", _S),
    "SubSnow": ("Snow sublimation", "", _S), "Qsm": ("Snowmelt", "", _S + " melt"),
    "RootMoist": ("Root-zone soil moisture", "kg m-2", "soil moisture drought plants"),
    "Evap": ("Evapotranspiration", "", "evaporation water plants drought"),
    "PotEvap": ("Potential evaporation", "", "evaporation drought"), "ECanop": ("Evaporation from plant canopies", "", "evaporation plants"),
    "ESoil": ("Evaporation from bare soil", "", "evaporation soil"), "EvapSnow": ("Evaporation from snow", "", _S),
    "Tveg": ("Plant transpiration", "", "plants vegetation transpiration"), "TVeg": ("Plant transpiration", "", "plants vegetation"),
    "CanopInt": ("Water held on plant canopies", "kg m-2", "plants water"),
    "Qair": ("Specific humidity", "kg kg-1", _H), "Psurf": ("Surface pressure", "Pa", "pressure"),
    "PSurf": ("Surface pressure", "Pa", "pressure"), "Wind": ("Wind speed", "m s-1", "wind windy"),
    "Wind_E": ("Eastward wind", "m s-1", "wind"), "Wind_N": ("Northward wind", "m s-1", "wind"),
    "SWdown": ("Sunlight reaching the ground", "W m-2", "sun sunlight solar radiation"),
    "LWdown": ("Downward longwave (heat) radiation", "W m-2", "radiation greenhouse heat"),
    "Swnet": ("Net sunlight absorbed", "W m-2", "sun solar radiation"), "SWnet": ("Net sunlight absorbed", "W m-2", "sun solar"),
    "Lwnet": ("Net longwave radiation", "W m-2", "radiation"), "LWnet": ("Net longwave radiation", "W m-2", "radiation"),
    "Albedo": ("Surface albedo (reflectivity)", "%", "reflect albedo ice snow"),
    "Qg": ("Ground heat flux", "W m-2", "heat soil"), "Qh": ("Sensible heat flux", "W m-2", "heat"),
    "Qle": ("Latent heat flux (evaporation energy)", "W m-2", "heat evaporation"), "Qf": ("Snow phase-change heat", "W m-2", _S),
    "Qs": ("Surface runoff", "", "runoff flood water"), "Qsb": ("Subsurface runoff", "", "runoff drainage"),
    "GWS": ("Groundwater storage", "mm", "groundwater aquifer well water table depletion drought"),
    "TWS": ("Total water storage", "mm", "water storage groundwater drought"),
    "Streamflow": ("Streamflow", "m3 s-1", "river flow flood water"), "CAPE": ("Storm energy (CAPE)", "J kg-1", "storm thunderstorm convection"),
    "FloodedArea": ("Flooded area", "", "flood"), "FloodedFrac": ("Flooded fraction", "", "flood"),
    "IrrigatedWater": ("Irrigation water", "", "irrigation agriculture farming water"),
    "SoilMoist_S": ("Surface soil moisture", "kg m-2", "soil moisture drought"),
    "SoilMoist_RZ": ("Root-zone soil moisture", "kg m-2", "soil moisture drought"),
    "SoilMoist_P": ("Soil moisture, whole profile", "kg m-2", "soil moisture drought"),
    "gws": ("Groundwater storage (percentile)", "%", "groundwater drought aquifer"),
    "rtzsm": ("Root-zone soil moisture (percentile)", "%", "soil moisture drought"),
    "sfsm": ("Surface soil moisture (percentile)", "%", "soil moisture drought"),
    "RZSM": ("Root-zone soil moisture (0–40 cm)", "m3 m-3", "soil moisture drought"),
    "CCI_ano": ("Root-zone soil moisture anomaly", "m3 m-3", "soil moisture drought anomaly"),
    # AIRS
    "SurfAirTemp": ("Surface air temperature", "K", _T), "SurfSkinTemp": ("Surface skin temperature", "K", _T),
    "CloudFrc": ("Cloud fraction", "", "cloud cloudy"), "CloudTopPres": ("Cloud-top pressure", "hPa", "cloud"),
    "CloudTopTemp": ("Cloud-top temperature", "K", "cloud"), "OLR": ("Outgoing longwave radiation", "W m-2", "radiation heat greenhouse"),
    "ClrOLR": ("Clear-sky outgoing longwave radiation", "W m-2", "radiation greenhouse"),
    "H2O_MMR_Surf": ("Surface water vapour", "g kg-1", _H), "RelHumSurf": ("Surface relative humidity", "%", _H),
    "TotH2OVap": ("Total water vapour", "kg m-2", _H), "TotO3": ("Total column ozone", "DU", "ozone hole uv"),
    "TropHeight": ("Tropopause height", "m", "atmosphere"), "TropPres": ("Tropopause pressure", "hPa", "atmosphere"),
    "TropTemp": ("Tropopause temperature", "K", "atmosphere"),
    # MERRA-2
    "T2M": ("Air temperature at 2 m", "K", _T), "T10M": ("Air temperature at 10 m", "K", _T),
    "T2MMAX": ("Daily maximum temperature", "K", _T + " heatwave"), "T2MMIN": ("Daily minimum temperature", "K", _T + " frost night"),
    "T2MMEAN": ("Daily mean temperature", "K", _T), "TS": ("Surface skin temperature", "K", _T),
    "TLML": ("Air temperature, lowest model level", "K", _T), "QLML": ("Humidity, lowest model level", "kg kg-1", _H),
    "SPEEDLML": ("Wind speed, lowest model level", "m s-1", "wind"), "QV2M": ("Specific humidity at 2 m", "kg kg-1", _H),
    "QV10M": ("Specific humidity at 10 m", "kg kg-1", _H), "TO3": ("Total column ozone", "Dobson", "ozone hole"),
    "TROPQ": ("Tropopause humidity", "kg kg-1", "atmosphere"), "TROPT": ("Tropopause temperature", "K", "atmosphere"),
    "PHIS": ("Surface height (geopotential)", "m2 s-2", "terrain"), "DISPH": ("Displacement height", "m", "terrain"),
    "PS": ("Surface pressure", "Pa", "pressure"), "TQV": ("Total precipitable water vapour", "kg m-2", _H),
    "CLDTMP": ("Cloud-top temperature", "K", "cloud"), "U10M": ("Eastward wind at 10 m", "m s-1", "wind"),
    "V10M": ("Northward wind at 10 m", "m s-1", "wind"), "U50M": ("Eastward wind at 50 m", "m s-1", "wind"),
    "V50M": ("Northward wind at 50 m", "m s-1", "wind"), "SPEED": ("Surface wind speed", "m s-1", "wind windy"),
    "SPEEDMAX": ("Maximum surface wind speed", "m s-1", "wind gust storm"), "ULML": ("Eastward surface wind", "m s-1", "wind"),
    "VLML": ("Northward surface wind", "m s-1", "wind"), "EFLUX": ("Latent heat flux", "W m-2", "heat evaporation"),
    "HFLUX": ("Sensible heat flux", "W m-2", "heat"), "EVAP": ("Evaporation", "kg m-2 s-1", "evaporation"),
    "PRECTOT": ("Total precipitation", "kg m-2 s-1", _P), "PRECTOTCORR": ("Total precipitation (bias corrected)", "kg m-2 s-1", _P),
    "LWGAB": ("Longwave radiation absorbed at the ground", "W m-2", "radiation"),
    "SWGDN": ("Sunlight reaching the ground", "W m-2", "sun sunlight solar"),
    "SWGDNCLR": ("Clear-sky sunlight reaching the ground", "W m-2", "sun solar"), "SWGNT": ("Net sunlight at the ground", "W m-2", "sun solar"),
    "LWGNT": ("Net longwave radiation at the ground", "W m-2", "radiation"), "GWETTOP": ("Topsoil wetness", "1", "soil moisture drought"),
    "ALBEDO": ("Surface albedo (reflectivity)", "1", "albedo reflect ice snow"), "ALBNIRDF": ("Near-infrared albedo", "1", "albedo"),
    "ALBVISDF": ("Visible albedo", "1", "albedo"), "EMIS": ("Surface emissivity", "1", "radiation"),
    "CLDTOT": ("Total cloud fraction", "1", "cloud cloudy"), "CLDLOW": ("Low cloud fraction", "1", "cloud cloudy fog"),
    "CLDMID": ("Mid-level cloud fraction", "1", "cloud"), "CLDHGH": ("High cloud fraction", "1", "cloud"),
    "DUCMASS": ("Dust column mass density", "kg m-2", "dust storm sand desert " + _A),
    "DUCMASS25": ("Fine dust (PM2.5) column mass", "kg m-2", "dust pm2.5 " + _A),
    "DUSMASS": ("Dust surface concentration", "kg m-3", "dust storm sand " + _A),
    "DUSMASS25": ("Fine dust (PM2.5) surface concentration", "kg m-3", "dust pm2.5 particulate " + _A),
    "DUEXTTAU": ("Dust optical thickness (dust haze)", "1", "dust haze " + _A), "DUEXTT25": ("Fine-dust optical thickness", "1", "haze visibility fine particles"),
    "DUSCATFM": ("Dust scattering (fine mode)", "1", "scattering"),
    "BCSMASS": ("Black carbon (soot) surface concentration", "kg m-3", "black carbon soot smoke fire diesel " + _A),
    "BCCMASS": ("Black carbon (soot) column mass", "kg m-2", "black carbon soot smoke fire " + _A),
    "OCSMASS": ("Organic carbon surface concentration", "kg m-3", "organic carbon smoke wildfire " + _A),
    "OCCMASS": ("Organic carbon column mass", "kg m-2", "organic carbon smoke wildfire " + _A),
    "SO2SMASS": ("Sulfur dioxide surface concentration", "kg m-3", "sulfur dioxide so2 coal industry volcano " + _A),
    "SO2CMASS": ("Sulfur dioxide column mass", "kg m-2", "sulfur dioxide so2 coal industry volcano " + _A),
    "SO4SMASS": ("Sulfate surface concentration", "kg m-3", "sulfate " + _A), "SO4CMASS": ("Sulfate column mass", "kg m-2", "sulfate " + _A),
    "SSSMASS25": ("Fine sea salt surface concentration", "kg m-3", "sea salt ocean " + _A),
    "SSCMASS25": ("Fine sea salt column mass", "kg m-2", "sea salt ocean"),
    "TOTEXTTAU": ("Aerosol optical thickness (haze)", "1", "haze smoke smog visibility " + _A),
    "TOTSCATAU": ("Aerosol scattering thickness", "1", "haze " + _A), "TOTANGSTR": ("Aerosol particle size (Angstrom)", "1", _A),
    # precipitation products
    "precipitation": ("Precipitation rate", "", _P), "precip": ("Precipitation", "mm/day", _P),
    "precipitationQualityIndex": ("Precipitation quality index", "", "rain quality"),
    "probabilityLiquidPrecipitation": ("Chance precipitation falls as rain", "%", _P + " " + _S),
    "probability_liquid_phase": ("Chance precipitation falls as rain", "%", _P + " " + _S),
    "randomError": ("Precipitation random error", "", "rain error"),
    # ocean biology (NOBM)
    "chl": ("Chlorophyll (ocean model)", "mg m-3", "chlorophyll ocean algae phytoplankton bloom"),
    "tot": ("Total phytoplankton (ocean model)", "mg m-3", "phytoplankton ocean algae plankton"),
    "dia": ("Diatoms (ocean model)", "mg m-3", "diatoms ocean plankton"), "cya": ("Cyanobacteria (ocean model)", "mg m-3", "cyanobacteria ocean algae"),
    "coc": ("Coccolithophores (ocean model)", "mg m-3", "plankton ocean"), "irn": ("Dissolved iron (ocean model)", "nM", "iron ocean"),
    "rno": ("Nitrate (ocean model)", "uM", "nitrate nutrient ocean"), "h": ("Herbivores (ocean model)", "mg m-3", "ocean zooplankton"),
    # greenhouse gas
    "XCO2": ("Carbon dioxide (CO₂), column average", "ppm",
             "carbon dioxide co2 greenhouse gas emissions anthropogenic fossil fuel climate carbon"),
    # OMI
    "ColumnAmountO3": ("Total column ozone", "DU", "ozone hole uv"),
    "ColumnAmountNO2TropCloudScreened": ("Tropospheric nitrogen dioxide (NO₂)", "molec cm-2",
                                         "no2 nitrogen dioxide traffic cars emissions anthropogenic industry pollution air quality"),
    "ColumnAmountNO2CloudScreened": ("Total nitrogen dioxide (NO₂)", "molec cm-2", "no2 nitrogen dioxide pollution emissions"),
    "ColumnAmountSO2": ("Sulfur dioxide (SO₂) column", "DU", "so2 sulfur dioxide volcano coal industry emissions pollution"),
    "key_science_data_column_amount": ("Formaldehyde (HCHO) column", "molec cm-2", "formaldehyde hcho pollution fire"),
    "UVAerosolIndex": ("UV aerosol index (smoke and dust)", "", "smoke dust fire " + _A),
    "RadiativeCloudFraction": ("Cloud fraction", "", "cloud cloudy"), "UVindex": ("UV index", "", "uv sunburn ultraviolet sun"),
    "ErythemalDailyDose": ("Daily sunburning UV dose", "J m-2", "uv sunburn ultraviolet"),
    "ErythemalDoseRate": ("Sunburning UV dose rate", "mW m-2", "uv sunburn ultraviolet"),
}
_VARS_CI = {k.lower(): v for k, v in VARS.items()}
_DEPTH = [  # (regex on variable, name template, units, keywords)
    (r"^SoilMoi(?:st)?0*(\d+)_(\d+)cm$", "Soil moisture, {0}–{1} cm", "kg m-2", "soil moisture drought"),
    (r"^SoilM_(\d+)_(\d+)cm$", "Soil moisture, {0}–{1} cm", "kg m-2", "soil moisture drought"),
    (r"^SMLiq_(\d+)_(\d+)cm$", "Liquid soil moisture, {0}–{1} cm", "kg m-2", "soil moisture"),
    (r"^SMAvail_(\d+)_(\d+)cm$", "Plant-available soil moisture, {0}–{1} cm", "kg m-2", "soil moisture drought plants"),
    (r"^Soil(?:TMP|Temp|T_)0*(\d+)_(\d+)cm$", "Soil temperature, {0}–{1} cm", "K", "soil temperature"),
    (r"^SoilT_(\d+)_(\d+)cm$", "Soil temperature, {0}–{1} cm", "K", "soil temperature"),
    (r"^SoilM_(layer\d|total)$", "Soil moisture ({0})", "kg m-2", "soil moisture drought"),
    (r"^SoilT(?:_(layer\d))?$", "Soil temperature{0}", "K", "soil temperature"),
    (r"^FinalAerosolOpticalDepth(\d+)$", "Aerosol optical depth at {0} nm (haze)", "", "haze smoke dust " + _A),
    (r"^FinalAerosolAbsOpticalDepth(\d+)$", "Absorbing aerosol optical depth at {0} nm (smoke, dust)", "", "smoke dust soot " + _A),
    (r"^FinalAerosolSingleScattAlb(\d+)$", "Aerosol single-scattering albedo at {0} nm", "", _A),
    (r"^Irradiance(\d+)$", "UV irradiance at {0} nm", "W m-2 nm-1", "uv ultraviolet sun"),
]


def _var_lookup(var):
    """Variable part -> (name, units, keywords, pass_note) or None."""
    note = ""
    base = var
    m = re.match(r"^(.*)_([AD])$", base)
    if m and m.group(1):  # AIRS orbits: A = ascending (~1:30 pm), D = descending (~1:30 am)
        base, note = m.group(1), " (daytime pass)" if m.group(2) == "A" else " (night-time pass)"
    for cand in (base, re.sub(r"_(inst|tavg|acc)(_max|_min)?$", "", base), re.sub(r"_(f_)?(inst|tavg|acc)$|_f$", "", base)):
        hit = _VARS_CI.get(cand.lower())
        if hit:
            extra = " (daily max)" if base.endswith("_max") else " (daily min)" if base.endswith("_min") else ""
            return hit[0] + extra + note, hit[1], hit[2]
        for pat, tmpl, units, kw in _DEPTH:
            dm = re.match(pat, cand)
            if dm:
                groups = [g if g is not None else "" for g in dm.groups()]
                if tmpl.endswith("{0}") and groups and groups[0] and not tmpl.endswith(" {0} nm"):
                    groups = [f", {groups[0]}" if "Soil temperature" in tmpl else groups[0]] + groups[1:]
                return tmpl.format(*groups) + note, units, kw
    return None


KELVIN = {"k", "kelvin", "degk", "deg k"}

FIRST_YEAR = [  # (prefix, first full year of data) for picking a sensible default start
    ("GLDAS_NOAH025", 2000), ("GLDAS_NOAH10", 2000), ("NLDAS", 1979), ("FLDAS", 1982), ("M2", 1980),
    ("GPM_3IMERG", 2001), ("OM", 2005), ("AIRS", 2003), ("MOD08", 2001), ("MYD08", 2003)]


def first_year(gid):
    return next((y for p, y in FIRST_YEAR if gid.startswith(p)), None)


def to_celsius_if_kelvin(units):
    """Display units: kelvin values are converted to °C when fetched, so the catalog says °C too."""
    return "°C" if (units or "").strip().lower() in KELVIN else (units or "")


def _humanize(v):
    v = re.sub(r"_(inst|tavg|acc|f)$", "", v)
    v = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", v.replace("_", " "))
    return v.strip() or v


def describe_id(gid):
    """Giovanni ID -> dict(name, units, product, step, first_year, last_year, region, keywords, known).
    Pure string work, no network."""
    product, step, var, first, last, region = "NASA Giovanni dataset", "", gid, None, None, ""
    for pat, label, st, fy, ly, reg in PRODUCTS:
        m = re.match(pat, gid)
        if m:
            g = m.groupdict()
            var = g["v"]
            product = label.format(c=MERRA_COLL.get(g.get("c"), g.get("c") or ""),
                                   m=NLDAS_MODEL.get(g.get("m"), g.get("m") or ""))
            step, first, last, region = st, fy, ly, reg
            break
    else:  # unknown product: split at the last version-like run of digits
        m = re.match(r"^(?P<p>.+?_\d+(?:_\d+)*)_(?P<v>[A-Za-z].*)$", gid)
        if m:
            product, var = m.group("p").replace("_", " "), m.group("v")
    known = _var_lookup(var)
    name, units, kw = known if known else (_humanize(var), "", "")
    return dict(name=name, units=units, product=product, step=step, first_year=first, last_year=last,
                region=region, keywords=kw, known=bool(known))


def make_entry(gid, name=None, units=None, definition=None, step=None, start=None, end=None, verified=False):
    d = describe_id(gid)
    name = name or d["name"]
    units_raw = units if units else d["units"]
    step = step or d["step"]
    kelvin = (units_raw or "").strip().lower() in KELVIN
    definition = definition or (f"{name} from {d['product']}" + (f", {step} data averaged to monthly means" if step
                                and step != "monthly" else "") + "." + (f" Covers {d['region']}." if d["region"] else "")
                                + (f" Record ends {d['last_year']}." if d["last_year"] else ""))
    if kelvin:
        definition += " Shown in °C (converted from kelvin)."
    return {
        "name": f"{name} ({d['product']})", "short_name": name,
        "units": to_celsius_if_kelvin(units_raw), "native_units": units_raw or "",
        "definition": definition, "source": f"NASA Giovanni · {d['product']}",
        "keywords": [k for k in (d["keywords"] + " satellite giovanni").split() if k],
        "provider": "giovanni", "giovanni_id": gid, "step": step or "",
        "start": start, "end": end, "verified": bool(verified),
        "first_year": _year(start) or d["first_year"] or first_year(gid),
        "last_year": _year(end) or d["last_year"], "region": d["region"],
    }


def _year(v):
    m = re.match(r"^\s*(\d{4})", str(v or ""))
    return int(m.group(1)) if m else None


# Built-in starter list, used until giovanni_vars.json is built. Only the GLDAS temperature ID is confirmed
# (NASA's own time series tutorial); DUCMASS is confirmed for Giovanni's averaging service. The rest follow
# NASA naming patterns and may be rejected; the app then shows NASA's message.
STARTER = [
    ("GLDAS_NOAH025_3H_2_1_Tair_f_inst", True),
    ("M2T1NXAER_5_12_4_DUCMASS", True),
    ("GLDAS_NOAH025_3H_2_1_Rainf_f_tavg", False),
    ("GLDAS_NOAH025_3H_2_1_SoilMoi0_10cm_inst", False),
    ("GLDAS_NOAH025_3H_2_1_SWE_inst", False),
    ("M2T1NXSLV_5_12_4_T2M", False),
    ("M2T1NXSLV_5_12_4_TO3", False),
    ("M2T1NXAER_5_12_4_TOTEXTTAU", False),
    ("M2T1NXAER_5_12_4_BCSMASS", False),
]


# ---------------------------------------------------------------------------------------------- normalize
_ID_KEYS = ("id", "variable", "variable_id", "variableId", "data", "name", "dataFieldId", "variable_name")
_NAME_KEYS = ("long_name", "longName", "dataFieldLongName", "title", "label", "display_name", "description")
_UNIT_KEYS = ("units", "unit", "dataFieldUnits", "dataFieldUnitsValue")
_DEF_KEYS = ("definition", "summary", "abstract")
_STEP_KEYS = ("temporal_resolution", "timeResolution", "dataProductTimeInterval", "interval", "time_step")
_START_KEYS = ("start", "start_time", "startTime", "dataProductBeginDateTime", "begin")
_END_KEYS = ("end", "end_time", "endTime", "dataProductEndDateTime", "stop")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9.]+){2,}$")


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if v not in (None, "", []):
            return v if not isinstance(v, list) else v[0]
    return None


def normalize(raw):
    """NASA's answer (shape not documented) -> {catalog_key: entry}. Handles: list of IDs, list of objects,
    {id: object|string}, and any of those wrapped in {"variables"|"items"|"results"|"data": ...}."""
    if isinstance(raw, dict):
        for wrap in ("variables", "items", "results", "data", "configured_variables", "configuredVariables"):
            if wrap in raw and isinstance(raw[wrap], (list, dict)):
                return normalize(raw[wrap])
    items = []  # (id, meta dict)
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, str):
                items.append((x, {}))
            elif isinstance(x, dict):
                gid = _first(x, _ID_KEYS)
                if isinstance(gid, str):
                    items.append((gid, x))
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                items.append((_first(v, _ID_KEYS) if not _ID_RE.match(k) else k, v))
            elif isinstance(v, list) and all(isinstance(s, str) for s in v):  # {collection: [var ids]}
                items.extend((s, {}) for s in v)
            else:
                items.append((k, {"long_name": v} if isinstance(v, str) and not _ID_RE.match(v) else {}))
    out = {}
    for gid, meta in items:
        if not isinstance(gid, str) or not _ID_RE.match(gid.strip()):
            continue
        gid = gid.strip()
        nm = _first(meta, _NAME_KEYS)
        out[PREFIX + gid] = make_entry(
            gid, name=nm if isinstance(nm, str) and nm.strip() and nm != gid else None,
            units=_first(meta, _UNIT_KEYS), definition=_first(meta, _DEF_KEYS),
            step=_first(meta, _STEP_KEYS), start=_first(meta, _START_KEYS), end=_first(meta, _END_KEYS),
            verified=True)  # it is on NASA's list
    return out


def load():
    """{catalog_key: entry}, from giovanni_vars.json if built, else the starter list. Never raises."""
    if os.path.exists(CATALOG_FILE):
        try:
            with open(CATALOG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except (OSError, ValueError):
            pass
    return {PREFIX + gid: make_entry(gid, verified=ok) for gid, ok in STARTER}


def is_built():
    return os.path.exists(CATALOG_FILE)


# ---------------------------------------------------------------------------------------------- build
def _token_from_env():
    path = os.path.join(HERE, ".env")
    tok = os.getenv("EARTHDATA_TOKEN", "")
    if not tok and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip().startswith("EARTHDATA_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    return tok


def build(token=None, session=None):
    import requests
    http = session or requests
    attempts = [{}]
    tok = token or _token_from_env()
    if tok:
        attempts.append({"Authorization": f"Bearer {tok}"})
    last = None
    for i, headers in enumerate(attempts + [None]):
        if headers is None:  # no token yet and the open request failed: ask once
            if tok or not sys.stdin.isatty():
                break
            import getpass
            tok = getpass.getpass("NASA wants a login for the list. Paste your Earthdata token (not saved): ").strip()
            if not tok:
                break
            headers = {"Authorization": f"Bearer {tok}"}
        r = http.get(LIST_URL, headers=headers, timeout=120)
        last = r
        if r.status_code == 200:
            raw = r.json()
            with open(RAW_FILE, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=1)
            cat = normalize(raw)
            if not cat:
                raise RuntimeError(f"NASA answered but I couldn't find variable IDs in it. Saved the answer to "
                                   f"{os.path.basename(RAW_FILE)}; send its first lines to fix the parser.")
            with open(CATALOG_FILE, "w", encoding="utf-8") as f:
                json.dump(cat, f, indent=1, ensure_ascii=False)
            named = sum(1 for v in cat.values() if describe_id(v["giovanni_id"])["known"])
            print(f"Saved {len(cat)} Giovanni variables to {os.path.basename(CATALOG_FILE)} "
                  f"({named} with hand-written names; the rest are named from their IDs).")
            for k in list(cat)[:8]:
                print("  ", cat[k]["giovanni_id"], "->", cat[k]["name"], f"[{cat[k]['units']}]")
            return cat
    code = getattr(last, "status_code", "no answer")
    raise RuntimeError(f"Couldn't download the list (HTTP {code}). If it's 401/403, put EARTHDATA_TOKEN=<token> "
                       f"in .env and run again.")


def rebuild_from_raw():
    """Re-make giovanni_vars.json from the saved NASA answer (after naming improvements; no network)."""
    with open(RAW_FILE, encoding="utf-8") as f:
        cat = normalize(json.load(f))
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=1, ensure_ascii=False)
    named = sum(1 for v in cat.values() if describe_id(v["giovanni_id"])["known"])
    print(f"Rebuilt {len(cat)} Giovanni variables ({named} with readable names) from {os.path.basename(RAW_FILE)}.")
    return cat


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    elif "--rebuild" in sys.argv:
        rebuild_from_raw()
    else:
        cat = load()
        print(f"{len(cat)} variables ({'built list' if is_built() else 'starter list'})")
        for k, v in list(cat.items())[:20]:
            print(f"  {k:50} {v['name']}  [{v['units']}]")
