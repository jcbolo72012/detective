"""Variable explorer: one table of everything the app can investigate, grouped in plain language."""
import re

import pandas as pd

GROUPS = [  # (group, pattern on name + definition). First match wins.
    ("Temperature", r"temperature|dew|frost|degree days|heating|cooling"),
    ("Water", r"precipitation|precipitable|rain|humidity|wetness|soil|snow|evapo|moisture"),
    ("Wind", r"wind"),
    ("Sun and clouds", r"irradiance|radiation|solar|insolation|sky|cloud|albedo|\buv|photosynthetically|clearness|"
                       r"sun|daylight|zenith|illuminance|longwave|shortwave"),
    ("Air", r"pressure|aerosol|ozone|air mass|density|optical|dust|carbon|sulfur|nitrogen|pollution"),
]
HOURLY = re.compile(r"_\d{2}$|_HR$|Hour-Based|\bat \d{2} Hour\b|hourly|GMT", re.I)


def group_of(name, definition):
    text = f"{name} {definition}"
    return next((g for g, pat in GROUPS if re.search(pat, text, re.I)), "Other")


def build_table(all_vars, curated):
    """all_vars: retrieve.ALL, curated: catalog.VARIABLES -> DataFrame, featured first."""
    rows = []
    for code, v in all_vars.items():
        gio = v.get("provider") == "giovanni"
        rows.append(dict(
            Variable=v["name"], Code=code, Units=v.get("units", ""),
            Source="NASA Giovanni" if gio else "NASA POWER",
            Group=group_of(v["name"], f"{v.get('definition', '')} {' '.join(v.get('keywords') or [])}"),
            Featured=code in curated,
            Hourly=(not gio) and bool(HOURLY.search(code) or HOURLY.search(v["name"])) and code not in curated,
            Definition=v.get("definition", "")))
    df = pd.DataFrame(rows)
    return df.sort_values(["Featured", "Group", "Variable"], ascending=[False, True, True]).reset_index(drop=True)


def filter_table(df, group="All", text="", show_hourly=False, ranked_codes=None, source=None):
    out = df if show_hourly else df[~df["Hourly"]]
    if source:
        out = out[out["Source"] == source]
    if group == "Featured":
        out = out[out["Featured"]]
    elif group != "All":
        out = out[out["Group"] == group]
    if ranked_codes is not None:  # search results: keep their order
        out = out.set_index("Code").reindex([c for c in ranked_codes if c in set(out["Code"])]).reset_index()
    elif text:
        t = text.lower()
        out = out[out["Variable"].str.lower().str.contains(t, regex=False)
                  | out["Definition"].str.lower().str.contains(t, regex=False)
                  | out["Code"].str.lower().str.contains(t, regex=False)]
    return out
