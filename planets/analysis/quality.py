"""Data check for one monthly series, run before any statistics.

check(series, units, start_year, end_year) -> (clean_series, report)
  - physical range check: values outside what the units allow (e.g. 80 °C, -5 mm/day of rain, 120 % humidity)
    are removed and listed
  - completeness: months expected vs received, the longest gap, and whether the record starts later or ends
    earlier than asked (e.g. a satellite product that stopped)
  - a traffic-light level: green (complete, nothing removed), amber (gaps, removals or a short record), red (under
    80 % of the expected months)
Missing values that NASA marks with fill codes (-999, -9999, declared fill values) are already dropped by the
source readers; they show up here as missing months.
"""
import numpy as np
import pandas as pd

# Limits wide enough for any real value on Earth, tight enough to catch corrupt ones. Units not listed (indices,
# fluxes that can be negative, wind components, anomalies) are not range-checked.
_BOUNDS = {
    "C": (-95, 65), "°C": (-95, 65), "K": (175, 340),
    "%": (0, 100), "m3 m-3": (0, 1),
    "mm/day": (0, 2000),
    "m/s": (0, 150),  # NASA POWER wind *speed*; Giovanni's "m s-1" includes signed wind components, so not checked
    "kPa": (0, 110), "hPa": (0, 1100), "Pa": (0, 110000),
    "MJ/m^2/day": (0, 100),
    "g/kg": (0, 100), "g kg-1": (0, 100), "kg kg-1": (0, 0.1),
    "ppm": (100, 2000), "DU": (0, 1000), "Dobson": (0, 1000), "Dobsons": (0, 1000),
    "Degrees": (-360, 360),
}
RED_COVERAGE, GREEN_COVERAGE, LONG_GAP_MONTHS = 0.80, 0.95, 12


def bounds(units):
    return _BOUNDS.get((units or "").strip())


def _longest_run(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def check(series, units, start_year, end_year):
    """Returns (clean_series, report). report is JSON-friendly; see module docstring."""
    s = series.dropna()
    s = s[~s.index.duplicated()].sort_index()
    lim = bounds(units)
    removed = []
    if lim is not None and len(s):
        bad = (s < lim[0]) | (s > lim[1])
        removed = [{"month": f"{d:%Y-%m}", "value": round(float(v), 4)} for d, v in s[bad].items()]
        s = s[~bad]

    window = pd.date_range(f"{int(start_year)}-01-01", f"{int(end_year)}-12-01", freq="MS")
    months = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    present = np.asarray(window.isin(months))
    expected, received = len(window), int(present.sum())
    coverage = received / expected if expected else 0.0
    if received:
        first, last = int(np.argmax(present)), int(len(present) - 1 - np.argmax(present[::-1]))
        inner_gap = _longest_run(~present[first:last + 1])
        starts_late, ends_early = first, len(present) - 1 - last
        first_month, last_month = f"{window[first]:%Y-%m}", f"{window[last]:%Y-%m}"
    else:
        inner_gap, starts_late, ends_early, first_month, last_month = 0, expected, 0, None, None

    issues = []
    if expected - received:
        issues.append(f"{expected - received} of {expected} months missing (no valid value from NASA)")
    if inner_gap >= 2:
        issues.append(f"longest gap {inner_gap} months")
    if starts_late >= 3:
        issues.append(f"record starts {first_month}, later than the {window[0]:%Y-%m} you asked for")
    if ends_early >= 3:
        issues.append(f"record ends {last_month}, earlier than the {window[-1]:%Y-%m} you asked for")
    if removed:
        issues.append(f"{len(removed)} value(s) outside the physically possible range removed")

    if coverage < RED_COVERAGE:
        level = "red"
    elif (coverage < GREEN_COVERAGE or inner_gap >= LONG_GAP_MONTHS or removed
          or starts_late >= LONG_GAP_MONTHS or ends_early >= LONG_GAP_MONTHS):
        level = "amber"
    else:
        level = "green"
    report = dict(level=level, expected=expected, received=received, coverage=round(coverage, 4),
                  longest_gap=int(inner_gap), starts_late=int(starts_late), ends_early=int(ends_early),
                  first=first_month, last=last_month, removed=removed, bounds=list(lim) if lim else None,
                  issues=issues)
    return s, report


def headline(report):
    """One short line for the badge."""
    r = report
    base = f"{r['received']} of {r['expected']} months"
    if r["level"] == "green":
        return base + " · complete" + (" · range-checked" if r["bounds"] else "")
    rest = [i for i in r["issues"] if "months missing" not in i and "physically possible" not in i]  # shown separately
    missing = r["expected"] - r["received"]
    return base + (f" ({missing} missing)" if missing else "") + ("; " + "; ".join(rest) if rest else "")
