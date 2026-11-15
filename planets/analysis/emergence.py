"""How many years of data a trend needs before it stands out from the noise ("years until significant").

For a straight-line trend b (per year) in noise with standard deviation sigma, measured with p points a year over
T years (n = T*p + 1 points), the slope's standard error is

    SE(b) = sigma / sqrt(Sxx),   Sxx = n (n^2 - 1) / (12 p^2)   (years^2)

and the t-test calls it significant once |b| / SE(b) >= t_crit (about 2). We find the smallest T where that
holds, using the trend and noise measured so far. This is the same test the app uses (p < 0.05), so the answer
agrees with the verdict on screen. Month-to-month persistence (lag-1 autocorrelation r) makes neighbouring points
partly repeat each other; the "persistence-aware" figure widens SE by sqrt((1 + r) / (1 - r)), which is slower
and more realistic.

Assumes the trend continues at its current estimated rate and the noise stays the same. Both are estimates.
"""
import numpy as np
from scipy import stats

MAX_YEARS = 300   # beyond this we just say "more than a few centuries"
POWER80 = 0.8416  # z for 80% power: how far past the threshold the expected t must be to be detected 4 times in 5


def _years_needed(slope, sigma, ppy, inflate=1.0, alpha=0.05, margin=0.0):
    """Smallest record length (years) at which |slope|/SE >= t_crit + margin, or None if beyond MAX_YEARS.
    margin 0: even odds that a record this long tests significant. margin 0.84: an 80% chance (80% power)."""
    if slope == 0 or not np.isfinite(slope) or sigma <= 0:
        return None
    lo, hi = 1, int(MAX_YEARS * ppy) + 1
    def ok(n):
        sxx = n * (n * n - 1) / (12.0 * ppy * ppy)
        se = inflate * sigma / np.sqrt(sxx)
        return abs(slope) / se >= stats.t.ppf(1 - alpha / 2, max(n - 2, 1)) + margin
    if not ok(hi):
        return None
    lo = 3
    while lo < hi:  # |slope|/SE grows with n much faster than t_crit falls, so this is monotone
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return (lo - 1) / ppy


def years_to_significance(res):
    """res = trend_detective.analyze(...) result. Returns a JSON-safe dict:
    status: 'significant' | 'on_track' | 'too_slow' | 'flat'
    record_years, needed_years, more_years, needed_years_persist, more_years_persist, detectable_from (year)."""
    anom, fit = res["anom"], res["fit"]
    x = (anom.index - anom.index[0]).days.values / 365.25
    y = anom.values.astype(float)
    n = len(y)
    record = float(x[-1] - x[0])
    ppy = (n - 1) / record if record > 0 else 12.0
    slope = float(np.polyfit(x, y, 1)[0])                      # per year (same line as the OLS test)
    resid = y - fit.values
    sigma = float(np.std(resid, ddof=2))
    r = float(np.clip(res["metrics"].get("lag1_autocorr_resid", 0.0), 0.0, 0.95))
    inflate = float(np.sqrt((1 + r) / (1 - r)))

    need = _years_needed(slope, sigma, ppy)
    need_p = _years_needed(slope, sigma, ppy, inflate)
    need_80 = _years_needed(slope, sigma, ppy, margin=POWER80)
    start_year = anom.index[0].year
    out = dict(record_years=round(record, 1), slope_per_decade=slope * 10, noise_sd=sigma, persistence=r,
               needed_years=None if need is None else round(need, 1),
               needed_years_persist=None if need_p is None else round(need_p, 1),
               needed_years_80=None if need_80 is None else round(need_80, 1), more_years_80=None,
               more_years=None, more_years_persist=None, detectable_from=None,
               significant=bool(res["metrics"].get("significant")))
    if need is None:
        out["status"] = "flat" if abs(slope) * 10 < 1e-12 else "too_slow"
        return out
    out["more_years"] = round(max(0.0, need - record), 1)
    if need_80 is not None:
        out["more_years_80"] = round(max(0.0, need_80 - record), 1)
    if need_p is not None:
        out["more_years_persist"] = round(max(0.0, need_p - record), 1)
    if out["significant"] or need <= record:
        out["status"] = "significant"
        out["detectable_from"] = int(start_year + int(np.ceil(need)))
    else:
        out["status"] = "on_track"
    return out


def _yrs(v):
    if v is None:
        return f"more than {MAX_YEARS} years"
    if v < 1:
        return "less than a year"
    return f"about {v:.0f} year{'s' if round(v) != 1 else ''}"


def describe(e, units, direction):
    """One plain-language sentence or two for the app (and the check on any AI wording)."""
    rate = f"{abs(e['slope_per_decade']):.2g} {units} per decade"
    article = "an" if direction[:1].lower() in "aeiou" else "a"
    if e["status"] == "flat":
        return "There's no measurable trend, so no amount of extra data would make one appear."
    if e["status"] == "too_slow":
        return (f"The estimated change ({rate}) is so small next to the ups and downs that it would take "
                f"{_yrs(None)} of data to stand out. For practical purposes: no detectable trend.")
    if e["status"] == "significant":
        t = (f"Already significant. With this much natural variation, {article} {direction} trend of {rate} needs "
             f"{_yrs(e['needed_years'])} of data to have even odds of standing out, so it became detectable around "
             f"{e['detectable_from']} (the record has {e['record_years']:.0f} years).")
        if e["more_years_persist"]:  # persistence-aware estimate says the record is still too short
            t += (f" Allowing for month-to-month persistence it needs {_yrs(e['needed_years_persist'])}, so "
                  f"treat the result as borderline.")
        return t
    t = (f"Not significant yet. If the trend keeps going at {rate}, a record of {_yrs(e['needed_years'])} "
         f"would have even odds of showing it ({_yrs(e['more_years'])} more than the {e['record_years']:.0f} we "
         f"have), and {_yrs(e['needed_years_80'])} would show it 4 times in 5 "
         f"({_yrs(e['more_years_80'] if e['needed_years_80'] else None)} more).")
    if e["needed_years_persist"] and e["needed_years_persist"] > e["needed_years"] + 1:
        t += (f" Allowing for month-to-month persistence, even odds needs "
              f"{_yrs(e['more_years_persist'])} more.")
    t += " The estimated rate itself is uncertain, so this is a rough guide, not a prediction."
    return t
