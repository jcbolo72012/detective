"""Trend statistics for one monthly series: everything the app's verdicts come from.

  analyze(series, season=None, keep_cycle=False) -> {series, anom, fit, metrics}
  - seasonal cycle removed (monthly anomalies), or one value per year for a season (e.g. "JJA")
  - keep_cycle=True (all months only): the seasonal cycle stays in, so the tests treat it as noise
  - OLS slope per decade with 95% CI and t-test p; Mann-Kendall test and Sen's slope; lag-1 autocorrelation
  - significant = both tests p < 0.05

Also: load_giovanni_csv() for Giovanni "Time Series, Area-Averaged" CSV exports.
"""
import io

import numpy as np
import pandas as pd
from scipy import stats


def load_giovanni_csv(path):
    """Giovanni CSVs start with metadata lines (Title, Bounding Box, Fill Value...).
    The data starts at the line beginning with 'time'."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    fill = None
    start = None
    for i, line in enumerate(lines):
        low = line.lower()
        if low.startswith("fill value"):
            try:
                fill = float(line.split(",")[-1])
            except ValueError:
                pass
        if low.startswith("time"):
            start = i
            break
    if start is None:
        raise ValueError("Could not find a header line starting with 'time'")
    df = pd.read_csv(io.StringIO("".join(lines[start:])))
    df.columns = ["time", "value"] + list(df.columns[2:])
    df["time"] = pd.to_datetime(df["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if fill is not None:
        df.loc[np.isclose(df["value"], fill), "value"] = np.nan
    s = df.set_index("time")["value"]
    return s.resample("MS").mean().dropna()  # monthly


def synthetic_series(n_years=22, seed=0):
    """Fake monthly data: trend + seasonal cycle + noise, for testing tonight."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2003-01-01", periods=n_years * 12, freq="MS")
    k = np.arange(len(t))
    y = 15 + 0.004 * k + 3 * np.sin(2 * np.pi * k / 12) + rng.normal(0, 0.6, len(t))
    return pd.Series(y, index=t)


# ---------------------------------------------------------------- 2. seasonality
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def parse_season(code):
    """'SON' or a month list like '9,10' or '9-11' -> list of month numbers."""
    code = code.strip().upper()
    if code in SEASONS:
        return SEASONS[code]
    if "-" in code:
        a, b = map(int, code.split("-"))
        return list(range(a, b + 1)) if a <= b else list(range(a, 13)) + list(range(1, b + 1))
    return [int(m) for m in code.split(",")]


def seasonal_annual(s, months):
    """One value per year = mean of the chosen months. Years missing any month are dropped.
    Seasons that wrap the new year (e.g. DJF) are labelled by the year of the later months."""
    sub = s[s.index.month.isin(months)]
    wraps = months != sorted(months)
    year = sub.index.year + ((sub.index.month >= months[0]) & wraps).astype(int) if wraps else sub.index.year
    g = sub.groupby(year)
    out = g.mean()[g.count() == len(months)]
    out.index = pd.to_datetime([f"{y}-01-01" for y in out.index])
    return out


def deseasonalize(s):
    clim = s.groupby(s.index.month).mean()
    anom = s - clim.reindex(s.index.month).values
    return anom, clim


# ---------------------------------------------------------------- 3. trend stats
def ols_trend(s):
    """Slope per decade, 95% CI, p-value (t-test on slope)."""
    x = (s.index - s.index[0]).days.values / 365.25  # years
    r = stats.linregress(x, s.values)
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    return {
        "ols_slope_per_decade": r.slope * 10,
        "ols_ci95_low": (r.slope - tcrit * r.stderr) * 10,
        "ols_ci95_high": (r.slope + tcrit * r.stderr) * 10,
        "ols_p_value": r.pvalue,
        "ols_r2": r.rvalue ** 2,
    }, r


def mann_kendall(y):
    """Mann-Kendall trend test (no ties correction) + Sen's slope (per step)."""
    y = np.asarray(y)
    n = len(y)
    diff = y[None, :] - y[:, None]
    iu = np.triu_indices(n, 1)
    S = np.sign(diff[iu]).sum()
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z = (S - np.sign(S)) / np.sqrt(var_s) if S != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slopes = diff[iu] / (iu[1] - iu[0])
    return {"mk_S": float(S), "mk_z": float(z), "mk_p_value": float(p),
            "sen_slope_per_step": float(np.median(slopes))}


def lag1_autocorr(y):
    y = np.asarray(y) - np.mean(y)
    return float(np.sum(y[1:] * y[:-1]) / np.sum(y * y))


# ---------------------------------------------------------------- 4. models
def analyze(s, season=None, keep_cycle=False):
    """Monthly pd.Series -> dict with the analysed series, anomalies, fitted line and metrics.
    All numbers the app (and the AI explanation) shows come from here.
    keep_cycle (all months only): test the raw monthly values, seasonal cycle included. The summer-winter swing
    then counts as noise, so the tests are weaker and their assumptions hold less well; the default removes it."""
    keep_cycle = bool(keep_cycle and not season)  # one value per year has no cycle to keep
    if season:
        s = seasonal_annual(s, parse_season(season))
        anom, steps_per_year = s - s.mean(), 1
    elif keep_cycle:
        anom, steps_per_year = s - s.mean(), 12
    else:
        anom, steps_per_year = deseasonalize(s)[0], 12
    if len(s) < 8:
        raise ValueError(f"Only {len(s)} points after processing; need a longer record.")
    ols, r = ols_trend(anom)
    mk = mann_kendall(anom.values)
    mk["sen_slope_per_decade"] = mk.pop("sen_slope_per_step") * 10 * steps_per_year
    x = (anom.index - anom.index[0]).days.values / 365.25
    fit = r.intercept + r.slope * x
    m = {**ols, **mk, "lag1_autocorr_resid": lag1_autocorr(anom.values - fit),
         "n_points": int(len(s)), "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
         "mean": float(s.mean()), "seasonal_cycle_kept": keep_cycle}
    m["significant"] = bool(m["ols_p_value"] < 0.05 and m["mk_p_value"] < 0.05)
    m["tests_agree"] = bool((m["ols_p_value"] < 0.05) == (m["mk_p_value"] < 0.05))
    m["direction"] = "increasing" if m["sen_slope_per_decade"] > 0 else "decreasing"
    m = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in m.items()}
    return {"series": s, "anom": anom, "fit": pd.Series(fit, index=anom.index), "metrics": m}
