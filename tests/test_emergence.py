"""Checks for emergence.py. Run: python tests/test_emergence.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, pandas as pd
from planets.analysis import emergence as E
from planets.analysis.trend import analyze

def series(years, slope_per_decade, noise, seed=0, start=1981):
    idx = pd.date_range(f"{start}-01-01", periods=years * 12, freq="MS"); k = np.arange(len(idx))
    rng = np.random.default_rng(seed)
    return pd.Series(10 + slope_per_decade / 120 * k + 5 * np.sin(2 * np.pi * k / 12) + rng.normal(0, noise, len(k)), idx)

def test_strong_trend_is_significant_with_emergence_year():
    e = E.years_to_significance(analyze(series(40, 0.5, 0.5)))
    print("  strong:", e["status"], e["needed_years"], e["detectable_from"])
    assert e["status"] == "significant" and e["needed_years"] < 40 and 1981 < e["detectable_from"] <= 2021

def test_weak_trend_needs_more_years_and_simulation_agrees():
    """Record too short now; extend the same process to the predicted length: it should usually test significant."""
    slope, noise = 0.12, 1.0
    e = E.years_to_significance(analyze(series(15, slope, noise, seed=4)))
    print("  weak:", e["status"], "needed", e["needed_years"], "more", e["more_years"])
    # simulate with the TRUE slope: predicted length from the true parameters vs empirical detection rate
    need_true = E._years_needed(slope / 10, noise, 12)
    need80 = E._years_needed(slope / 10, noise, 12, margin=E.POWER80)
    hits80 = sum(analyze(series(int(round(need80)), slope, noise, seed=s))["metrics"]["ols_p_value"] < 0.05 for s in range(60))
    print(f"  80% target: {need80:.1f} y, detected in {hits80}/60")
    assert 40 <= hits80 <= 57
    hits = sum(analyze(series(int(round(need_true)), slope, noise, seed=s))["metrics"]["ols_p_value"] < 0.05 for s in range(60))
    short = sum(analyze(series(max(3, int(round(need_true * 0.5))), slope, noise, seed=s))["metrics"]["ols_p_value"] < 0.05 for s in range(60))
    print(f"  true-parameter check: needs {need_true:.1f} y; detected in {hits}/60 at that length, {short}/60 at half")
    # at the threshold length the expected t equals t_crit: ~50% detection; at half length far fewer
    assert 20 <= hits <= 42 and short < hits

def test_flat_and_tiny():
    s = series(30, 0.0, 1.0); s[:] = 10 + 5 * np.sin(2 * np.pi * np.arange(len(s)) / 12) + np.random.default_rng(1).normal(0, 1, len(s))
    e = E.years_to_significance(analyze(s))
    assert e["status"] in ("too_slow", "on_track", "significant")
    txt = E.describe(e, "°C", "rising"); assert isinstance(txt, str) and len(txt) > 20

def test_persistence_slows_it():
    idx = pd.date_range("1981-01-01", periods=480, freq="MS"); k = np.arange(480); rng = np.random.default_rng(2)
    r = np.zeros(480)
    for i in range(1, 480): r[i] = 0.7 * r[i - 1] + rng.normal(0, 0.5)
    e = E.years_to_significance(analyze(pd.Series(0.002 * k + r, idx)))
    print("  persistent noise: plain", e["needed_years"], "persistence-aware", e["needed_years_persist"])
    assert e["persistence"] > 0.4 and (e["needed_years_persist"] or 1e9) > e["needed_years"]

def test_season_mode_one_value_per_year():
    e = E.years_to_significance(analyze(series(40, 0.4, 1.0), season="JJA"))
    assert e["needed_years"] is not None and abs(e["record_years"] - 39) < 1.01

def test_describe_all_statuses():
    for st in ("significant", "on_track", "too_slow", "flat"):
        e = dict(status=st, slope_per_decade=0.2, record_years=20, needed_years=30.0, more_years=10.0,
                 needed_years_80=41.0, more_years_80=21.0,
                 needed_years_persist=45.0, more_years_persist=25.0, detectable_from=2010)
        print("  ", st, "->", E.describe(e, "°C", "rising")[:90])

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
    print("ALL EMERGENCE TESTS PASSED")
