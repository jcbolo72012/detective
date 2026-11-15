"""Data check and gap-aware Sen's slope. Run: python tests/test_quality.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from planets.analysis import quality, trend


def monthly(n_years=45, start=1981, slope_per_decade=0.5, seed=0):
    idx = pd.date_range(f"{start}-01-01", periods=n_years * 12, freq="MS")
    k = np.arange(len(idx))
    rng = np.random.default_rng(seed)
    return pd.Series(15 + slope_per_decade / 120 * k + 8 * np.sin(2 * np.pi * k / 12) + rng.normal(0, 0.6, len(k)), index=idx)


def test_complete_is_green():
    s, r = quality.check(monthly(), "C", 1981, 2025)
    assert r["level"] == "green" and r["received"] == r["expected"] == 540 and not r["removed"], r
    assert len(s) == 540


def test_impossible_value_removed():
    raw = monthly()
    raw.iloc[100] = 80.0     # 80 °C monthly temperature: corrupt
    raw.iloc[200] = -150.0
    s, r = quality.check(raw, "°C", 1981, 2025)
    assert len(r["removed"]) == 2 and r["level"] == "amber", r
    assert s.max() < 80 and s.min() > -150 and len(s) == 538


def test_precip_and_humidity_bounds():
    rain = pd.Series([1.0, -5.0, 3.0], index=pd.date_range("2000-01-01", periods=3, freq="MS"))
    _, r = quality.check(rain, "mm/day", 2000, 2000)
    assert [x["value"] for x in r["removed"]] == [-5.0], r
    rh = pd.Series([50.0, 120.0], index=pd.date_range("2000-01-01", periods=2, freq="MS"))
    _, r = quality.check(rh, "%", 2000, 2000)
    assert [x["value"] for x in r["removed"]] == [120.0], r


def test_unknown_units_not_range_checked():
    s = pd.Series([-1e3, 1e3], index=pd.date_range("2000-01-01", periods=2, freq="MS"))
    _, r = quality.check(s, "W m-2", 2000, 2000)
    assert r["bounds"] is None and not r["removed"]


def test_gaps_and_truncation():
    raw = monthly()
    raw = raw.drop(raw.index[120:135])           # 15-month gap
    _, r = quality.check(raw, "C", 1981, 2025)
    assert r["longest_gap"] == 15 and r["level"] == "amber", r
    _, r = quality.check(monthly()["1981":"2021-12"], "C", 1981, 2025)   # stops 4 years early
    assert r["ends_early"] == 48 and r["level"] == "amber" and any("ends 2021-12" in i for i in r["issues"]), r
    _, r = quality.check(monthly()["1981":"2010-12"], "C", 1981, 2025)   # 2/3 of the record
    assert r["level"] == "red", r


def test_nan_and_duplicates_handled():
    raw = monthly()
    raw.iloc[5] = np.nan
    raw = pd.concat([raw, raw.iloc[[10]]])
    s, r = quality.check(raw, "C", 1981, 2025)
    assert r["received"] == 539 and s.index.is_unique


def test_sen_slope_unchanged_without_gaps():
    s = monthly(seed=3)
    m = trend.analyze(s)["metrics"]
    y = trend.deseasonalize(s)[0].values
    old = trend.mann_kendall(y)["sen_slope_per_step"] * 120  # consecutive steps = the old behaviour
    assert abs(m["sen_slope_per_decade"] - old) < 1e-12


def test_sen_slope_correct_with_gaps():
    s = monthly(slope_per_decade=0.5, seed=4)
    rng = np.random.default_rng(1)
    holes = s.drop(s.index[np.sort(rng.choice(len(s), 160, replace=False))])  # 30% of months missing
    holes = holes.drop(holes.index[200:260])                                  # plus a long gap
    m = trend.analyze(holes)["metrics"]
    y = trend.deseasonalize(holes)[0].values
    naive = trend.mann_kendall(y)["sen_slope_per_step"] * 120   # treats the gaps as if they weren't there
    assert abs(m["sen_slope_per_decade"] - 0.5) < 0.06, m["sen_slope_per_decade"]
    assert abs(naive - 0.5) > abs(m["sen_slope_per_decade"] - 0.5), (naive, m["sen_slope_per_decade"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("ALL QUALITY TESTS PASSED")
