"""Tests for lstm_numpy.py and forecast_lstm.py. Run:  python test_forecast_lstm.py   (or pytest)"""
import time

import numpy as np
import pandas as pd

import forecast_lstm as F
from lstm_numpy import LSTM, gradient_check


def _series(n_years=40, resid="noise", seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_years * 12)
    if resid == "noise":
        r = rng.normal(0, 0.6, t.size)
    else:  # persistent anomalies: a warm or wet spell lingers for months (AR(1), like real monthly anomalies)
        r = np.zeros(t.size)
        for i in range(1, t.size):
            r[i] = 0.85 * r[i - 1] + rng.normal(0, 0.5)
    return 15 + 0.003 * t + 6 * np.sin(2 * np.pi * t / 12) + r


def test_gradients_exact():
    for seed in range(3):
        assert gradient_check(n_in=2, n_hidden=6, T=9, B=5, seed=seed) < 1e-7


def test_lstm_learns_sequence():
    rng = np.random.default_rng(0)
    x = np.sin(np.arange(400) * 2 * np.pi / 17) + rng.normal(0, 0.1, 400)
    X = np.stack([x[i:i + 24] for i in range(376)])[..., None]
    Y = x[24:, None]
    net = LSTM(1, 12, seed=0)
    net.fit(X[:300], Y[:300], epochs=80)
    assert np.mean((net.predict(X[300:]) - Y[300:]) ** 2) < 0.05


def test_noise_residuals_lstm_adds_little():
    res = F.run(dict(values=_series().tolist(), start="1985-01", horizon=24))
    s = res["scores"]
    assert s[F.LINE]["skill"] > 0.15                                   # the line beats repeating last year
    assert s[F.HYBRID]["rmse"] < 1.15 * s[F.LINE]["rmse"]             # residual design: can't lose badly
    assert "helped" not in res["verdict"]


def test_structured_residuals_lstm_helps_short_range():
    """Lingering anomalies are predictable for a few months, then fade: the LSTM should win early on and not
    lose badly over two years. (Even the ideal model gains little at 2 years on such a series.)"""
    wins = 0
    for seed in (1, 2, 3):
        res = F.run(dict(values=_series(resid="ar", seed=seed).tolist(), start="1985-01", horizon=24, members=2))
        s = res["scores"]
        print("  structured seed", seed, {m: round(v["rmse_short"], 3) for m, v in s.items()},
              "short wins", res["wins_short"], "/", res["n_folds"], "|", res["verdict"][:60])
        wins += s[F.HYBRID]["rmse_short"] < s[F.LINE]["rmse_short"]
        assert s[F.HYBRID]["rmse"] < 1.15 * s[F.LINE]["rmse"]
    assert wins >= 2


def test_speed_on_largest_series():
    t0 = time.time()
    res = F.run(dict(values=_series(n_years=50).tolist(), start="1975-01", horizon=120, members=3, epochs=120))
    dt = time.time() - t0
    print(f"  600 months, 4 backtests + 3 members: {dt:.1f}s")
    assert dt < 30 and len(res["future"][F.HYBRID]) == 120


def test_deterministic():
    p = dict(values=_series().tolist(), start="1985-01", horizon=12, members=2, seed=7)
    assert F.run(p)["future"] == F.run(p)["future"]


def test_bands_widen_and_contain_path():
    res = F.run(dict(values=_series().tolist(), start="1985-01", horizon=96))
    for m in (F.LINE, F.HYBRID):
        lo, hi = (np.array(res["bands"][m][k]) for k in ("lo", "hi"))
        mid = np.array(res["future"][m])
        width = hi - lo
        assert np.all(lo <= mid) and np.all(mid <= hi)
        assert width[-1] > width[0] and np.all(np.diff(width) >= -1e-4)  # values are rounded to 5 decimals


def test_band_coverage_roughly_95():
    """Bands come from backtest errors, so on fresh data from the same process ~95% should fall inside."""
    inside = total = 0
    for seed in range(4):
        y = _series(n_years=44, seed=seed)
        res = F.run(dict(values=y[:-48].tolist(), start="1985-01", horizon=48, members=1))
        lo, hi = np.array(res["bands"][F.LINE]["lo"]), np.array(res["bands"][F.LINE]["hi"])
        inside += int(np.sum((y[-48:] >= lo) & (y[-48:] <= hi)))
        total += 48
    print(f"  95% band coverage on unseen months: {inside / total:.0%}")
    assert 0.80 <= inside / total <= 1.0


def test_validation():
    good = _series().tolist()
    bad = [dict(values=good[:60], start="2000-01"), dict(values=good[:-1] + [np.nan], start="2000-01"),
           dict(values=good, start="2000-13"), dict(start="2000-01"), dict(values=good * 2, start="2000-01")]
    for p in bad:
        try:
            F.validate(p)
            raise AssertionError(f"accepted {list(p)}")
        except ValueError:
            pass
    _, _, cfg = F.validate(dict(values=good, start="1985-01", horizon=999, epochs=999, hidden=999, members=99, folds=99))
    assert (cfg["horizon"], cfg["epochs"], cfg["hidden"], cfg["members"]) == (120, 300, 32, 5)
    assert cfg["folds"] <= 6
    _, _, cfg = F.validate(dict(values=good[:100], start="1985-01", folds=4))  # short record: fewer backtests
    assert cfg["folds"] == 1


def test_run_series_and_figure():
    idx = pd.date_range("1990-01-01", periods=360, freq="MS")
    s = pd.Series(_series(30), idx)
    res = F.run_series(s, horizon=24)
    fig = F.figure(s, res, "Test", "°C")
    assert fig.axes and res["start"] == "1990-01"
    gappy = s.drop(s.index[100])
    try:
        F.run_series(gappy)
        raise AssertionError("accepted a gap")
    except ValueError:
        pass


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            fn()
            print(f"ok   {name} ({time.time() - t0:.1f}s)")
    print("ALL FORECAST TESTS PASSED")
