"""Honest LSTM forecasting for one monthly series, in NumPy (lstm_numpy.py). No PyTorch, runs in the app.

What it does, in plain words:
  1. The straight line does the obvious part: trend + seasonal cycle (least squares, two harmonics).
  2. The LSTM only learns what the line misses (the leftovers, or residuals), so "line + LSTM" can't lose to the
     line just by forgetting the seasons, and any gain is the LSTM's own.
  3. Tested from several starting points (rolling origin): cut the record at 4 dates, forecast the next 24
     months each time, compare with what actually happened. One split on climate data is mostly luck.
  4. Skill: how much better each model is than "repeat last year" (0% = no better, 100% = perfect,
     negative = worse). Plus how many of the tests the LSTM beat the line, over two years and over the first
     6 months separately: leftover patterns (a warm spell that lingers) usually only help for a few months.
  5. Bands: built from how wrong each model actually was in those tests, at each lead time, widening further
     out. A few LSTMs with different random starts are averaged, and their disagreement widens the band too.

    res = run_series(series, horizon=60)       # series: monthly pandas Series, no gaps
    fig = figure(series, res, "Air temperature, Phoenix", "°C")

Self-test:  python forecast_lstm.py
"""
import time

import numpy as np

from lstm_numpy import LSTM

LIMITS = dict(min_points=96, max_points=600, max_horizon=120, max_epochs=300, max_hidden=32, max_members=5,
              max_folds=6)
WINDOW = 24       # months the LSTM sees before each prediction
TEST_LEN = 24     # months forecast in each backtest
SHORT = 6         # "the next few months": what an LSTM can realistically add is usually here
FOLD_STEP = 12    # months between backtest starting points
MIN_TRAIN = 72    # months of training data a backtest needs
MODELS = ("Line (trend + seasons)", "Line + LSTM", "Repeat last year")
LINE, HYBRID, NAIVE = MODELS


# ------------------------------------------------------------------------------------------------ input
def validate(payload):
    """Clean, clamped settings, or ValueError with a message a user can act on."""
    try:
        vals = np.array([float(v) for v in payload["values"]])
        start = str(payload["start"])
        y0, m0 = int(start[:4]), int(start[5:7])
        cfg = dict(horizon=int(payload.get("horizon", 60)), epochs=int(payload.get("epochs", 120)),
                   hidden=int(payload.get("hidden", 16)), members=int(payload.get("members", 3)),
                   folds=int(payload.get("folds", 4)), seed=int(payload.get("seed", 0)))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Expected {values: [...], start: 'YYYY-MM', horizon, epochs, hidden, members, folds}.")
    if not (1 <= m0 <= 12 and 1800 <= y0 <= 2200):
        raise ValueError("start must look like 2001-01.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("The series has gaps or invalid values; the forecast needs an unbroken monthly record.")
    L = LIMITS
    if not (L["min_points"] <= len(vals) <= L["max_points"]):
        raise ValueError(f"Need between {L['min_points']} and {L['max_points']} monthly values "
                         f"({L['min_points'] // 12} to {L['max_points'] // 12} years); this series has {len(vals)}.")
    cfg.update(horizon=int(np.clip(cfg["horizon"], 1, L["max_horizon"])),
               epochs=int(np.clip(cfg["epochs"], 10, L["max_epochs"])),
               hidden=int(np.clip(cfg["hidden"], 4, L["max_hidden"])),
               members=int(np.clip(cfg["members"], 1, L["max_members"])))
    # as many folds as the record allows (each needs MIN_TRAIN months before it)
    possible = (len(vals) - TEST_LEN - MIN_TRAIN) // FOLD_STEP + 1
    cfg["folds"] = int(np.clip(cfg["folds"], 1, min(L["max_folds"], possible)))
    return vals, (y0, m0), cfg


# ------------------------------------------------------------------------------------------------ models
def _design(k, k0):
    t = (k - k0) / 120.0  # decades
    ph = 2 * np.pi * (k % 12) / 12
    return np.column_stack([np.ones_like(t, dtype=float), t, np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)])


def fit_line(k, y):
    """Least-squares trend + two seasonal harmonics. Returns predict(k)."""
    k0 = k[0]
    beta, *_ = np.linalg.lstsq(_design(k, k0), y, rcond=None)
    return lambda kk: _design(np.asarray(kk), k0) @ beta


def _windows(z, w=WINDOW):
    X = np.stack([z[i:i + w] for i in range(len(z) - w)])[..., None]
    return X, z[w:, None]


def fit_residual_lstm(r, epochs, hidden, seed):
    """Train on residuals (standardised), the last 10% of windows held out for early stopping.
    Returns predict(n): the next n residuals, fed back one step at a time."""
    mu, sd = float(r.mean()), float(r.std()) or 1.0
    z = (r - mu) / sd
    X, Y = _windows(z)
    n_val = max(1, len(X) // 10)
    net = LSTM(1, hidden, seed=seed)
    net.fit(X[:-n_val], Y[:-n_val], epochs=epochs, lr=0.01, batch=64, seed=seed, patience=15,
            X_val=X[-n_val:], Y_val=Y[-n_val:])

    def predict(n):
        hist = list(z[-WINDOW:])
        out = []
        for _ in range(n):
            p = float(net.predict(np.array(hist[-WINDOW:])[None, :, None])[0, 0])
            out.append(p)
            hist.append(p)
        return np.array(out) * sd + mu

    return predict


def forecast_paths(k, y, n_ahead, cfg, members=1, seed=0):
    """Paths for the n_ahead months after k[-1]: {model: values}, plus the LSTM members' spread."""
    kf = k[-1] + 1 + np.arange(n_ahead)
    line = fit_line(k, y)
    base = line(kf)
    resid = y - line(k)
    runs = np.array([fit_residual_lstm(resid, cfg["epochs"], cfg["hidden"], seed + m)(n_ahead) for m in range(members)])
    return {LINE: base, HYBRID: base + runs.mean(axis=0), NAIVE: np.resize(y[-12:], n_ahead)}, runs.std(axis=0)


# ------------------------------------------------------------------------------------------------ evaluation
def _rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def backtest(k, y, cfg):
    """Forecast TEST_LEN months from cfg['folds'] starting points; errors per model, fold and lead."""
    n = len(y)
    origins = [n - TEST_LEN - FOLD_STEP * (cfg["folds"] - 1 - j) for j in range(cfg["folds"])]
    folds, lead_err = [], {m: [] for m in MODELS}
    for j, o in enumerate(origins):
        paths, _ = forecast_paths(k[:o], y[:o], TEST_LEN, cfg, members=1, seed=cfg["seed"] + 100 * j)
        actual = y[o:o + TEST_LEN]
        fold = {"origin": int(o), "rmse": {}, "mae": {}, "rmse_short": {}}
        for m, p in paths.items():
            e = actual - p
            fold["rmse"][m], fold["mae"][m] = _rmse(e), float(np.mean(np.abs(e)))
            fold["rmse_short"][m] = _rmse(e[:SHORT])
            lead_err[m].append(e)
        folds.append(fold)
        last = (o, paths)
    return folds, {m: np.array(v) for m, v in lead_err.items()}, last


def _band_sigma(lead_errors, horizon, extra=None):
    """Typical error at each lead from the backtests, smoothed; beyond TEST_LEN it keeps growing like sqrt(lead)."""
    rms = np.sqrt(np.mean(lead_errors ** 2, axis=0))                  # (TEST_LEN,)
    rms = np.maximum.accumulate(np.convolve(np.pad(rms, 2, mode="edge"), np.ones(5) / 5, "valid"))
    leads = np.arange(1, horizon + 1)
    sig = np.where(leads <= TEST_LEN, rms[np.minimum(leads, TEST_LEN) - 1], rms[-1] * np.sqrt(leads / TEST_LEN))
    if extra is not None:
        sig = np.sqrt(sig ** 2 + extra ** 2)
    return np.maximum.accumulate(sig)  # uncertainty never shrinks further out


# ------------------------------------------------------------------------------------------------ main entry
def run(payload):
    """payload: {values, start 'YYYY-MM', horizon, epochs, hidden, members, folds, seed}. Returns a JSON-safe dict."""
    t0 = time.time()
    vals, start, cfg = validate(payload)
    k = start[0] * 12 + (start[1] - 1) + np.arange(len(vals))
    folds, lead_err, (o_last, p_last) = backtest(k, vals, cfg)

    mean_rmse = {m: float(np.mean([f["rmse"][m] for f in folds])) for m in MODELS}
    mean_mae = {m: float(np.mean([f["mae"][m] for f in folds])) for m in MODELS}
    short_rmse = {m: float(np.sqrt(np.mean(lead_err[m][:, :SHORT] ** 2))) for m in MODELS}
    skill = {m: 1 - mean_rmse[m] / mean_rmse[NAIVE] if mean_rmse[NAIVE] > 0 else 0.0 for m in MODELS}
    wins = int(sum(f["rmse"][HYBRID] < f["rmse"][LINE] for f in folds))
    wins_short = int(sum(f["rmse_short"][HYBRID] < f["rmse_short"][LINE] for f in folds))

    future, spread = forecast_paths(k, vals, cfg["horizon"], cfg, members=cfg["members"], seed=cfg["seed"])
    bands = {}
    for m in (LINE, HYBRID):
        s = _band_sigma(lead_err[m], cfg["horizon"], spread if m == HYBRID else None)
        bands[m] = {"lo": (future[m] - 1.96 * s).tolist(), "hi": (future[m] + 1.96 * s).tolist()}

    best = min(MODELS, key=lambda m: mean_rmse[m])
    gain = 1 - mean_rmse[HYBRID] / mean_rmse[LINE] if mean_rmse[LINE] > 0 else 0.0
    gain_short = 1 - short_rmse[HYBRID] / short_rmse[LINE] if short_rmse[LINE] > 0 else 0.0
    nf = len(folds)
    if gain > 0.03 and wins >= max(1, int(np.ceil(0.6 * nf))):
        verdict = (f"The LSTM helped: line + LSTM beat the line alone in {wins} of {nf} two-year tests, cutting the "
                   f"typical error by {gain:.0%}. It found patterns in the ups and downs that the line can't see.")
    elif gain_short > 0.05 and wins_short >= max(1, int(np.ceil(0.6 * nf))):
        verdict = (f"The LSTM helped only for the next few months: over the first {SHORT} months it cut the typical "
                   f"error by {gain_short:.0%} (better in {wins_short} of {nf} tests), but over two years it was "
                   f"{'no better than' if gain > -0.05 else 'worse than'} the line. That's normal: an unusually warm "
                   f"or wet spell carries on for a while, then fades, and beyond that only the trend and seasons "
                   f"are predictable.")
    elif gain > -0.05:
        verdict = (f"The LSTM added little: line + LSTM beat the line alone in {wins} of {nf} tests, and the typical "
                   f"error changed by {-gain:+.0%}. The trend and seasons explain almost everything predictable here.")
    else:
        verdict = (f"The LSTM made forecasts worse: it beat the line alone in only {wins} of {nf} tests, and the "
                   f"typical error grew by {-gain:.0%}. What's left after the trend and seasons looks like noise "
                   f"to it, so trust the line.")
    if skill[LINE] < 0.05:
        verdict += " Note that even the line barely beats repeating last year, so treat any projection loosely."

    r = lambda a: [round(float(v), 5) for v in a]
    return dict(
        start=f"{start[0]:04d}-{start[1]:02d}", n_points=len(vals), horizon=cfg["horizon"], settings=cfg,
        models=list(MODELS), folds=folds, wins=wins, wins_short=wins_short, n_folds=nf, short_months=SHORT,
        scores={m: dict(rmse=mean_rmse[m], mae=mean_mae[m], skill=skill[m], rmse_short=short_rmse[m])
                for m in MODELS},
        best=best, verdict=verdict,
        future={m: r(v) for m, v in future.items()},
        bands={m: {"lo": r(b["lo"]), "hi": r(b["hi"])} for m, b in bands.items()},
        last_test={"origin": int(o_last), "n": TEST_LEN, "pred": {m: r(v) for m, v in p_last.items()}},
        seconds=round(time.time() - t0, 2),
    )


def run_series(series, horizon=60, **settings):
    """run() for a monthly pandas Series (index = dates). Raises ValueError on gaps."""
    import pandas as pd
    s = series.dropna()
    per = s.index.to_period("M")
    if len(per.unique()) != len(per) or (len(per) and (per[-1] - per[0]).n != len(per) - 1):
        raise ValueError("The forecast needs an unbroken monthly series (this one has gaps).")
    return run(dict(values=s.values.tolist(), start=str(per[0]), horizon=horizon, **settings))


def figure(series, res, title, units, years_shown=15):
    """Two panels: the record and projections as measured, then the same with the seasonal cycle removed,
    where the line and the LSTM visibly differ. Dotted: the last backtest (forecast vs what happened)."""
    import matplotlib.pyplot as plt
    import pandas as pd
    s = series.dropna()
    p0 = pd.Period(res["start"], "M")
    lt = res["last_test"]
    t_idx = pd.period_range(p0 + lt["origin"], periods=lt["n"], freq="M").to_timestamp()
    f_idx = pd.period_range(p0 + res["n_points"], periods=res["horizon"], freq="M").to_timestamp()
    clim = s.groupby(s.index.month).mean()                       # normal value for each calendar month
    norm = lambda idx: clim.reindex(idx.month).values
    colors = {LINE: "#33658a", HYBRID: "#d1495b"}
    cut = s.index[-1] - pd.DateOffset(years=years_shown)
    shown = s[s.index >= cut]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    for ax, anomaly in zip(axes, (False, True)):
        sub = (lambda idx, v: np.asarray(v) - norm(idx)) if anomaly else (lambda idx, v: np.asarray(v))
        ax.plot(shown.index, sub(shown.index, shown.values), color="#444", lw=0.9, label="observed")
        ax.axvspan(t_idx[0], t_idx[-1], color="#f6d55c", alpha=0.2, label="last backtest")
        for m in (LINE, HYBRID):
            ax.plot(t_idx, sub(t_idx, lt["pred"][m]), color=colors[m], lw=1.1, ls=":")
            ax.fill_between(f_idx, sub(f_idx, res["bands"][m]["lo"]), sub(f_idx, res["bands"][m]["hi"]),
                            color=colors[m], alpha=0.12, lw=0)
            ax.plot(f_idx, sub(f_idx, res["future"][m]), color=colors[m], lw=1.6, label=m)
        if anomaly:
            ax.axhline(0, color="#999", lw=0.6)
        ax.set_ylabel(f"{units} above normal" if anomaly else units, fontsize=9)
        ax.set_title("Seasonal cycle removed: where the LSTM can add something" if anomaly else title,
                     loc="left", fontsize=10 if anomaly else 11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7.5, loc="upper left", ncol=4, frameon=False)
    fig.text(0.01, 0.005, "Shaded: 95% bands from past forecast errors. Dotted: forecasts made in the last backtest.",
             fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(40 * 12)
    y = 15 + 0.003 * t + 6 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.6, t.size)
    res = run(dict(values=y.tolist(), start="1985-01", horizon=36))
    for m, v in res["scores"].items():
        print(f"{m:24} rmse {v['rmse']:.3f}  skill vs repeat-last-year {v['skill']:+.0%}")
    print(res["verdict"], f"({res['seconds']}s)")
