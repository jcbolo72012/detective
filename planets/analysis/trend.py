"""
Earth Trend Detective - starter template
NASA Space Apps Challenge

Pipeline:
  1. Load a Giovanni "Time Series, Area-Averaged" CSV (or synthetic demo data)
  2. Resample to monthly and remove the seasonal cycle (anomalies)
  3. Trend + significance: OLS slope with 95% CI and p-value, Mann-Kendall + Sen's slope
  4. Chronological holdout: linear model (trend + seasonality) vs LSTM
  5. Refit on all data and project forward
  6. Log params, metrics and figures to MLflow (if installed)

Usage:
  python trend_detective.py --demo
  python trend_detective.py --csv giovanni_export.csv --name "Precip, Sahel" --units "mm/day"
  python trend_detective.py --csv omi_ozone_antarctica.csv --season SON --name "Antarctic ozone" --units DU
    (--season: one value per year from those months; also DJF/MAM/JJA, "9,10" or "9-11")
  mlflow ui      # then open http://127.0.0.1:5000

Install:  pip install pandas numpy scipy scikit-learn matplotlib mlflow torch
torch and mlflow are optional: the script skips the LSTM / logging if they are missing.
"""
import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 1. data
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
def linear_features(idx, t0, seasonal=True):
    k = ((idx.year - t0.year) * 12 + (idx.month - t0.month)).values.astype(float)
    if not seasonal:  # annual series: just intercept + trend
        return np.column_stack([np.ones_like(k), k])
    m = idx.month.values
    return np.column_stack([np.ones_like(k), k,
                            np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12),
                            np.sin(4 * np.pi * m / 12), np.cos(4 * np.pi * m / 12)])


def fit_linear(train, seasonal=True):
    t0 = train.index[0]
    X = linear_features(train.index, t0, seasonal)
    beta, *_ = np.linalg.lstsq(X, train.values, rcond=None)
    return lambda idx: linear_features(idx, t0, seasonal) @ beta


if HAS_TORCH:
    class LSTMNet(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])


def fit_lstm(train, window=24, epochs=300, hidden=32, lr=1e-2, seed=0):
    """Returns predict(n_steps) that rolls forward recursively from the end of train."""
    torch.manual_seed(seed)
    mu, sd = train.mean(), train.std()
    z = ((train.values - mu) / sd).astype(np.float32)
    X = np.stack([z[i:i + window] for i in range(len(z) - window)])[..., None]
    Y = z[window:][:, None]
    X, Y = torch.tensor(X), torch.tensor(Y)
    model = LSTMNet(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), Y)
        loss.backward()
        opt.step()

    def predict(n_steps):
        hist = list(z[-window:])
        preds = []
        model.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                x = torch.tensor(np.array(hist[-window:], dtype=np.float32)[None, :, None])
                p = model(x).item()
                preds.append(p)
                hist.append(p)
        return np.array(preds) * sd + mu

    return predict, float(loss.item())


def metrics(y, yhat, prefix):
    err = np.asarray(y) - np.asarray(yhat)
    return {f"{prefix}_rmse": float(np.sqrt(np.mean(err ** 2))),
            f"{prefix}_mae": float(np.mean(np.abs(err)))}


# ---------------------------------------------------------------- 5. plots
def plot_trend(s, anom, r, title, units, path, season=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    mk = "o-" if season else "-"
    ax[0].plot(s.index, s.values, mk, lw=1, ms=4)
    ax[0].set_title(f"{title}: " + (f"{season} mean, one value per year" if season else "raw monthly series"))
    ax[0].set_ylabel(units)
    x = (anom.index - anom.index[0]).days.values / 365.25
    lab = "anomaly vs. record mean" if season else "anomaly (seasonal cycle removed)"
    ax[1].plot(anom.index, anom.values, mk, lw=1, ms=4, color="gray", label=lab)
    ax[1].plot(anom.index, r.intercept + r.slope * x, color="crimson", lw=2,
               label=f"trend {r.slope*10:+.3g} {units}/decade (p={r.pvalue:.3g})")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel(f"anomaly ({units})")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forecast(s, test_idx, preds, future_idx, future, title, units, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(s.index, s.values, color="k", lw=1, label="observed")
    if test_idx is not None:
        ax.axvspan(test_idx[0], test_idx[-1], color="gold", alpha=0.2, label="held-out test")
    for name, p in preds.items():
        ax.plot(test_idx, p, lw=1.5, label=f"{name} (test)")
    for name, f in future.items():
        ax.plot(future_idx, f, lw=1.5, ls="--", label=f"{name} (projection)")
    ax.set_title(title)
    ax.set_ylabel(units)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--name", default="Demo variable")
    ap.add_argument("--units", default="units")
    ap.add_argument("--test_months", type=int, default=36)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--experiment", default="earth-trend-detective")
    ap.add_argument("--season", help="Analyse one value per year: SON, DJF, MAM, JJA, '9,10' or '9-11'. "
                                     "E.g. --season SON for the Antarctic ozone hole.")
    ap.add_argument("--test_years", type=int, default=5, help="holdout length when --season is used")
    a = ap.parse_args()

    s = synthetic_series() if a.demo or not a.csv else load_giovanni_csv(a.csv)
    print(f"{len(s)} monthly points, {s.index[0]:%Y-%m} to {s.index[-1]:%Y-%m}")

    annual = bool(a.season)
    if annual:
        months = parse_season(a.season)
        s = seasonal_annual(s, months)
        print(f"Season {a.season} (months {months}): {len(s)} years, {s.index[0].year} to {s.index[-1].year}")
        if len(s) < 10:
            print("Warning: fewer than 10 years, so significance tests will be weak.")
        steps_per_year, n_test, horizon = 1, a.test_years, max(1, a.horizon // 12)
        window = min(a.window, max(2, (len(s) - n_test) // 3))
        freq = "YS"
    else:
        steps_per_year, n_test, horizon, window, freq = 12, a.test_months, a.horizon, a.window, "MS"

    # trend + significance on anomalies
    anom = s - s.mean() if annual else deseasonalize(s)[0]
    ols, r = ols_trend(anom)
    mk = mann_kendall(anom.values)
    mk["sen_slope_per_decade"] = mk.pop("sen_slope_per_step") * 10 * steps_per_year
    rho = lag1_autocorr(anom.values - (r.intercept + r.slope * (anom.index - anom.index[0]).days.values / 365.25))
    trend = {**ols, **mk, "lag1_autocorr_resid": rho}
    sig = trend["mk_p_value"] < 0.05 and trend["ols_p_value"] < 0.05
    print(json.dumps(trend, indent=2))
    print("Verdict:", "SIGNIFICANT trend (p<0.05)" if sig else "no significant trend at p<0.05")
    if rho > 0.3:
        print(f"Note: residual lag-1 autocorrelation {rho:.2f}, so p-values are optimistic.")

    # holdout comparison
    train, test = s.iloc[:-n_test], s.iloc[-n_test:]
    lname = "linear trend" if annual else "linear+seasonal"
    preds = {lname: fit_linear(train, not annual)(test.index)}
    m = metrics(test, preds[lname], "test_linear")
    if HAS_TORCH:
        lstm_pred, train_loss = fit_lstm(train, window, a.epochs)
        preds["LSTM"] = lstm_pred(len(test))
        m.update(metrics(test, preds["LSTM"], "test_lstm"))
        m["lstm_train_loss"] = train_loss
    if annual:  # baseline: persistence (last training year)
        m.update(metrics(test, np.repeat(train.iloc[-1], len(test)), "test_persistence"))
    else:       # baseline: repeat last year
        m.update(metrics(test, np.tile(train.iloc[-12:].values, 1 + len(test) // 12)[:len(test)],
                         "test_seasonal_naive"))
    print(json.dumps(m, indent=2))

    # future projection: refit on everything
    step = pd.offsets.YearBegin() if annual else pd.offsets.MonthBegin()
    future_idx = pd.date_range(s.index[-1] + step, periods=horizon, freq=freq)
    future = {lname: fit_linear(s, not annual)(future_idx)}
    if HAS_TORCH:
        future["LSTM"] = fit_lstm(s, window, a.epochs)[0](horizon)

    f1 = os.path.join(OUT, "trend.png")
    f2 = os.path.join(OUT, "holdout_and_projection.png")
    plot_trend(s, anom, r, a.name, a.units, f1, a.season)
    plot_forecast(s, test.index, preds, future_idx, future, a.name, a.units, f2)
    pd.DataFrame(future, index=future_idx).to_csv(os.path.join(OUT, "projection.csv"))
    print("Saved figures to", OUT)

    if HAS_MLFLOW:
        mlflow.set_experiment(a.experiment)
        with mlflow.start_run(run_name=a.name):
            mlflow.log_params({"variable": a.name, "units": a.units, "source": a.csv or "synthetic",
                               "season": a.season or "all-months", "n_points": len(s),
                               "n_test": n_test, "window": window, "epochs": a.epochs,
                               "horizon_steps": horizon})
            mlflow.log_metrics({**trend, **m, "significant": int(sig)})
            for f in (f1, f2, os.path.join(OUT, "projection.csv")):
                mlflow.log_artifact(f)
        print("Logged to MLflow. Run `mlflow ui` to view.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- reusable entry point (used by app.py)
def analyze(s, season=None):
    """Monthly pd.Series -> dict with the analysed series, anomalies, fitted line and metrics.
    All numbers the app (and the AI explanation) shows come from here."""
    if season:
        s = seasonal_annual(s, parse_season(season))
        anom, steps_per_year = s - s.mean(), 1
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
         "mean": float(s.mean())}
    m["significant"] = bool(m["ols_p_value"] < 0.05 and m["mk_p_value"] < 0.05)
    m["tests_agree"] = bool((m["ols_p_value"] < 0.05) == (m["mk_p_value"] < 0.05))
    m["direction"] = "increasing" if m["sen_slope_per_decade"] > 0 else "decreasing"
    m = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in m.items()}
    return {"series": s, "anom": anom, "fit": pd.Series(fit, index=anom.index), "metrics": m}
