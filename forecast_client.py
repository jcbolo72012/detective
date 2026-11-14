"""Run the LSTM forecast either on the AWS Lambda (FORECAST_URL + FORECAST_TOKEN) or locally if torch is installed."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests


class Unavailable(Exception):
    pass


def backend():
    """'remote', 'local' or None."""
    if os.getenv("FORECAST_URL") and os.getenv("FORECAST_TOKEN"):
        return "remote"
    try:
        import torch  # noqa: F401
        return "local"
    except ImportError:
        return None


def _payload(series, horizon, epochs, hidden):
    s = series.dropna()
    per = s.index.to_period("M")
    if len(s) < 72:
        raise Unavailable("Need at least 6 years of monthly data for a forecast.")
    if len(per.unique()) != len(per) or (per[-1] - per[0]).n != len(per) - 1:
        raise Unavailable("The forecast needs an unbroken monthly series (this one has gaps).")
    return dict(values=[float(v) for v in s.values], start=str(per[0]), horizon=horizon, epochs=epochs, hidden=hidden)


def forecast(series, horizon=60, epochs=150, hidden=32):
    payload = _payload(series, horizon, epochs, hidden)
    mode = backend()
    if mode == "remote":
        try:
            r = requests.post(os.environ["FORECAST_URL"], json=payload, timeout=75,
                              headers={"x-forecast-token": os.environ["FORECAST_TOKEN"]})
        except requests.RequestException as e:
            raise Unavailable(f"Couldn't reach the forecast service ({type(e).__name__}).")
        if r.status_code != 200:
            try:
                msg = r.json().get("error", "")
            except ValueError:
                msg = ""
            raise Unavailable(msg or f"Forecast service returned {r.status_code}.")
        return r.json()
    if mode == "local":
        from lstm_service import forecast as fc
        return fc.run(payload)
    raise Unavailable("Forecasting isn't enabled on this server.")


def figure(series, res, title, units):
    s = series.dropna()
    per0 = pd.Period(res["start"], "M")
    test_idx = pd.period_range(per0 + (res["n_points"] - res["n_test"]), periods=res["n_test"], freq="M").to_timestamp()
    fut_idx = pd.period_range(per0 + res["n_points"], periods=res["horizon"], freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(s.index, s.values, color="k", lw=1, label="observed")
    ax.axvspan(test_idx[0], test_idx[-1], color="gold", alpha=0.2, label="held-out test years")
    for name, p in res["test_pred"].items():
        ax.plot(test_idx, p, lw=1.2, label=f"{name} (test)")
    for name, f in res["future"].items():
        ax.plot(fut_idx, f, lw=1.6, ls="--", label=f"{name} (projection)")
    ax.set_title(title)
    ax.set_ylabel(units)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig
