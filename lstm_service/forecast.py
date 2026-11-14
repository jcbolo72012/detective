"""Holdout comparison + projection for one monthly series. numpy-only except the LSTM, which needs torch.

Used two ways: imported by forecast_client.py (local fallback) and by handler.py (AWS Lambda).
Everything the caller controls is clamped by validate() because the Lambda URL is public.

Self-test:  python forecast.py --selftest      (needs numpy and torch)
"""
import json
import sys

import numpy as np

LIMITS = dict(min_points=72, max_points=600, max_horizon=120, max_epochs=300, max_hidden=64)
WINDOW = 24


def validate(payload):
    """Return a clean payload or raise ValueError with a user-readable message."""
    try:
        vals = [float(v) for v in payload["values"]]
        start = str(payload["start"])
        y, m = int(start[:4]), int(start[5:7])
        horizon = int(payload.get("horizon", 60))
        epochs = int(payload.get("epochs", 150))
        hidden = int(payload.get("hidden", 32))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Expected {values: [...], start: 'YYYY-MM', horizon, epochs, hidden}.")
    if not (1 <= m <= 12 and 1800 <= y <= 2200):
        raise ValueError("start must look like 2001-01.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("values must all be finite numbers (fill or drop gaps first).")
    L = LIMITS
    if not (L["min_points"] <= len(vals) <= L["max_points"]):
        raise ValueError(f"Need between {L['min_points']} and {L['max_points']} monthly values.")
    return dict(values=vals, start=(y, m),
                horizon=int(np.clip(horizon, 1, L["max_horizon"])),
                epochs=int(np.clip(epochs, 10, L["max_epochs"])),
                hidden=int(np.clip(hidden, 4, L["max_hidden"])))


def _month_index(start, n):
    y, m = start
    k = y * 12 + (m - 1) + np.arange(n)
    return k  # absolute month number


def _features(k, k0):
    t = (k - k0) / 120.0  # decades
    ph = 2 * np.pi * (k % 12) / 12
    return np.column_stack([np.ones_like(t), t, np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)])


def fit_linear(k_train, y_train):
    k0 = k_train[0]
    beta, *_ = np.linalg.lstsq(_features(k_train, k0), y_train, rcond=None)
    return lambda k: _features(k, k0) @ beta


def fit_naive_seasonal(y_train):
    last = np.asarray(y_train[-12:])
    return lambda n: np.resize(last, n)  # repeats last year's values


def fit_lstm(y_train, epochs, hidden, window=WINDOW, lr=1e-2, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    mu, sd = float(np.mean(y_train)), float(np.std(y_train)) or 1.0
    z = ((np.asarray(y_train) - mu) / sd).astype(np.float32)
    X = np.stack([z[i:i + window] for i in range(len(z) - window)])[..., None]
    Y = z[window:][:, None]
    X, Y = torch.tensor(X), torch.tensor(Y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(X), Y)
        loss.backward()
        opt.step()

    def predict(n):
        hist = list(z[-window:])
        out = []
        net.eval()
        with torch.no_grad():
            for _ in range(n):
                p = net(torch.tensor(np.array(hist[-window:], dtype=np.float32)[None, :, None])).item()
                out.append(p)
                hist.append(p)
        return np.array(out) * sd + mu

    return predict


def _err(y, yhat):
    e = np.asarray(y) - np.asarray(yhat)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))))


def run(payload, lstm_fn=fit_lstm):
    c = validate(payload)
    y = np.array(c["values"])
    n = len(y)
    k = _month_index(c["start"], n)
    n_test = int(np.clip(round(0.2 * n), 24, n // 3))
    ktr, kte, ytr, yte = k[:-n_test], k[-n_test:], y[:-n_test], y[-n_test:]

    lin = fit_linear(ktr, ytr)
    pred = {"Linear trend + seasons": lin(kte),
            "Repeat last year (naive)": fit_naive_seasonal(ytr)(n_test),
            "LSTM": lstm_fn(ytr, c["epochs"], c["hidden"])(n_test)}
    scores = {name: _err(yte, p) for name, p in pred.items()}

    # refit on everything and project forward
    kf = k[-1] + 1 + np.arange(c["horizon"])
    future = {"Linear trend + seasons": fit_linear(k, y)(kf),
              "LSTM": lstm_fn(y, c["epochs"], c["hidden"])(c["horizon"])}

    best = min(scores, key=lambda s: scores[s]["rmse"])
    if best == "LSTM":
        verdict = ("On the held-out years the LSTM had the lowest error of the three models, "
                   "so its projection is at least competitive. Still treat long projections cautiously.")
    else:
        verdict = (f"On the held-out years the LSTM (RMSE {scores['LSTM']['rmse']:.3g}) did not beat "
                   f"{best} (RMSE {scores[best]['rmse']:.3g}). Treat its projection with caution; the trend "
                   f"significance tests are the stronger evidence here.")
    r = lambda a: [round(float(v), 4) for v in a]
    return dict(n_points=n, n_test=n_test, start=f"{c['start'][0]:04d}-{c['start'][1]:02d}",
                horizon=c["horizon"], epochs=c["epochs"], hidden=c["hidden"],
                test_pred={m: r(v) for m, v in pred.items()}, scores=scores,
                future={m: r(v) for m, v in future.items()}, best=best, verdict=verdict)


def _selftest():
    rng = np.random.default_rng(0)
    t = np.arange(240)
    y = 10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, t.size)
    out = run(dict(values=y.tolist(), start="2005-01", horizon=36, epochs=60))
    print(json.dumps({k: out[k] for k in ("n_points", "n_test", "scores", "best", "verdict")}, indent=2))
    assert len(out["future"]["LSTM"]) == 36 and len(out["test_pred"]["LSTM"]) == out["n_test"]
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
