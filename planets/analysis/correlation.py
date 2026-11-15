"""How closely do two series move together? All numbers computed here; the AI only words them.

Method (monthly or one-value-per-year anomalies, seasonal cycle already removed):
  * detrended correlation: remove each series' own trend first, so two things that merely both drift
    don't look linked. This is the headline number.
  * trend-included correlation: same on anomalies WITH trends, shown next to it. A big gap means a
    shared trend, not shared year-to-year behaviour.
  * significance accounts for autocorrelation: neighbouring months aren't independent, so the effective
    sample size is shrunk (Bretherton et al. 1999): n_eff = n (1 - r1a r1b) / (1 + r1a r1b).
    Plain Pearson p-values overstate significance on climate series.
  * lead/lag: correlation of A(t) with B(t+k), k = -12..+12 months, exploratory (25 looks).
"""
import numpy as np
from scipy import stats

LAGS = range(-12, 13)


def _lag1(x):
    x = np.asarray(x, float) - np.mean(x)
    d = float((x * x).sum())
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else 0.0


def _n_eff(x, y):
    n = len(x)
    phi = _lag1(x) * _lag1(y)
    ne = n * (1 - phi) / (1 + phi) if phi > 0 else n  # only ever shrink
    return float(max(5.0, min(float(n), ne)))


def pearson_adj(x, y):
    """Pearson r with autocorrelation-adjusted p and 95% CI. Returns dict or None if too short/constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 8 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r = float(np.clip(np.corrcoef(x, y)[0, 1], -0.999999, 0.999999))
    ne = _n_eff(x, y)
    t = r * np.sqrt((ne - 2) / (1 - r * r))
    p = float(2 * stats.t.sf(abs(t), ne - 2))
    z, se = np.arctanh(r), 1 / np.sqrt(ne - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return dict(r=r, p=p, ci_low=float(lo), ci_high=float(hi), n=int(n), n_eff=round(ne, 1))


def strength(r):
    a = abs(r)
    return "very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else \
        "strong" if a < 0.8 else "very strong"


def correlate(ra, rb, season=None):
    """ra, rb: analyze() results for A and B (same season setting). Returns a JSON-friendly dict."""
    aa, ab = ra["anom"].align(rb["anom"], join="inner")
    if len(aa) < 8:
        return None
    da = aa - ra["fit"].reindex(aa.index)
    db = ab - rb["fit"].reindex(ab.index)
    det = pearson_adj(da.values, db.values)
    raw = pearson_adj(aa.values, ab.values)
    if det is None or raw is None:
        return None
    rho, _ = stats.spearmanr(da.values, db.values)
    out = dict(detrended=det, with_trends=raw, spearman_detrended=round(float(rho), 3),
               strength=strength(det["r"]), significant=bool(det["p"] < 0.05),
               direction="together" if det["r"] > 0 else "opposite",
               shared_trend=bool(abs(raw["r"]) - abs(det["r"]) > 0.25 and raw["p"] < 0.05),
               lags=None, best_lag=None)
    if not season and len(da) >= 48:
        lags = []
        for k in LAGS:
            # A(t) vs B(t+k): positive k means B follows A
            bb = db.shift(-k)
            ok = bb.notna()
            if ok.sum() < 36:
                continue
            res = pearson_adj(da[ok].values, bb[ok].values)
            if res:
                lags.append(dict(lag=k, r=round(res["r"], 3), p=res["p"]))
        if lags:
            out["lags"] = lags
            best = max(lags, key=lambda d: abs(d["r"]))
            r0 = next((d["r"] for d in lags if d["lag"] == 0), 0.0)
            # only claim a lead/lag if it is clearly better than simultaneous AND still significant
            out["best_lag"] = dict(lag=best["lag"], r=best["r"], p=best["p"], r_at_zero=r0,
                                   clear=bool(best["lag"] != 0 and abs(best["r"]) - abs(r0) > 0.1 and best["p"] < 0.01))
    # rounded copy of the two blocks for display
    for k in ("detrended", "with_trends"):
        out[k] = {kk: (round(v, 4) if isinstance(v, float) and kk != "p" else v) for kk, v in out[k].items()}
    return out


def describe(c, label_a, label_b):
    """Plain-language paragraph built from the numbers (the no-AI explanation, and the check on any AI text)."""
    if not c:
        return ""
    d, w = c["detrended"], c["with_trends"]
    sig = "statistically significant" if c["significant"] else "not statistically significant"
    t = (f"With each trend removed, {label_a} and {label_b} show a {c['strength']} relationship "
         f"(r = {d['r']:+.2f}, 95% range {d['ci_low']:+.2f} to {d['ci_high']:+.2f}, {sig}"
         f"{', p = %.2g' % d['p'] if c['significant'] else ''}). ")
    if c["significant"]:
        t += ("When one is above its trend, the other tends to be too. " if d["r"] > 0
              else "When one is above its trend, the other tends to be below its. ")
    if c["shared_trend"]:
        t += (f"Counting the trends, the link looks stronger (r = {w['r']:+.2f}), but that part comes from both "
              "drifting over time, which is not evidence the two are connected. ")
    bl = c.get("best_lag")
    if bl and bl["clear"]:
        who, whom = (label_a, label_b) if bl["lag"] > 0 else (label_b, label_a)
        t += (f"Exploratory: the link is strongest when {whom} follows {who} by {abs(bl['lag'])} month"
              f"{'s' if abs(bl['lag']) != 1 else ''} (r = {bl['r']:+.2f}, versus {bl['r_at_zero']:+.2f} with no delay). ")
    t += ("Correlation shows that they move together, not that one causes the other: a third factor "
          "(like a weather pattern) can drive both.")
    return t
