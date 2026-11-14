"""Compare two trends. All numbers are computed here; the AI only words them.

Same units (e.g. temperature in two places): also trend-tests the difference series A - B, which is the
direct test of "is A changing faster than B?". Different units: compares direction and significance only."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import correlation
from trend_detective import analyze

KEYS = ("sen_slope_per_decade", "ols_slope_per_decade", "ols_ci95_low", "ols_ci95_high", "mk_p_value",
        "ols_p_value", "significant", "tests_agree", "direction", "n_points", "start", "end",
        "lag1_autocorr_resid")


def _brief(m):
    return {k: (round(m[k], 5) if isinstance(m[k], float) else m[k]) for k in KEYS}


def compare(sa, sb, season=None, same_units=True):
    ra, rb = analyze(sa, season), analyze(sb, season)
    ma, mb = ra["metrics"], rb["metrics"]
    rd = None
    common = sa.index.intersection(sb.index)
    identical = len(common) >= 24 and float((sa[common] - sb[common]).abs().max()) < 1e-9
    if identical:
        raise ValueError("A and B returned identical data. The two places are probably in the same NASA grid "
                         "cell (about 50 km across), so pick places further apart.")
    if same_units and len(common) >= 24:
        rd = analyze(sa[common] - sb[common], season)
    # do the wiggles around each trend move together? (trends removed first; autocorrelation-adjusted p)
    corr = correlation.correlate(ra, rb, season)
    corr_summary = {k: v for k, v in corr.items() if k != "lags"} if corr else None

    sa_sig, sb_sig = ma["significant"], mb["significant"]
    same_dir = ma["direction"] == mb["direction"]
    if sa_sig and sb_sig and not same_dir:
        relation = "opposite"
    elif sa_sig and sb_sig:
        if rd is not None and rd["metrics"]["significant"]:
            a_bigger = abs(ma["sen_slope_per_decade"]) > abs(mb["sen_slope_per_decade"])
            relation = "same_direction_a_faster" if a_bigger else "same_direction_b_faster"
        else:
            relation = "same_direction_similar" if rd is not None else "same_direction"
    elif sa_sig or sb_sig:
        relation = "only_a" if sa_sig else "only_b"
    else:
        relation = "neither"
    summary = dict(
        relation=relation, same_units=same_units,
        ci_ranges_overlap=bool(not (ma["ols_ci95_high"] < mb["ols_ci95_low"] or mb["ols_ci95_high"] < ma["ols_ci95_low"]))
        if same_units else None,
        difference=_brief(rd["metrics"]) if rd is not None else None,
        difference_significant=bool(rd["metrics"]["significant"]) if rd is not None else None,
        correlation=corr_summary,
        detrended_correlation=corr["detrended"]["r"] if corr else None,
        detrended_correlation_p=corr["detrended"]["p"] if corr else None)
    return dict(a=ra, b=rb, diff=rd, corr=corr, summary=summary, a_brief=_brief(ma), b_brief=_brief(mb))


def headline(c, label_a, label_b):
    """One plain sentence for the verdict banner: (text, kind) with kind in success|warning|info."""
    s, ma, mb = c["summary"], c["a"]["metrics"], c["b"]["metrics"]
    rel = s["relation"]
    if rel == "opposite":
        return (f"Opposite trends: {label_a} is {ma['direction']} while {label_b} is {mb['direction']}, "
                "and both are significant."), "success"
    if rel in ("same_direction_a_faster", "same_direction_b_faster"):
        fast, slow = (label_a, label_b) if rel.endswith("a_faster") else (label_b, label_a)
        return (f"Both are {ma['direction']}, and {fast} is changing significantly faster than {slow}."), "success"
    if rel == "same_direction_similar":
        return (f"Both are {ma['direction']} significantly, at rates that can't be told apart."), "info"
    if rel == "same_direction":
        return f"Both are {ma['direction']} significantly.", "info"
    if rel in ("only_a", "only_b"):
        yes, no = (label_a, label_b) if rel == "only_a" else (label_b, label_a)
        m = ma if rel == "only_a" else mb
        extra = ""
        if s["difference_significant"] is False:
            extra = " But the difference between the two is not itself significant, so don't read much into the contrast."
        return f"Only {yes} shows a significant trend ({m['direction']}); {no} does not.{extra}", "warning"
    return "Neither shows a significant trend over this period.", "warning"


def explain_template(ctx):
    a, b, s = ctx["a"], ctx["b"], ctx["comparison"]
    def one(x):
        m = x["metrics"]
        return (f"**{x['label']}**: {m['direction']} by {abs(m['sen_slope_per_decade']):.3g} {x['units']} per decade "
                f"(95% range {m['ols_ci95_low']:+.3g} to {m['ols_ci95_high']:+.3g}), "
                f"{'significant' if m['significant'] else 'not significant'}.")
    t = ctx["headline"] + "\n\n- " + one(a) + "\n- " + one(b)
    if s.get("difference"):
        d = s["difference"]
        t += (f"\n- **Difference (A − B)**: changing by {d['sen_slope_per_decade']:+.3g} {a['units']} per decade, "
              f"{'significant' if d['significant'] else 'not significant'}. This is the direct test of whether "
              "one is changing faster than the other.")
    if s.get("correlation"):
        t += "\n\n" + correlation.describe(s["correlation"], a["label"], b["label"])
    return t


def overlay_figure(c, label_a, label_b, units_a, units_b, season=None):
    """Both anomaly series and trend lines on one chart. Different units -> standard scores (z)."""
    same = c["summary"]["same_units"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    mk = "o-" if season else "-"
    for r, label, col in ((c["a"], label_a, "#33658a"), (c["b"], label_b, "#d1495b")):
        scale = 1.0 if same else (float(r["anom"].std()) or 1.0)
        ax.plot(r["anom"].index, r["anom"].values / scale, mk, lw=0.9, ms=3, color=col, alpha=0.45)
        ax.plot(r["fit"].index, r["fit"].values / scale, lw=2.6, color=col,
                ls="-" if r["metrics"]["significant"] else "--", label=label)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(f"difference from normal ({units_a})" if same else "difference from normal (standard scores)")
    ax.set_title("Thin lines: data with the seasonal cycle removed. Thick lines: trend (dashed = not significant).",
                 fontsize=9, loc="left")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def scatter_figure(c, label_a, label_b, units_a, units_b, season=None):
    """Detrended anomalies of A vs B, with the fitted line. Each dot is one month (or year)."""
    ra, rb = c["a"], c["b"]
    aa, ab = ra["anom"].align(rb["anom"], join="inner")
    x, y = (aa - ra["fit"].reindex(aa.index)).values, (ab - rb["fit"].reindex(ab.index)).values
    d = c["corr"]["detrended"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(x, y, s=10, alpha=0.5, color="#33658a")
    if np.std(x) > 0:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, b0 + b1 * xs, color="#d1495b", lw=2)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel(f"{label_a}: above/below its trend ({units_a})", fontsize=8)
    ax.set_ylabel(f"{label_b} ({units_b})", fontsize=8)
    ax.set_title(f"r = {d['r']:+.2f}  ({'month' if not season else 'year'} by {'month' if not season else 'year'})".replace(
        " (month by month)", " (each dot = one month)").replace(" (year by year)", " (each dot = one year)"), fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def lag_figure(c, label_a, label_b):
    """Correlation at each delay. Positive delay: B follows A."""
    lags = c["corr"]["lags"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ks = [d["lag"] for d in lags]
    ax.bar(ks, [d["r"] for d in lags], color=["#d1495b" if d["lag"] == 0 else "#33658a" for d in lags])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(f"months by which {label_b} follows {label_a}  (negative: {label_b} leads)", fontsize=8)
    ax.set_ylabel("correlation", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
