"""Plain-language definitions for every statistic the app shows. Used for hover/tap tooltips (ⓘ) and the sidebar glossary."""

TERMS = {
    "change_per_decade": ("Change per decade",
        "How much the value changes every 10 years, using Sen's slope: the median of all pairwise slopes, so a few "
        "extreme months can't drag it around. Positive means rising, negative means falling."),
    "ci95": ("95% range",
        "The range the true trend most likely falls in (a 95% confidence interval from linear regression, per decade). "
        "If the range includes zero, the data can't rule out 'no trend'."),
    "mk_p": ("Mann-Kendall p-value",
        "Result of the Mann-Kendall trend test, which only asks whether values tend to go up or down over time, "
        "ignoring size and without assuming a bell curve. The p-value is the chance of seeing a pattern this strong "
        "if there were really no trend. Below 0.05 is the usual cutoff for 'significant'."),
    "ols_p": ("t-test p-value",
        "Result of the regression t-test: the chance of a slope this steep if the true slope were zero. Below 0.05 "
        "is the usual cutoff. This app calls a trend significant only when this and Mann-Kendall both agree."),
    "significant": ("Significant?",
        "Yes means both statistical tests (regression t-test and Mann-Kendall) found a trend with p < 0.05. "
        "It means 'unlikely to be chance', not 'large' or 'important'."),
    "anomaly": ("Anomaly",
        "The value minus what is normal for that time of year. Subtracting the seasonal cycle lets a real long-term "
        "change show up instead of being hidden by summer-winter swings."),
    "p_value": ("p-value",
        "The probability of seeing a result at least this strong if nothing real were going on. Small (under 0.05) "
        "means 'probably not a fluke'. It is not the probability that the finding is true."),
    "difference": ("A − B (difference)",
        "The gap between the two series, tested for a trend. If that gap is significantly growing or shrinking, one "
        "really is changing faster than the other. Only available when both have the same units."),
    "correlation": ("Correlation (trends removed)",
        "Pearson's r, from −1 to +1: how closely the two series' ups and downs match once each one's own trend is "
        "subtracted. +1 move together perfectly, 0 unrelated, −1 move opposite. It does not show that one causes the other."),
    "corr_range": ("95% range (correlation)",
        "The range the true correlation most likely falls in. A wide range means the data are too short to pin it down."),
    "corr_p": ("p-value (correlation)",
        "Chance of a correlation this strong between unrelated series. Adjusted for neighbouring months being alike "
        "(autocorrelation), which makes it stricter than a standard test."),
    "n_eff": ("Effective sample size",
        "Because this month resembles last month, 240 monthly values carry less independent information than 240 "
        "separate measurements. This is the 'equivalent number of independent points' used for significance."),
    "autocorr": ("Autocorrelation",
        "How much one month resembles the next after the trend is removed. High values mean slow-changing noise, "
        "which makes trends look more significant than they are, so tests are interpreted with extra caution."),
    "lag": ("Delay (lag)",
        "Whether one series tends to follow the other by some months. Positive: B follows A. Many delays are tried, "
        "so a standout is a hint to investigate, not a finding."),
    "spearman": ("Spearman check",
        "A rank-based version of correlation that isn't thrown off by a few extreme values. If it agrees with the "
        "main number, the relationship is not driven by outliers."),
    "rmse": ("Test RMSE",
        "Root mean squared error on the held-out years the model never saw: the typical size of its miss, in the "
        "same units as the data, with big misses weighted more. Lower is better."),
    "mae": ("Test MAE",
        "Mean absolute error on the held-out years: the average size of the model's miss, in the data's units. "
        "Lower is better."),
    "skill": ("Skill",
        "How much better a forecast is than simply repeating last year's values: 0% means no better, 100% means "
        "perfect, negative means worse. Measured on backtests, never on data the model trained on."),
    "backtest": ("Backtest",
        "Pretend it's an earlier year: hide everything after that date, forecast the next two years, then compare "
        "with what really happened. Doing it from several dates shows whether a model's success was luck."),
    "band": ("Forecast band",
        "The shaded range around a projection, built from how wrong the model actually was in its backtests. "
        "It widens further out because forecasts get less certain the further ahead they reach."),
    "years_until": ("Years until significant",
        "How long a record a trend this size needs before it stands out from the natural ups and downs, if it "
        "keeps going at the rate measured so far. 'Even odds' is the length at which a record would test "
        "significant about half the time; 4-in-5 needs longer. Already-significant trends show the year they "
        "first became detectable."),
    "holdout": ("Held-out test",
        "The last part of the record is hidden while the model trains, then used to check its predictions. A model "
        "that only looks good on data it has already seen proves nothing."),
}


def tip(key):
    """Tooltip text for st.metric(help=...)."""
    return TERMS[key][1]


def column_config(**cols):
    """st.dataframe column_config with a tooltip on each header. Usage: column_config(**{"95% range": "ci95"})."""
    import streamlit as st
    return {name: st.column_config.TextColumn(name, help=tip(key)) for name, key in cols.items()}
