# PLaNETS roadmap

What we plan to build next, grouped by area. Contributions and ideas are welcome.

## Data access and accounts

- [ ] **"Log in with Earthdata" button.** Replace pasting a token with NASA's standard sign-in, so Giovanni and PO.DAAC data work in one click. Needs the app registered with NASA's Earthdata Login.
- [ ] **More NASA and partner sources.** Microsoft Planetary Computer (e.g. TerraClimate drought data) alongside NASA POWER, Giovanni and PO.DAAC.
- [ ] **Higher-resolution datasets** where they exist, such as ~1 km sea surface temperature (MUR, 2002 onward) and ~1 km North American weather (Daymet), with a clear note that a shorter record makes a weaker test.
- [ ] **Global series:** global average temperature and global mean sea level, as reference lines for any local trend.
- [ ] **Fewer, larger requests:** fetch several variables at once from NASA POWER.

## Data quality and cleaning

Every series already passes a data check before any statistics: impossible values are removed, missing months and gaps are counted, records that stop early are flagged, and a green, amber or red badge shows how complete the evidence is. Next:

- [ ] **Spike detection.** Flag months far outside that month's normal range, re-run the trend without them, and report whether the verdict changes.
- [ ] **Jump detection.** Test for sudden shifts that usually mean a change of instrument, satellite or model version rather than a change in climate.
- [ ] **Minimum-evidence rule.** Withhold the verdict when coverage or record length is too low, and say "not enough reliable data" instead.
- [ ] **Cross-checks between sources.** Where two NASA products measure the same thing, compare them and show whether they agree.
- [ ] **A wider bad-data test suite** covering each source's missing-value codes, duplicates and step changes.

## Search and question understanding

- [ ] **Fine-tuned embedding model** for matching questions to the right variable, trained on real student questions.
- [ ] **Places that aren't cities:** seas, bays, gulfs, lakes, mountain ranges and regions (today an ocean question snaps to the nearest city).
- [ ] **Pick a place on a map** or type coordinates.

## Models and forecasting

- [ ] **Global LSTM** trained across many places and variables, so a forecast for one place learns from patterns seen everywhere (design in `docs/global_lstm_spec.md`).
- [ ] **Stronger baselines** for the forecast panel to beat before it is shown as useful.

## Investigation tools

- [ ] **Opposite-trend finder:** scan many places and show where a variable is rising fastest and falling fastest.
- [ ] **Before vs after:** has the trend at one place changed between two periods?
- [ ] **Season vs season:** is summer changing faster than winter at the same place?
- [ ] **Trend leaderboard** across a set of featured places and variables.

## Reliability

- [ ] **Offline fallback** of precomputed featured cases, so demos and classrooms work when NASA's services are slow.
