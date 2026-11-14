# Wiring the NumPy LSTM (option A) into the app

`lstm_numpy.py` and `forecast_lstm.py` are standalone: nothing imports them yet. To switch the forecast panel over
(about 20 minutes, in `app.py` → `render_forecast`):

1. **Backend check.** Replace `forecast_client.backend()` with: always available (NumPy only). Keep the Lambda
   path only if you still want it as an option.
2. **Run it.** `res = forecast_lstm.run_series(series, horizon=horizon, epochs=epochs)`; catch `ValueError` and show
   it with `st.warning` (gaps, too short, too long).
3. **Table.** One row per model from `res["scores"]`: RMSE (2 years), RMSE (first `res["short_months"]` months),
   skill vs repeat-last-year (`skill`, as a %). Add `glossary` tooltips; new terms: *skill*, *backtest*, *band*.
4. **Verdict.** `st.info(res["verdict"])`, plus `f"Beat the line in {res['wins']} of {res['n_folds']} tests"`.
5. **Chart.** `fig = forecast_lstm.figure(series, res, title, units)`; `st.pyplot(fig)`; `plt.close(fig)`.
6. **Keep** the 5-per-session cap and the "not in season mode" rule. Typical run: 2–5 seconds.
7. **Cache** by series: wrap in `st.cache_data` keyed on (variable, place, years, horizon, epochs), passing the
   series values, so reruns and Back are instant.

Result dict (JSON-safe): `start, n_points, horizon, settings, models, folds, wins, wins_short, n_folds,
short_months, scores{model: rmse, mae, skill, rmse_short}, best, verdict, future{model: [...]},
bands{model: {lo, hi}}, last_test{origin, n, pred{model: [...]}}, seconds`.

Tests: `python test_forecast_lstm.py` (about 40 s; includes an exact gradient check and a band-coverage check).
