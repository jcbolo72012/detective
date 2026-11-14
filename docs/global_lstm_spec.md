# Spec: a global pretrained LSTM forecaster (option B)

Status: proposal, not built. The per-series NumPy LSTM (`forecast_lstm.py`, option A) is the working version; this is
its possible successor.

## Idea in one paragraph

Today the app trains a small LSTM from scratch on the one series a user asks about (about 500 months). One series
is very little to learn from, so the LSTM mostly matches the straight line and helps, at best, for a few months.
Instead: train **one** model offline on thousands of series (many places × the featured variables), export it to
**ONNX**, and in the app only run it. The app needs no training (milliseconds, not seconds), and the model has
seen how anomalies behave across climates. The pitch: "a forecaster trained on 500 places, tested on places it
never saw."

## Why ONNX

`onnxruntime` is already installed in the app: it comes with `fastembed`, which the embedding search uses. So
running the model adds **no dependency**. PyTorch is needed only on the training machine.

## Data

| Item | Choice |
|---|---|
| Places | ~500 points: the 14 presets plus GeoNames cities sampled across latitude bands and continents (stratified so the tropics, deserts, high latitudes and coasts are all represented). Fixed random seed; list saved as `train/places.csv`. |
| Variables | The 8 featured NASA POWER variables (T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, GWETROOT, RH2M, WS2M, ALLSKY_SFC_SW_DWN). |
| Period | 1981 to last full year, monthly. ALLSKY from 1984. |
| Download | POWER takes up to 20 parameters per request, so one request per place: ~500 requests. Run 5 at a time (same politeness as the app), cache each place as `train/raw/<place>.parquet`, resume on restart. Roughly 10–20 minutes. |
| Split | **By place**: 80% of places train, 10% validate, 10% test, so test places are unseen. **By time** as well: the last 3 years of every series are held out of training, and the final report is on unseen places *and* unseen years. |

## Model

- **Target**: what the line misses. For each series, fit trend + 2 seasonal harmonics on the training years only,
  then standardise the residuals by that series' training standard deviation. The model predicts standardised
  residuals; the app adds the line back. Same design as option A, so results stay comparable.
- **Inputs per time step** (window of 36 months): standardised residual; month of year as sin/cos.
- **Static inputs** (repeated each step or fed to the head): variable (one-hot, 8); latitude; sin/cos longitude;
  the series' lag-1 autocorrelation of residuals (tells the model how "lingering" this place's anomalies are).
- **Network**: LSTM, 1–2 layers, 32–64 hidden units, dropout 0.1, then a linear head.
- **Direct multi-horizon output**: predict all 24 future months at once rather than feeding predictions back, which
  avoids the drift and overshoot the recursive version shows.
- **Quantiles**: output the 10th, 50th and 90th percentiles per month (pinball loss), so bands come from the model
  itself. Check coverage on the test places; fall back to backtest-error bands (option A's method) if it's off.
- **Training**: Adam, early stopping on validation places, a few minutes on a laptop CPU. Seeds fixed.

## Evaluation (the slide)

On test places and held-out years, per variable and overall:

- RMSE and skill against "repeat last year", for: line; line + per-series NumPy LSTM (option A); line + global
  LSTM (B). First 6 months and full 24 months reported separately.
- Coverage of the 80% band (10th–90th): should be near 80%.
- The headline we'd hope to show: B beats A on unseen places, especially at 1–6 months. If it doesn't, say so;
  that is still a finding.

Log every training run to MLflow (parameters, metrics, the evaluation table as an artifact) and write a short
**model card** (`models/global_lstm.md`): training places and years, variables, metrics, known limits (not a
weather forecast; no future emissions scenarios; monthly anomalies only).

## Files

| File | Purpose |
|---|---|
| `train/fetch_training_data.py` | Pick places, download POWER (cached, resumable). |
| `train/prepare.py` | Line fit on training years, residuals, windows, splits → `train/dataset.npz`. |
| `train/train_global_lstm.py` | PyTorch training, MLflow logging, ONNX export. Needs `torch` (training machine only). |
| `train/evaluate.py` | The comparison table above. |
| `models/global_lstm.onnx` | The model (expected well under 1 MB). |
| `models/global_lstm.json` | Metadata: version, window, variables, input order, normalisation, training dates, metrics. |
| `forecast_global.py` | App side: build inputs, run with `onnxruntime`, return the **same dict as `forecast_lstm.run()`** so `forecast_lstm.figure()` and the panel work unchanged. |
| `test_forecast_global.py` | Model loads; output shapes; quantiles ordered; same input → same output; unknown variable → clear error. |

## App integration

- In the forecast panel, use the global model when the variable is one of the 8 it was trained on, else option A.
  Show which one ran ("pretrained on 500 places" vs "trained just now on this series").
- No training at request time, so no session cap is needed for it; keep the cap for option A.
- Giovanni variables: not covered unless added to training. Option A handles them.

## Effort and order

1. Fetch + prepare data: ~45 min (mostly waiting on downloads).
2. Train + export + evaluate: ~60 min.
3. `forecast_global.py`, tests, panel switch: ~45 min.

Roughly 2.5 hours. Do 1–2 first and only wire it in if the evaluation shows it beats option A.

## Risks

- **POWER rate limits** during the bulk download: cache and resume, keep 5 requests at a time.
- **Leakage**: fit the line and the normalisation on training years only, or the test looks better than it is.
- **ONNX export of LSTMs**: use a recent opset (≥ 14) with dynamic batch size; check that onnxruntime output matches PyTorch to about 1e-5 on a sample.
- **Expectations**: monthly anomalies are mostly noise after a few months. Even a good global model will mainly
  help at short range; the trend tests remain the main evidence.

## Open questions

- Add Giovanni variables to training (needs a token for the bulk pull)?
- Predict several variables jointly (rain helping soil moisture)? That ties into the lead/lag correlation, but adds
  complexity; better as a second version.
