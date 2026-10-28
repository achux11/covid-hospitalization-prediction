# COVID-19 Hospitalization Forecast (US, Next-Day) with XGBoost

Forecasts tomorrow's number of hospitalized COVID-19 patients in the US from information available today, and checks whether a machine learning model actually beats a simple "tomorrow = today" baseline.

## Why this version exists (leakage fix)

The first version of this project reported R² = 0.975, but that number was inflated by **data leakage**:

- The 7- and 14-day rolling averages included the same day being predicted.
- Same-day ICU and ventilator counts were used as features, but they are reported at the same time as the target, so they would not be known in advance.

This version fixes both. Every feature for day *t* uses only data available by the end of day *t − 1*, and results are compared against a naive persistence baseline.

## Data

Google Health COVID-19 Open Data, `hospitalizations.csv` ([download](https://storage.googleapis.com/covid19-open-data/v3/hospitalizations.csv)). The raw file has about 1.77M rows across all locations. The model uses the **US national series** (one row per day).

## Approach

| Step | Detail |
|---|---|
| Cleaning | Parse dates, keep 2020–2025, keep `location_key == "US"`, drop missing values and zero counts, sort by date |
| Features (all from the past) | Hospitalizations 1, 7 and 14 days ago; 7- and 14-day rolling means of *previous* days; yesterday's ICU and ventilator counts; day of week; month |
| Split | Time-ordered: first 80% of days for training, last 20% for testing (no shuffling) |
| Model | XGBoost regressor (300 trees, depth 5, learning rate 0.05) |
| Baseline | Naive persistence: tomorrow's value equals today's |
| Metrics | MAE, RMSE, R², MAPE for both the model and the baseline |

## Results

**Pending re-run on the full data.** The corrected script has been tested end to end; the numbers below will be filled in from `outputs/metrics.json` after running it on the real file.

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| XGBoost (next-day) | – | – | – | – |
| Naive baseline (tomorrow = today) | – | – | – | – |

On a smooth daily series, the naive baseline is strong, so the honest question this project answers is how much (if at all) the model improves on it.

## Run

```bash
pip install -r requirements.txt
python covid_hospitalization_forecast.py --data hospitalizations.csv
```

In Google Colab, upload `hospitalizations.csv`, then run:

```python
!pip install -q -r requirements.txt
!python covid_hospitalization_forecast.py --data /content/hospitalizations.csv
```

Outputs (in `outputs/`): `metrics.json`, `feature_importance.csv`, `test_predictions.csv`, `forecast_vs_actual.png`, `feature_importance.png`.

## Tech stack

Python, pandas, NumPy, XGBoost, scikit-learn (metrics), Matplotlib, Seaborn.

## Next steps

- Forecast 7 and 14 days ahead, where a model is more likely to beat persistence
- Rolling-origin (walk-forward) backtests instead of one split
- State-level models

## Author

Achintya Vamshi Nudurupati, M.S. Business Analytics and AI, University of Texas at Dallas · [GitHub](https://github.com/achux11)
