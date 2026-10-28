"""
COVID-19 hospitalization forecast (US, next-day) with XGBoost.

Leakage-free version. Every feature for day t uses only information
available at the end of day t-1:
  - lags of the target (1, 7, 14 days)
  - rolling means of the target computed on PAST days only (shifted by 1)
  - ICU and ventilator counts from the PREVIOUS day
  - calendar features (day of week, month)

The model is compared against a naive persistence baseline
("tomorrow = today"), which is hard to beat on a smooth series.

Data: Google Health COVID-19 Open Data, hospitalizations.csv
https://storage.googleapis.com/covid19-open-data/v3/hospitalizations.csv

Run (Colab or locally):
    pip install -r requirements.txt
    python covid_hospitalization_forecast.py --data hospitalizations.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "current_hospitalized_patients"
ICU = "current_intensive_care_patients"
VENT = "current_ventilator_patients"


def load_us(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Raw rows: {len(df):,}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"].dt.year >= 2020) & (df["date"].dt.year <= 2025)]

    us = df[df["location_key"] == "US"].copy()
    us = us.dropna(subset=[TARGET, ICU, VENT])
    us = us[us[TARGET] > 0]  # filter BEFORE building features so lags stay aligned
    us = us.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    print(f"US national rows after cleaning: {len(us):,}")
    return us


def build_features(us: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    y = us[TARGET]
    past = y.shift(1)  # yesterday's value: the latest thing we know

    us["lag_1"] = past
    us["lag_7"] = y.shift(7)
    us["lag_14"] = y.shift(14)
    us["rolling_7"] = past.rolling(7).mean()   # past 7 days, excluding today
    us["rolling_14"] = past.rolling(14).mean()
    us["icu_lag_1"] = us[ICU].shift(1)         # yesterday's ICU load
    us["vent_lag_1"] = us[VENT].shift(1)       # yesterday's ventilator load
    us["day_of_week"] = us["date"].dt.dayofweek
    us["month"] = us["date"].dt.month

    features = ["lag_1", "lag_7", "lag_14", "rolling_7", "rolling_14",
                "icu_lag_1", "vent_lag_1", "day_of_week", "month"]
    us = us.dropna(subset=features).reset_index(drop=True)
    return us, features


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def main(data_path: str, out_dir: str = "outputs") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    us, features = build_features(load_us(data_path))

    # Time-ordered split: train on the past, test on the future
    split = int(len(us) * 0.8)
    train, test = us.iloc[:split], us.iloc[split:]
    print(f"Train: {len(train)} days ({train.date.min().date()} to {train.date.max().date()})")
    print(f"Test:  {len(test)} days ({test.date.min().date()} to {test.date.max().date()})")

    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(train[features], train[TARGET])
    pred = model.predict(test[features])

    results = {
        "test_days": len(test),
        "xgboost": metrics(test[TARGET].values, pred),
        "naive_baseline_tomorrow_equals_today": metrics(test[TARGET].values, test["lag_1"].values),
    }
    results["xgboost_beats_baseline_on_MAE"] = (
        results["xgboost"]["MAE"] < results["naive_baseline_tomorrow_equals_today"]["MAE"]
    )
    print(json.dumps(results, indent=2))
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    importance = (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
                  .sort_values("importance", ascending=False))
    importance.to_csv(out / "feature_importance.csv", index=False)
    print(importance.to_string(index=False))

    comparison = pd.DataFrame({
        "date": test["date"].values,
        "actual": test[TARGET].values,
        "xgboost": pred,
        "naive_baseline": test["lag_1"].values,
    })
    comparison.to_csv(out / "test_predictions.csv", index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(comparison.date, comparison.actual, label="Actual", linewidth=2.5)
    plt.plot(comparison.date, comparison.xgboost, label="XGBoost (next-day)", linewidth=2)
    plt.plot(comparison.date, comparison.naive_baseline, label="Naive baseline", linewidth=1.5, linestyle="--")
    plt.title("US COVID-19 Hospitalizations: Next-Day Forecast on Held-Out Period")
    plt.xlabel("Date"); plt.ylabel("Hospitalized patients")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "forecast_vs_actual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance, x="importance", y="feature", color="#4C72B0")
    plt.title("XGBoost Feature Importance (leakage-free features)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved outputs to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hospitalizations.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    main(args.data, args.out)
