
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from Part_A.features import get_data, get_inference_features
from Part_B_Demand.predict_demand import predict_demand

# Access the official project evaluation helpers
HARNESS_PATH = (
    Path(__file__).resolve().parents[2]
    / "comp9417_project_kit"
    / "harness"
)
sys.path.insert(0, str(HARNESS_PATH))

from dayahead import make_inputs, targets, baseline, mae, mape

print("Evaluation imports successful!")

# Load the electricity dataset
df = get_data()
ridge_path = Path(__file__).resolve().parent / "artifacts" / "ridge_model.joblib"
ridge_model = joblib.load(ridge_path)

# Choose the first evaluation day
target_day = "2015-01-01"

# Prepare historical data and next-day weather
history, weather_next = make_inputs(df, target_day)

print("Target day:", target_day)
print("History rows:", len(history))
print("Next-day weather rows:", len(weather_next))

# Generate features for the next day
X_next = get_inference_features(history, weather_next)

# Predict NSW electricity demand
rf_pred = predict_demand(X_next)

print("Total predictions:", len(rf_pred))
print("First 5 predictions:", rf_pred[:5])

# Get actual NSW electricity demand
actual = targets(df, target_day)["nsw_demand_mw"]

# Evaluate Random Forest predictions
rf_mae = mae(actual, rf_pred)
rf_mape = mape(actual, rf_pred)

print("\n--- Random Forest Evaluation ---")
print(f"MAE: {rf_mae:.2f} MW")
print(f"MAPE: {rf_mape:.2f}%")

# Yesterday baseline
yesterday_pred = baseline(
    history, weather_next, lag_days=1
)["nsw_demand_mw"]

yesterday_mae = mae(actual, yesterday_pred)
yesterday_mape = mape(actual, yesterday_pred)

print("\n--- Yesterday Baseline Evaluation ---")
print(f"MAE: {yesterday_mae:.2f} MW")
print(f"MAPE: {yesterday_mape:.2f}%")

# Last week baseline
lastweek_pred = baseline(
    history, weather_next, lag_days=7
)["nsw_demand_mw"]

lastweek_mae = mae(actual, lastweek_pred)
lastweek_mape = mape(actual, lastweek_pred)

print("\n--- Last Week Baseline Evaluation ---")
print(f"MAE: {lastweek_mae:.2f} MW")
print(f"MAPE: {lastweek_mape:.2f}%")

# Create a list of evaluation dates
evaluation_days = pd.date_range(
    start="2015-01-01",
    end="2026-10-31",
    freq="D"
)

print("\n--- Evaluation Dates ---")

daily_results = []
for day in evaluation_days:
    day_str = day.strftime("%Y-%m-%d")

    # Prepare inputs for this day
    history, weather_next = make_inputs(df, day_str)

    # Skip days with incomplete data
    if len(weather_next) != 48:
        print(f"Skipping {day_str}: expected 48 intervals, got {len(weather_next)}")
        continue

    X_next = get_inference_features(history, weather_next)

    # Check whether prediction features contain invalid values
    if not np.isfinite(X_next.to_numpy(dtype=float)).all():
        print(f"Skipping {day_str}: invalid prediction features")
        continue

    # Predict 48 half-hour intervals
    predictions = predict_demand(X_next)

    # Show progress every 30 days
    if len(daily_results) % 30 == 0:
        print(f"Processing: {day_str}")

    # Get actual NSW demand
    actual = targets(df, day_str)["nsw_demand_mw"]  
    
    # Skip days with incomplete actual demand
    if len(actual) != 48 or actual.isna().any():
        print(f"Skipping {day_str}: incomplete actual demand")
        continue  

    # Last Week Baseline prediction
    lastweek_pred = baseline(
        history, weather_next, lag_days=7
    )["nsw_demand_mw"]

    lastweek_mae = mae(actual, lastweek_pred)
    lastweek_mape = mape(actual, lastweek_pred)

    # Calculate prediction errors
    daily_mae = mae(actual, predictions)
    daily_mape = mape(actual, predictions)

    # Yesterday baseline for the same day
    yesterday_pred = baseline(
        history, weather_next, lag_days=1
    )["nsw_demand_mw"]

    # Check baseline prediction lengths
    if len(yesterday_pred) != 48 or len(lastweek_pred) != 48:
        print(f"Skipping {day_str}: incomplete baseline predictions")
        continue

    yesterday_daily_mae = mae(actual, yesterday_pred)
    yesterday_daily_mape = mape(actual, yesterday_pred)

    daily_results.append({
        "date": day_str,
        "MAE": daily_mae,
        "MAPE": daily_mape,
        "Yesterday_MAE": yesterday_daily_mae,
        "Yesterday_MAPE": yesterday_daily_mape,
        "LastWeek_MAE": lastweek_mae,
        "LastWeek_MAPE": lastweek_mape
    })

    # print(f"  MAE: {daily_mae:.2f} MW")
    # print(f"  MAPE: {daily_mape:.2f}%")

# Convert daily evaluation results into a DataFrame
results_df = pd.DataFrame(daily_results)

# Evaluation coverage
total_days = len(evaluation_days)
evaluated_days = len(results_df)
skipped_days = total_days - evaluated_days

print("\n--- Evaluation Coverage ---")
print(f"Total days: {total_days}")
print(f"Evaluated days: {evaluated_days}")
print(f"Skipped days: {skipped_days}")
print(f"Coverage: {evaluated_days / total_days * 100:.2f}%")

# Calculate annual evaluation summary
results_df["year"] = pd.to_datetime(results_df["date"]).dt.year

annual_summary = results_df.groupby("year")[
    ["MAE", "MAPE", "Yesterday_MAE", "Yesterday_MAPE",
     "LastWeek_MAE", "LastWeek_MAPE"]
].mean()

print("\n--- Annual Evaluation Summary ---")
print(annual_summary.round(2))

# Save annual evaluation results
annual_summary.to_csv(
    "Part_B_Demand/annual_evaluation.csv"
)

print("Annual evaluation results saved successfully!")

print("\n--- Daily Evaluation Summary ---")
print(results_df)

# Calculate average daily errors for January 2015
average_mae = results_df["MAE"].mean()
average_mape = results_df["MAPE"].mean()

print("\n--- Overall Evaluation Summary (2015 - 2026) ---")
print(f"Average Daily MAE: {average_mae:.2f} MW")
print(f"Average Daily MAPE: {average_mape:.2f}%")

# Calculate average Yesterday Baseline errors
average_yesterday_mae = results_df["Yesterday_MAE"].mean()
average_yesterday_mape = results_df["Yesterday_MAPE"].mean()

print("\n--- Yesterday Baseline Monthly Summary ---")
print(f"Average Daily MAE: {average_yesterday_mae:.2f} MW")
print(f"Average Daily MAPE: {average_yesterday_mape:.2f}%")

# Compare Random Forest against Yesterday Baseline
mae_improvement = (
    (average_yesterday_mae - average_mae)
    / average_yesterday_mae
) * 100

print("\n--- Model Improvement ---")
print(f"Random Forest MAE improvement: {mae_improvement:.2f}%")

# Calculate average Last Week Baseline errors
average_lastweek_mae = results_df["LastWeek_MAE"].mean()
average_lastweek_mape = results_df["LastWeek_MAPE"].mean()

print("\n--- Last Week Baseline Monthly Summary ---")
print(f"Average Daily MAE: {average_lastweek_mae:.2f} MW")
print(f"Average Daily MAPE: {average_lastweek_mape:.2f}%")

# Compare Random Forest against Last Week Baseline
lastweek_improvement = (
    (average_lastweek_mae - average_mae)
    / average_lastweek_mae
) * 100

print("\n--- Last Week Baseline Improvement ---")
print(f"Random Forest MAE improvement: {lastweek_improvement:.2f}%")

# Save daily evaluation results
results_df.to_csv(
    "Part_B_Demand/daily_evaluation.csv",
    index=False
)

print("Daily evaluation results saved successfully!")

print("\n--- Ridge Regression vs Baselines (2025) ---")

ridge_results = []

for day in pd.date_range("2025-01-01", "2025-12-31"):
    day_str = day.strftime("%Y-%m-%d")

    history, weather_next = make_inputs(df, day_str)

    if len(weather_next) != 48:
        continue

    X_next = get_inference_features(history, weather_next)

    if not np.isfinite(X_next.to_numpy(dtype=float)).all():
        continue

    actual = targets(df, day_str)["nsw_demand_mw"]

    if len(actual) != 48 or actual.isna().any():
        continue

    ridge_pred = ridge_model.predict(X_next)

    ridge_results.append({
        "MAE": mae(actual, ridge_pred),
        "MAPE": mape(actual, ridge_pred)
    })

ridge_results_df = pd.DataFrame(ridge_results)

print(f"Evaluated days: {len(ridge_results_df)}")
print(f"Ridge MAE: {ridge_results_df['MAE'].mean():.2f} MW")
print(f"Ridge MAPE: {ridge_results_df['MAPE'].mean():.2f}%")
