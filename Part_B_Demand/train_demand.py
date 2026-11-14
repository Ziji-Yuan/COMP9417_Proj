# Part B — NSW Electricity Demand Regression

# Feature engineering from Part A
from Part_A.features import get_features

# Regression models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


print("Part B imports successful!")

# Load training data: 2010–2013
X_train, y_train, meta_train = get_features(
    start="2010-01-01",
    end="2013-12-31"
)

# Load validation data: 2014
X_val, y_val, meta_val = get_features(
    start="2014-01-01",
    end="2014-12-31"
)

# Select NSW demand as our regression target
y_train = y_train["nsw_demand_mw"]
y_val = y_val["nsw_demand_mw"]

# Load combined training data: 2010–2014
X_final, y_final, meta_final = get_features(
    start="2010-01-01",
    end="2014-12-31"
)

# Select NSW demand target
y_final = y_final["nsw_demand_mw"]

print("Final training features:", X_final.shape)
print("Final training target:", y_final.shape)

print("Training features:", X_train.shape)
print("Training target:", y_train.shape)
print("Validation features:", X_val.shape)
print("Validation target:", y_val.shape)

# Model 1: Ridge Regression
#ridge_model = Ridge(alpha=1.0)
ridge_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1000.0)
)

# Train the model using 2010–2013 data
ridge_model.fit(X_train, y_train)

# Predict NSW demand for 2014
ridge_pred = ridge_model.predict(X_val)

print("Ridge Regression training completed!")
print("First 5 predictions:", ridge_pred[:5])
print("First 5 actual values:", y_val.iloc[:5].values)

# Evaluate Ridge Regression
ridge_mae = mean_absolute_error(y_val, ridge_pred)
ridge_mape = mean_absolute_percentage_error(y_val, ridge_pred) * 100

print("\n--- Ridge Regression Evaluation ---")
print(f"MAE: {ridge_mae:.2f} MW")
print(f"MAPE: {ridge_mape:.2f}%")

# Baseline 1: Same half-hour yesterday
yesterday_pred = X_val["nsw_demand_lag_1d"]

yesterday_mae = mean_absolute_error(y_val, yesterday_pred)
yesterday_mape = mean_absolute_percentage_error(
    y_val, yesterday_pred
) * 100

print("\n--- Yesterday Baseline ---")
print(f"MAE: {yesterday_mae:.2f} MW")
print(f"MAPE: {yesterday_mape:.2f}%")

# Baseline 2: Same half-hour last week
lastweek_pred = X_val["nsw_demand_lag_7d"]

lastweek_mae = mean_absolute_error(y_val, lastweek_pred)
lastweek_mape = mean_absolute_percentage_error(
    y_val, lastweek_pred
) * 100

print("\n--- Last Week Baseline ---")
print(f"MAE: {lastweek_mae:.2f} MW")
print(f"MAPE: {lastweek_mape:.2f}%")

# Model 2: Random Forest Regression
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict 2014 demand
rf_pred = rf_model.predict(X_val)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_val, rf_pred)
rf_mape = mean_absolute_percentage_error(y_val, rf_pred) * 100

print("\n--- Random Forest Evaluation ---")
print(f"MAE: {rf_mae:.2f} MW")
print(f"MAPE: {rf_mape:.2f}%")

from pathlib import Path
import joblib

# Create folder for trained models
artifact_dir = Path("Part_B_Demand/artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)

# Refit selected models using all 2010–2014 data
ridge_model.fit(X_final, y_final)
rf_model.fit(X_final, y_final)

print("Final models trained on 2010–2014 successfully!")

# Save trained models
joblib.dump(ridge_model, artifact_dir / "ridge_model.joblib")
joblib.dump(rf_model, artifact_dir / "rf_model.joblib")

print("Both regression models saved successfully!")