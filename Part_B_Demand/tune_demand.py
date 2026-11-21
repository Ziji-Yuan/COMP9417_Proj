
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from Part_A.features import get_features


# ============================================================
# 1. Load training and validation data
# ============================================================

X_train, y_train, _ = get_features(
    start="2010-01-01",
    end="2013-12-31"
)

X_val, y_val, _ = get_features(
    start="2014-01-01",
    end="2014-12-31"
)

y_train = y_train["nsw_demand_mw"]
y_val = y_val["nsw_demand_mw"]

results = []


def evaluate(model, model_name, parameter, value):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    mape = mean_absolute_percentage_error(y_val, predictions) * 100

    results.append({
        "Model": model_name,
        "Parameter": parameter,
        "Value": value,
        "MAE_MW": round(mae, 4),
        "MAPE_percent": round(mape, 4),
    })

    print(
        f"{model_name} | {parameter}={value} "
        f"| MAE={mae:.2f} MW | MAPE={mape:.2f}%"
    )


# ============================================================
# 2. Ridge Regression: alpha search
# ============================================================

for alpha in [1, 100, 1000, 10000]:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    evaluate(model, "Ridge", "alpha", alpha)


# ============================================================
# 3. Random Forest: max_depth search
# Fixed: 100 trees, min_samples_leaf=5
# ============================================================

for depth in [10, 15, 20, 25]:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=depth,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    evaluate(model, "Random Forest", "max_depth", depth)


# ============================================================
# 4. Random Forest: min_samples_leaf search
# Fixed: 100 trees, max_depth=20
# ============================================================

for leaf in [1, 2, 5, 10]:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=leaf,
        random_state=42,
        n_jobs=-1
    )

    evaluate(model, "Random Forest", "min_samples_leaf", leaf)


# ============================================================
# 5. Save results
# ============================================================

results_df = pd.DataFrame(results)

output_path = Path("Part_B_Demand/demand_tuning_results.csv")
results_df.to_csv(output_path, index=False)

print("\n--- Manual Hyperparameter Tuning Summary ---")
print(results_df.to_string(index=False))

print(f"\nResults saved to: {output_path}")