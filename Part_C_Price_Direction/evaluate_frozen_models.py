import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from Part_A.features import FeatureConfig, get_features

ARTIFACT_DIR = Path("Part_C_Price_Direction/artifacts")
OUTPUT_DIR = Path("Part_C_Price_Direction/results")

MODEL_FILES = {
    "logistic_regression": ARTIFACT_DIR / "logistic_regression.joblib",
    "random_forest": ARTIFACT_DIR / "random_forest.joblib",
}


def load_models():
    """Load the frozen Stage 1 models."""
    models = {}

    for name, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        models[name] = joblib.load(path)
        print(f"Loaded {name}: {path}")

    return models


def evaluate_predictions(y_true, predictions):
    """Calculate accuracy and balanced accuracy safely."""

    y_true = pd.Series(y_true).astype(int)
    predictions = pd.Series(predictions).astype(int)

    recalls = []

    for label in (0, 1):
        mask = y_true == label

        if mask.any():
            recall = (predictions[mask] == label).mean()
            recalls.append(recall)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": sum(recalls) / len(recalls),
    }


def summarise_results(df, group_column):
    """Aggregate interval-level predictions by a time period."""
    prediction_columns = [
        "logistic_regression",
        "random_forest",
        "yesterday_baseline",
        "last_week_baseline",
    ]

    rows = []

    for period, group in df.groupby(group_column, sort=True):
        y_true = group["actual"].astype(int)

        row = {
            group_column: period,
            "observations": len(group),
            "class_0_count": int((y_true == 0).sum()),
            "class_1_count": int((y_true == 1).sum()),
        }

        for method in prediction_columns:
            metrics = evaluate_predictions(
                y_true,
                group[method].astype(int),
            )

            row[f"{method}_accuracy"] = metrics["accuracy"]
            row[f"{method}_balanced_accuracy"] = metrics["balanced_accuracy"]

        rows.append(row)

    return pd.DataFrame(rows)


def run_evaluation(start_date, end_date):
    print("\nLoading frozen models...")
    models = load_models()

    print("\nLoading historical features...")
    X, y, meta = get_features(
        start=start_date,
        end=end_date,
        config=FeatureConfig(),
    )

    # Check feature compatibility with the fitted models.
    for name, model in models.items():
        expected = getattr(model, "feature_names_in_", None)

        if expected is not None and list(expected) != list(X.columns):
            raise ValueError(f"Feature columns do not match for {name}.")

    print("Feature matrix:", X.shape)
    print("Target matrix:", y.shape)

    # Align the actual labels and metadata with the feature rows.
    actual = y["nsw_price_up"].astype(int)

    predictions = pd.DataFrame(index=X.index)
    predictions["timestamp"] = meta["timestamp"]
    predictions["market_day"] = meta["market_day"]
    predictions["slot"] = meta["slot"]
    predictions["actual"] = actual

    # Generate predictions chronologically, one market day at a time.
    print("\nGenerating predictions...")

    for market_day, day_index in meta.groupby("market_day", sort=True).groups.items():

        X_day = X.loc[day_index]

        if len(X_day) != 48:
            raise ValueError(f"{market_day}: expected 48 rows, got {len(X_day)}")

        # Frozen model predictions: no fitting or updating.
        for name, model in models.items():
            predictions.loc[day_index, name] = model.predict(X_day)

        # Baselines use the same previous-day price reference.
        reference = meta.loc[day_index, "price_reference_prev_day_mean"]

        predictions.loc[day_index, "yesterday_baseline"] = (
            X.loc[day_index, "nsw_price_lag_1d"].to_numpy() > reference.to_numpy()
        ).astype(int)

        predictions.loc[day_index, "last_week_baseline"] = (
            X.loc[day_index, "nsw_price_lag_7d"].to_numpy() > reference.to_numpy()
        ).astype(int)

    prediction_columns = [
        "logistic_regression",
        "random_forest",
        "yesterday_baseline",
        "last_week_baseline",
    ]

    predictions[prediction_columns] = predictions[prediction_columns].astype(int)

    print("\nAll predictions generated.")
    print("Prediction rows:", len(predictions))

    # Create output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save interval-level predictions.
    date_label = f"{start_date}_to_{end_date}"
    interval_path = OUTPUT_DIR / f"interval_predictions_{date_label}.csv"

    predictions.to_csv(interval_path, index=False)

    # Generate daily, monthly and yearly summaries.
    predictions["month"] = (
        pd.to_datetime(predictions["market_day"]).dt.to_period("M").astype(str)
    )
    predictions["year"] = pd.to_datetime(predictions["market_day"]).dt.year

    daily = summarise_results(predictions, "market_day")
    monthly = summarise_results(predictions, "month")
    yearly = summarise_results(predictions, "year")

    daily_path = OUTPUT_DIR / f"daily_metrics_{date_label}.csv"
    monthly_path = OUTPUT_DIR / f"monthly_metrics_{date_label}.csv"
    yearly_path = OUTPUT_DIR / f"yearly_metrics_{date_label}.csv"

    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    yearly.to_csv(yearly_path, index=False)

    print("\nOverall results:")

    for method in prediction_columns:
        metrics = evaluate_predictions(
            predictions["actual"],
            predictions[method],
        )

        print(
            f"{method}: "
            f"Accuracy = {metrics['accuracy']:.4f}, "
            f"Balanced Accuracy = "
            f"{metrics['balanced_accuracy']:.4f}"
        )

    print("\nSaved results:")
    print(interval_path)
    print(daily_path)
    print(monthly_path)
    print(yearly_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Stage 1 models chronologically."
    )

    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="First market day (inclusive).",
    )
    parser.add_argument(
        "--end",
        default="2015-01-31",
        help="Last market day (inclusive).",
    )

    args = parser.parse_args()

    run_evaluation(args.start, args.end)


if __name__ == "__main__":
    main()
