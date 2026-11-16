from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("Part_C_Price_Direction/results")

INPUT_PATH = RESULTS_DIR / "interval_predictions_2015-01-01_to_2026-10-31.csv"

OUTPUT_PATH = RESULTS_DIR / "price_predictions_for_part_d.csv"


def main():
    predictions = pd.read_csv(INPUT_PATH)

    model_columns = [
        "logistic_regression",
        "random_forest",
    ]

    required_columns = [
        "timestamp",
        "actual",
        *model_columns,
    ]

    missing = set(required_columns) - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert wide predictions into long format.
    long_predictions = predictions.melt(
        id_vars=["timestamp", "actual"],
        value_vars=model_columns,
        var_name="model",
        value_name="y_pred",
    )

    long_predictions = long_predictions.rename(columns={"actual": "y_true"})

    long_predictions["task"] = "price"
    long_predictions["strategy"] = "frozen"

    # Keep exactly the columns expected by Part D.
    long_predictions = long_predictions[
        [
            "timestamp",
            "task",
            "model",
            "strategy",
            "y_true",
            "y_pred",
        ]
    ]

    long_predictions["y_true"] = long_predictions["y_true"].astype(int)
    long_predictions["y_pred"] = long_predictions["y_pred"].astype(int)

    # Basic validation.
    if long_predictions.duplicated(["timestamp", "model", "strategy"]).any():
        raise ValueError("Duplicate prediction rows found.")

    if not long_predictions["y_pred"].isin([0, 1]).all():
        raise ValueError("Invalid price prediction labels.")

    expected_rows = len(predictions) * len(model_columns)

    if len(long_predictions) != expected_rows:
        raise ValueError("Unexpected number of exported rows.")

    long_predictions.to_csv(OUTPUT_PATH, index=False)

    print("Export successful!")
    print("Output:", OUTPUT_PATH)
    print("Rows:", len(long_predictions))
    print("Models:", long_predictions["model"].unique())
    print(
        "Date range:",
        long_predictions["timestamp"].min(),
        "to",
        long_predictions["timestamp"].max(),
    )


if __name__ == "__main__":
    main()
