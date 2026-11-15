import time
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

from Part_C_Price_Direction.train import load_classification_data
from Part_C_Price_Direction.models import create_random_forest


def main():

    # Load the chronological datasets
    X_train, y_train, _, X_val, y_val, _ = load_classification_data()

    # Hyperparameter configurations
    configurations = [
        {"n_estimators": trees, "max_depth": depth, "min_samples_leaf": leaf}
        for trees in [100, 200]
        for depth in [10, None]
        for leaf in [5, 10]
    ]

    results = []

    for i, params in enumerate(configurations, start=1):

        print(f"\nExperiment {i}/{len(configurations)}")
        print("Parameters:", params)

        model = create_random_forest()
        model.set_params(**params)

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        predictions = model.predict(X_val)

        result = {
            **params,
            "accuracy": accuracy_score(y_val, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_val, predictions),
            "precision": precision_score(y_val, predictions, zero_division=0),
            "recall": recall_score(y_val, predictions, zero_division=0),
            "training_time_seconds": training_time,
        }

        results.append(result)

        print(f"Balanced accuracy: {result['balanced_accuracy']:.4f}")
        print(f"Training time: {training_time:.2f} seconds")

    # Sort by balanced accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="balanced_accuracy",
        ascending=False,
    )

    # Save experiment results
    output_path = "Part_C_Price_Direction/random_forest_tuning.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFINAL RANDOM FOREST TUNING RESULTS")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
