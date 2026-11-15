import joblib
import numpy as np
from pathlib import Path

from Part_A.features import FeatureConfig, get_features


def main():

    # Load saved models
    artifact_dir = Path("Part_C_Price_Direction/artifacts")

    models = {
        "Logistic Regression": joblib.load(artifact_dir / "logistic_regression.joblib"),
        "Random Forest": joblib.load(artifact_dir / "random_forest.joblib"),
    }

    print("Both models loaded successfully.")

    # Generate January 2015 feature data
    config = FeatureConfig()

    X, y, meta = get_features(
        start="2015-01-01",
        end="2015-01-31",
        config=config,
    )

    print("\nTest dataset shape:", X.shape)

    # Check feature dimensions
    assert X.shape[1] == 47, "Unexpected number of features!"

    expected_predictions = len(X)

    # Test predictions
    for name, model in models.items():

        predictions = model.predict(X)

        # Validate prediction output
        assert (
            len(predictions) == expected_predictions
        ), f"{name} produced an unexpected number of predictions!"

        assert set(np.unique(predictions)).issubset(
            {0, 1}
        ), f"{name} produced invalid class labels!"

        print(f"\n{name}")
        print("Number of predictions:", len(predictions))

        print("Predicted class distribution:")
        print(
            {
                int(label): int(count)
                for label, count in zip(*np.unique(predictions, return_counts=True))
            }
        )

        if name == "Logistic Regression":

            probabilities = model.predict_proba(X)[:, 1]

            print("\nPositive probability statistics:")
            print(f"Minimum: {probabilities.min():.4f}")
            print(f"Maximum: {probabilities.max():.4f}")
            print(f"Mean: {probabilities.mean():.4f}")
            print(f"Median: {np.median(probabilities):.4f}")

        print("\nPredictions (first 48 intervals):")
        print(predictions[:48])

    print("\nAll saved model tests passed.")


if __name__ == "__main__":
    main()
