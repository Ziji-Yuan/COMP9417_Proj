import time
import joblib
from pathlib import Path

from Part_A.features import FeatureConfig, get_features
from Part_C_Price_Direction.models import (
    create_logistic_regression,
    create_random_forest,
)


def main():

    # Load the full Stage 1 training dataset (2010–2014)
    config = FeatureConfig()

    X_train, y_train, meta_train = get_features(
        start="2010-01-01",
        end="2014-12-31",
        config=config,
    )

    # Extract classification target
    y_train = y_train["nsw_price_up"].astype(int)

    print("Training dataset:", X_train.shape)
    print("Target dataset:", y_train.shape)

    print("\nClass distribution:")
    print(y_train.value_counts(normalize=True))

    # Initialise selected models
    models = {
        "logistic_regression": create_logistic_regression(),
        "random_forest": create_random_forest(),
    }

    # Create artifacts directory
    output_dir = Path("Part_C_Price_Direction/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Refit and save models
    for name, model in models.items():

        print(f"\nTraining {name}...")

        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        # Save trained model
        output_path = output_dir / f"{name}.joblib"
        joblib.dump(model, output_path)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model saved to: {output_path}")

    print("\nBoth models successfully refitted.")


if __name__ == "__main__":
    main()
