import joblib
import numpy as np
from pathlib import Path

from Part_A.features import FeatureConfig, get_features
from Part_C_Price_Direction.predict_price import predict_price


def main():

    print("Generating features for 1 January 2015...")

    model_path = Path("Part_C_Price_Direction/artifacts") / "random_forest.joblib"

    config = FeatureConfig()

    X, y, meta = get_features(
        start="2015-01-01",
        end="2015-01-01",
        config=config,
    )

    print("\n--- Input Information ---")
    print("Input shape:", X.shape)
    print("Number of features:", X.shape[1])

    print("\n--- Generating Predictions ---")

    predictions = predict_price(X)

    model = joblib.load(model_path)

    direct_predictions = model.predict(X)

    assert np.array_equal(
        predictions, direct_predictions
    ), "Interface predictions do not match direct model predictions"

    print("Interface matches model: True")

    print("Prediction shape:", predictions.shape)
    print("Unique predictions:", set(predictions))

    print("\nFirst 10 predictions:")
    print(predictions[:10])

    assert X.shape == (48, 47), "Unexpected feature shape"
    assert predictions.shape == (48,), "Expected 48 predictions"
    assert set(predictions).issubset({0, 1}), "Invalid prediction values"

    print("\nSUCCESS: Price prediction interface works!")


if __name__ == "__main__":
    main()
