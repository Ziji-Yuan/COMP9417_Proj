import pandas as pd

from Part_A.features import FeatureConfig, get_features
from Part_C_Price_Direction.models import (
    create_logistic_regression,
    create_random_forest,
)


def load_classification_data():
    """Load chronological training and validation datasets."""

    config = FeatureConfig()

    # Training period: 2010–2013
    X_train, y_train, meta_train = get_features(
        start="2010-01-01", end="2013-12-31", config=config
    )

    # Validation period: 2014
    X_val, y_val, meta_val = get_features(
        start="2014-01-01", end="2014-12-31", config=config
    )

    # Extract classification target
    y_train = y_train["nsw_price_up"].astype(int)
    y_val = y_val["nsw_price_up"].astype(int)

    return X_train, y_train, meta_train, X_val, y_val, meta_val


if __name__ == "__main__":

    X_train, y_train, meta_train, X_val, y_val, meta_val = load_classification_data()

    print("Training dataset:", X_train.shape)
    print("Validation dataset:", X_val.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nValidation class distribution:")
    print(y_val.value_counts(normalize=True))
