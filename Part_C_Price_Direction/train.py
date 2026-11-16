import pandas as pd
import time

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

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


def evaluate_model(name, model, X_val, y_val):
    """Evaluate a trained classifier on the validation dataset."""

    y_pred = model.predict(X_val)

    results = {
        "model": name,
        "accuracy": accuracy_score(y_val, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
    }

    print(f"\n{'=' * 50}")
    print(name)
    print(f"{'=' * 50}")

    for metric, value in results.items():
        if metric != "model":
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    return results


if __name__ == "__main__":

    X_train, y_train, meta_train, X_val, y_val, meta_val = load_classification_data()

    print("Training dataset:", X_train.shape)
    print("Validation dataset:", X_val.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nValidation class distribution:")
    print(y_val.value_counts(normalize=True))

    # Initialise models
    models = {
        "Logistic Regression": create_logistic_regression(),
        "Random Forest": create_random_forest(),
    }

    results = []

    # Train and evaluate each model
    for name, model in models.items():

        print(f"\n Training {name}...")

        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        print(f"Training time: {training_time:.2f} seconds")

        model_results = evaluate_model(name, model, X_val, y_val)

        model_results["training_time_seconds"] = training_time

        results.append(model_results)

    # Compare models
    results_df = pd.DataFrame(results)

    print("\n\nMODEL COMPARISON")
    print(results_df.to_string(index=False))
