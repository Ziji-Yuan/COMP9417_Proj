from pathlib import Path
import joblib
import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# Selected model for prediction
MODEL_PATH = ARTIFACT_DIR / "random_forest.joblib"


def load_price_model():
    """
    Load the trained price-direction classification model.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict_price(X_next, model=None):
    """
    Predict NSW electricity price direction for the next market day.

    Parameters
    ----------
    X_next : pandas.DataFrame
        Feature matrix containing 48 half-hourly intervals.

    model : optional
        Previously loaded classification model.
        If None, the model is loaded automatically.

    Returns
    -------
    numpy.ndarray
        48 binary predictions:
        1 = price above previous day's average
        0 = price at or below previous day's average
    """

    if len(X_next) != 48:
        raise ValueError(f"Expected 48 feature rows, received {len(X_next)}")

    if model is None:
        model = load_price_model()

    predictions = model.predict(X_next)

    return np.asarray(predictions, dtype=int)
