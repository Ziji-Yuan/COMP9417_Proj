from pathlib import Path
import joblib

# Location of the saved Random Forest model
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "rf_model.joblib"


def load_demand_model():
    """Load the trained NSW demand regression model."""
    model = joblib.load(MODEL_PATH)
    return model

def predict_demand(X_next):
    """Predict NSW demand for the next 48 half-hour intervals."""

    if len(X_next) != 48:
        raise ValueError("Expected exactly 48 rows of features")

    model = load_demand_model()
    predictions = model.predict(X_next)

    return predictions
