from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def create_logistic_regression():
    """
    Create a Logistic Regression classification pipeline.

    StandardScaler standardises the input features before
    Logistic Regression learns its coefficients.
    """

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=100, max_iter=2000, random_state=8)),
        ]
    )

    return model


def create_random_forest():
    """
    Create a Random Forest classification model.

    Random Forest combines multiple decision trees
    to capture nonlinear relationships.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=8,
        n_jobs=-1,
    )

    return model
