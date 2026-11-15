# Part C — Price Direction Classification

## Overview

Part C focuses on predicting the direction of NSW electricity prices for the next market day using binary classification models.

The target variable is `nsw_price_up`:

- `1`: The next interval's NSW price is higher than the previous market day's mean NSW price.
- `0`: Otherwise.

Each market day contains 48 half-hourly intervals, so the model produces 48 binary predictions for each forecast day.

## Project Structure

```text
Part_C_Price_Direction/
├── artifacts/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
├── results/
├── tests/
│   └── test_predict_price.py
├── models.py
├── train.py
├── refit_models.py
├── evaluate_frozen_models.py
├── predict_price.py
├── model_analysis.ipynb
├── tune_logistic_regression.py
├── tune_random_forest.py
├── logistic_regression_tuning.csv
└── random_forest_tuning.csv
```

## Models

Two classification models were developed and evaluated:

1. **Logistic Regression**
   - Uses a StandardScaler preprocessing pipeline.
   - Provides a simpler linear classification baseline.

2. **Random Forest**
   - Uses an ensemble of decision trees.
   - Captures nonlinear relationships between input features.

Both models were tuned using the 2010–2013 training data and 2014 validation data.

The selected hyperparameters and model configurations are defined in `models.py`.

## Training and Model Selection

The chronological training and validation protocol follows the project specification:

| Dataset    | Period    | Purpose                                      |
| ---------- | --------- | -------------------------------------------- |
| Training   | 2010–2013 | Model training and hyperparameter tuning     |
| Validation | 2014      | Model selection                              |
| Refit      | 2010–2014 | Final model fitting before frozen evaluation |

After model selection, both models were refitted using the combined 2010–2014 dataset.

The trained models are saved in `artifacts/`.

## Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Balanced Accuracy
- Precision
- Recall

Frozen model evaluation was conducted chronologically from January 2015 to October 2026.

The evaluation also compares the classifiers against two baseline methods:

- Yesterday's same-half-hour price direction.
- Last week's same-half-hour price direction.

Further analysis includes:

- Overall model performance.
- Yearly accuracy and balanced accuracy.
- Yearly class distribution.
- Monthly accuracy confidence intervals.
- Independent half-hour and day-block bootstrap confidence intervals.
- Actual versus predicted price direction.

Detailed analysis and visualisations are available in `model_analysis.ipynb`.

## Prediction Interface

The module `predict_price.py` provides a reusable prediction interface for integration with Part E.

### Function

```python
predict_price(X_next, model=None)
```

**Input:**

- `X_next`: A feature DataFrame containing 48 rows and 47 features for the next market day.
- `model`: An optional fitted classification model. If omitted, the default saved model is loaded.

**Output:**

- A NumPy array containing 48 binary predictions (`0` or `1`).

### Example Usage

```python
from Part_C_Price_Direction.predict_price import predict_price

predictions = predict_price(X_next)

print(predictions.shape)  # (48,)
```

To use an already loaded or retrained model:

```python
predictions = predict_price(X_next, model=trained_model)
```

This allows Part E to reuse the same prediction interface during chronological backtesting and model adaptation without loading the model from disk for every forecast day.

## Testing

The prediction interface can be tested from the project root:

```bash
python -m Part_C_Price_Direction.tests.test_predict_price
```

The test verifies:

- The feature input has the expected shape `(48, 47)`.
- The prediction output has the expected shape `(48,)`.
- Predictions contain only binary values (`0` or `1`).
- Interface predictions match predictions from the directly loaded model.

## Model Artifacts

The trained model files are stored in:

```text
Part_C_Price_Direction/artifacts/
```

The required artifacts are:

- `logistic_regression.joblib`
- `random_forest.joblib`

Artifact distribution and repository tracking instructions will be finalised during team integration.
