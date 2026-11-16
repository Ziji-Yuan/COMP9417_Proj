# Part B — NSW Electricity Demand Regression

## 1. Overview

This module predicts electricity demand in New South Wales (NSW) for the next market day.

The model generates 48 demand predictions, with each prediction representing a 30-minute interval.

This module uses the shared feature engineering pipeline from Part A and provides predictions for integration with Part D (Drift Detection) and Part E (Backtesting).

The main objective is to achieve lower prediction errors than simple historical baselines.

## 2. Regression Models

Two regression algorithms were implemented and compared:

### Ridge Regression
Ridge Regression is a linear regression model with L2 regularisation. It helps reduce overfitting by penalising large model coefficients.

- Feature scaling: StandardScaler
- Regularisation parameter: alpha = 1000

### Random Forest Regression
Random Forest combines multiple decision trees to capture non-linear relationships between the input features and electricity demand.

- Number of trees: 100
- Maximum tree depth: 20
- Minimum samples per leaf: 2
- Random state: 42

### Model Selection
Both models were evaluated using the 2014 validation dataset.

Random Forest achieved better validation performance:

| Model | MAE (MW) | MAPE |
|---|---:|---:|
| Ridge Regression | 271.08 | 3.44% |
| Random Forest | 211.34 | 2.65% |

Therefore, Random Forest was selected as the final demand prediction model.

## 3. Training and Validation Strategy

The dataset was divided chronologically to prevent data leakage.

### Stage 1: Model Development

- Training period: January 2010 – December 2013
- Validation period: January 2014 – December 2014

Both Ridge Regression and Random Forest were trained using historical data from 2010–2013 and evaluated on the 2014 validation set.

### Final Model Training

After selecting Random Forest, the final regression models were trained using data from January 2010 to December 2014.

The trained models were saved as:

- `artifacts/ridge_model.joblib`
- `artifacts/rf_model.joblib`

The Random Forest model is used for demand prediction.

### Stage 2: Historical Evaluation

The final Random Forest model was evaluated on historical data from January 2015 to October 2026.

Predictions were generated day by day using the available historical information and the shared feature engineering functions from Part A.

This evaluation measures the performance of the initial fixed model before any drift-based retraining.

## 4. Evaluation Results

The final Random Forest model was evaluated from January 2015 to October 2026, covering 4,322 market days.

Model performance was measured using:

- Mean Absolute Error (MAE), measured in MW.
- Mean Absolute Percentage Error (MAPE), measured as a percentage.

Two historical baselines were used for comparison:

- Yesterday baseline: demand from the same half-hour interval on the previous day.
- Last-week baseline: demand from the same half-hour interval seven days earlier.

### Overall Results (2015–2026)

| Method | MAE (MW) | MAPE |
|---|---:|---:|
| Random Forest | 348.67 | 4.74% |
| Yesterday Baseline | 470.64 | 6.15% |
| Last-Week Baseline | 569.73 | 7.42% |

Random Forest reduced MAE by approximately 25.92% compared with the yesterday baseline and 38.80% compared with the last-week baseline.

These results show that Random Forest outperformed both historical baselines over the evaluation period.

### Evaluation Output

The evaluation script generates two CSV files:

- `daily_evaluation.csv`: Daily MAE and MAPE for each method.
- `annual_evaluation.csv`: Annual average MAE and MAPE for each method.

The results represent the performance of the fixed model without drift-based retraining.

## 5. How to Run

All commands should be executed from the project root directory (`COMP9417_Proj`).

### 5.1 Train the Regression Models

```bash
python -m Part_B_Demand.train_demand
```

This script:
- Loads the shared features from Part A.
- Trains Ridge Regression and Random Forest.
- Evaluates both models using the 2014 validation data.
- Retrains the final models using data from 2010–2014.
- Saves the trained models in `Part_B_Demand/artifacts/`.

### 5.2 Run Historical Evaluation

```bash
python -m Part_B_Demand.evaluate_demand
```

This script:
- Evaluates the fixed Random Forest model from 2015 to October 2026.
- Compares predictions against yesterday and last-week baselines.
- Calculates daily and annual MAE and MAPE.
- Saves evaluation results as CSV files.

### 5.3 Model Requirements

The prediction module requires:

- The shared Part A feature engineering module.
- The project dataset in the expected data directory.
- The Python dependencies used by the project.
- The trained Random Forest model at `Part_B_Demand/artifacts/rf_model.joblib`.

The Random Forest model is tracked using Git LFS. Team members should ensure Git LFS is installed when cloning or pulling the repository.

## 6. Integration with Part E (Backtesting)

Part B provides a prediction function that can be used by the backtesting module.

### Prediction Function

```python
from Part_B_Demand.predict_demand import predict_demand

# X_next contains 48 rows of features from Part A
predictions = predict_demand(X_next)
```

### Input

- `X_next`: A feature DataFrame generated by Part A.
- Expected shape: 48 rows with the same feature columns used during training.
- Each row represents one 30-minute interval of the next market day.

### Output

- A NumPy array containing 48 predicted NSW electricity demand values.
- Predictions are expressed in megawatts (MW).
- The predictions follow the interval order of `X_next`.

### Important Notes

- The function loads the trained Random Forest model from `artifacts/rf_model.joblib`.
- The model must be available locally before running predictions.
- The function does not retrain the model.
- Historical evaluation results represent the fixed model without drift adaptation.
- Part E is responsible for integrating predictions into the chronological backtesting process.

