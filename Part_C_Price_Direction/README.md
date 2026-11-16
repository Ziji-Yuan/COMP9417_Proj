# Part C — Price Direction Classification

## 1. Overview

Part C focuses on predicting the direction of NSW electricity prices for the next market day using binary classification models.

The target variable is `nsw_price_up`:

- `1`: The next interval's NSW price is higher than the previous market day's mean NSW price.
- `0`: Otherwise.

Each market day contains 48 half-hourly intervals, so the model produces 48 binary predictions for each forecast day.

The module uses the shared feature engineering pipeline from Part A and provides prediction outputs and reusable interfaces for integration with Part D (Drift Detection) and Part E (Backtesting and Adaptation).

## 2. Project Structure

```text
Part_C_Price_Direction/
├── artifacts/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
├── results/
│   └── price_predictions_for_part_d.csv
├── tests/
│   └── test_predict_price.py
├── models.py
├── train.py
├── refit_models.py
├── evaluate_frozen_models.py
├── export_part_d_predictions.py
├── predict_price.py
├── model_analysis.ipynb
├── tune_logistic_regression.py
├── tune_random_forest.py
├── logistic_regression_tuning.csv
└── random_forest_tuning.csv
```

## 3. Classification Models

Two classification models were developed and evaluated.

### 3.1 Logistic Regression

Logistic Regression is a linear classification model that estimates the probability of a binary outcome.

- Uses a StandardScaler preprocessing pipeline.
- Regularisation parameter: `C = 100`.
- Maximum iterations: 2,000.
- Random state: 8.

### 3.2 Random Forest

Random Forest combines multiple decision trees to capture nonlinear relationships between input features and electricity price direction.

- Number of trees: 200.
- Maximum tree depth: None.
- Minimum samples per leaf: 5.
- Maximum features: `sqrt`.
- Random state: 8.
- Parallel processing: `n_jobs = -1`.

Both models were tuned using the 2010–2013 training data and 2014 validation data. The selected hyperparameters and model configurations are defined in `models.py`.

## 4. Training and Model Selection

The chronological training and validation protocol follows the project specification.

| Dataset    | Period    | Purpose                                      |
| ---------- | --------- | -------------------------------------------- |
| Training   | 2010–2013 | Model training and hyperparameter tuning     |
| Validation | 2014      | Model selection                              |
| Refit      | 2010–2014 | Final model fitting before frozen evaluation |

### 4.1 Validation Results (2014)

Both models were evaluated using the 2014 validation dataset.

| Model               | Accuracy | Balanced Accuracy | Precision | Recall |
| ------------------- | -------: | ----------------: | --------: | -----: |
| Logistic Regression |   0.7175 |            0.7132 |    0.7997 | 0.5590 |
| Random Forest       |   0.7950 |            0.7948 |    0.7906 | 0.7868 |

Random Forest achieved higher validation accuracy and balanced accuracy than Logistic Regression. It was therefore selected as the primary model for the subsequent adaptation stage.

### 4.2 Final Model Training

After model selection, both models were refitted using the combined 2010–2014 dataset.

The trained models are saved in:

- `artifacts/logistic_regression.joblib`
- `artifacts/random_forest.joblib`

Random Forest is the selected model for the adaptation experiments, while Logistic Regression is retained as a secondary comparison model.

## 5. Frozen Model Evaluation

The final refitted models were evaluated chronologically from January 2015 to October 2026.

The evaluation dataset contains:

- 4,322 market days.
- 48 half-hourly intervals per market day.
- 207,456 observations per model.

The models were evaluated without retraining or updating their parameters during this period.

### 5.1 Evaluation Metrics

The following classification metrics were used:

- **Accuracy:** Proportion of correctly classified intervals.
- **Balanced Accuracy:** Average recall across the two classes, accounting for class imbalance.
- **Precision:** Proportion of positive predictions that were correct.
- **Recall:** Proportion of actual positive cases correctly identified.

Two historical baseline methods were also used for comparison:

- **Yesterday baseline:** Uses the previous market day's same-half-hour price relative to the previous market day's mean price.
- **Last-week baseline:** Uses the same-half-hour price from seven days earlier relative to the previous market day's mean price.

### 5.2 Overall Results (January 2015 – October 2026)

| Method              | Accuracy | Balanced Accuracy |
| ------------------- | -------: | ----------------: |
| Logistic Regression |   0.7243 |            0.7055 |
| Random Forest       |   0.6630 |            0.6357 |
| Yesterday Baseline  |   0.7381 |            0.7342 |
| Last-Week Baseline  |   0.7081 |            0.7042 |

### 5.3 General Findings

The frozen evaluation produced several important findings:

1. **Logistic Regression outperformed Random Forest over the full evaluation period.** Although Random Forest achieved better validation performance in 2014, Logistic Regression achieved higher overall accuracy and balanced accuracy during the subsequent historical evaluation.

2. **The yesterday baseline achieved the highest overall accuracy and balanced accuracy.** Both classifiers therefore struggled to consistently outperform this simple historical baseline over the full evaluation period.

3. **Random Forest experienced substantial performance deterioration over time.** Its accuracy declined from 0.7736 in 2015 to 0.5588 in the available 2026 evaluation period.

4. **Logistic Regression also experienced performance deterioration.** Its accuracy declined from 0.7465 in 2015 to 0.6552 in 2026.

5. **Performance changes were particularly noticeable from 2022 onwards.** Both classifiers experienced weaker results during this period, although the extent of deterioration differed between models.

These findings highlight the importance of evaluating models over extended chronological periods rather than relying exclusively on validation performance.

The results motivate further investigation into changing input distributions, prediction errors and potential adaptation strategies. However, performance deterioration alone does not establish the presence or cause of concept drift.

### 5.4 Evaluation Outputs

The evaluation scripts generate interval-level predictions and daily, monthly and yearly performance summaries.

Additional analysis includes:

- Monthly accuracy confidence intervals.
- Independent half-hour bootstrap confidence intervals.
- Day-block bootstrap confidence intervals.
- Actual versus predicted price direction.

Detailed analysis, visualisations, yearly comparisons and model behaviour are available in `model_analysis.ipynb`.

## 6. Prediction Interface

The module `predict_price.py` provides a reusable prediction interface for integration with Part E.

### 6.1 Function

```python
predict_price(X_next, model=None)
```

**Input:**

- `X_next`: A feature DataFrame containing 48 rows and 47 features for the next market day.
- `model`: An optional fitted classification model. If omitted, the default saved model is loaded.

**Output:**

- A NumPy array containing 48 binary predictions (`0` or `1`).
- Predictions follow the interval order of `X_next`.

### 6.2 Example Usage

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

The optional `model` argument also allows Part E to supply an updated classifier after retraining.

## 7. Integration and Handover

### 7.1 Handover to Part D

Part C provides the historical price direction predictions required by Part D for its drift and day-type analysis.

The exported file is:

```text
results/price_predictions_for_part_d.csv
```

It contains interval-level predictions from both Logistic Regression and Random Forest, covering January 2015 to October 2026.

The CSV follows Part D's required format:

`timestamp, task, model, strategy, y_true, y_pred`

The file contains 414,912 rows, with 207,456 predictions per model.

### 7.2 Handover to Part E

Part C provides the following files for integration with Part E:

- `artifacts/random_forest.joblib` — selected classification model for adaptation experiments.
- `artifacts/logistic_regression.joblib` — secondary classification model.
- `predict_price.py` — reusable prediction interface supporting both saved and externally supplied fitted models.

Part E can use these components for chronological backtesting and model adaptation.

## 8. Testing

The prediction interface can be tested from the project root:

```bash
python -m Part_C_Price_Direction.tests.test_predict_price
```

The test verifies:

- The feature input has the expected shape `(48, 47)`.
- The prediction output has the expected shape `(48,)`.
- Predictions contain only binary values (`0` or `1`).
- Interface predictions match predictions from the directly loaded model.

## 9. Model Artifacts and Dependencies

The trained model files are stored in:

```text
Part_C_Price_Direction/artifacts/
```

Required artifacts:

- `logistic_regression.joblib`
- `random_forest.joblib`

The module requires:

- Python 3.11.
- The shared Part A feature engineering module.
- The project dataset in the expected data directory.
- The Python dependencies used by the project.
- The trained model artifacts for loading saved predictions.

Artifact distribution and repository tracking should follow the team's agreed Git LFS and integration practices.
