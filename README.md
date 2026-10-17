# London Weather Temperature Estimation with MLflow

Python machine learning project comparing three scikit-learn regressors for **same-day London mean temperature estimation**, with model artifacts, hyperparameters and test-set RMSE tracked in MLflow.

## Verified DataCamp results

The following scores were observed in the DataCamp notebook using a chronologically ordered 80/20 train/test split:

| Model | Test RMSE (°C) |
| --- | ---: |
| Random Forest | **0.906931** |
| Linear Regression | 0.909923 |
| Decision Tree (max_depth=10) | 1.074221 |

All three satisfied the project's target of **RMSE ≤ 3 °C**. Training: **12,244** observations; testing: **3,061** observations. Dates span 1979-01-01 through 2020-12-31. These are recorded notebook results, not independently reproduced runs of this repository.

**Important limitation:** The features include `min_temp` and `max_temp` from the same day. This is therefore *same-day temperature estimation*, **not** a next-day or future weather forecast. RMSE should not be interpreted as prospective forecasting performance.

## Workflow

1. Load `london_weather.csv`, parse `date` as `%Y%m%d`, and sort chronologically.
2. Drop records missing the target `mean_temp`.
3. Derive calendar month/year and select weather-related features.
4. Hold out the last 20% of records for evaluation.
5. Impute missing feature values with the training-data median inside a scikit-learn pipeline.
6. Train Linear Regression, Decision Tree, and Random Forest models.
7. Log each model's parameters, test RMSE and pipeline artifact with MLflow.
8. Retrieve all runs into `experiment_results` using `mlflow.search_runs()` and rank by RMSE.

## Run locally

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python london_weather_mlflow.py
```

**Dataset:** This repository does **not** include `london_weather.csv`. Download/export it from your authorized DataCamp project environment, if permitted by its dataset license, and put it alongside `london_weather_mlflow.py`. The script will explain this requirement if the file is absent.

MLflow stores run information locally by default. If installed, you can inspect it with:

```bash
mlflow ui
```

Then open the URL printed by MLflow.

## Skills

- Python and pandas data preprocessing
- Time-ordered train/test evaluation
- scikit-learn pipelines and regression
- Missing-value imputation without train/test leakage
- RMSE-based model comparison
- MLflow experiment tracking and model artifacts

## Attribution

Created as a portfolio adaptation of a DataCamp London weather regression and MLflow project. Original exercise and dataset are associated with DataCamp. The dataset is omitted pending redistribution permission.
