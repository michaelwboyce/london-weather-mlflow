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

Requires Python 3.10+ and `london_weather.csv` in the repository directory.

```bash
python -m pip install -r requirements.txt
python london_weather_mlflow.py
```

If the CSV has not yet been uploaded to this repository, obtain it from the authorized DataCamp project and place it alongside the script.

MLflow stores run information locally by default. To inspect it:

```bash
mlflow ui
```

Then open the URL printed by MLflow.

## Dataset and attribution

Data: London Weather Data, associated with the [Kaggle London Weather Data collection](https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data) by Emmanuel F. Werr (reported as **CC0 / public domain**), compiled from historical weather observations. The project file contains 15,341 rows and 10 columns covering 1979–2020. The CSV was supplied from the [DataCamp Predicting Temperature in London project](https://www.datacamp.com/projects/1725). The uploaded CSV's exact byte-for-byte identity with Kaggle's version has not been independently verified.

The Python workflow is a portfolio adaptation of the DataCamp MLflow exercise. Please retain source attribution when redistributing the dataset.

## Skills

- Python and pandas data preprocessing
- Time-ordered train/test evaluation
- scikit-learn pipelines and regression
- Missing-value imputation without train/test leakage
- RMSE-based model comparison
- MLflow experiment tracking and model artifacts
