"""Estimate London's daily mean temperature and track models with MLflow.

This is same-day estimation: maximum and minimum temperatures are used as
predictors, so results must not be described as future-day forecasts.
"""
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

DATA_FILE = Path(__file__).parent / "london_weather.csv"
FEATURES = [
    "cloud_cover", "sunshine", "global_radiation",
    "max_temp", "min_temp", "precipitation",
    "pressure", "snow_depth", "month", "year",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE.name}. Obtain the dataset from the "
            "DataCamp project and place it next to this script, "
            "subject to the dataset's license."
        )

    weather = pd.read_csv(DATA_FILE)
    weather["date"] = pd.to_datetime(
        weather["date"].astype("string"), format="%Y%m%d"
    )
    weather = (
        weather.dropna(subset=["mean_temp"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    weather["month"] = weather["date"].dt.month
    weather["year"] = weather["date"].dt.year

    X = weather[FEATURES]
    y = weather["mean_temp"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    mlflow.set_experiment("london-weather-temperature-estimation")

    for model_name, estimator in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("shuffle", False)
            for name, value in estimator.get_params().items():
                mlflow.log_param(name, value)
            mlflow.log_metric("rmse", rmse)
            # Use artifact_path for compatibility with the DataCamp MLflow setup.
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            print(f"{model_name}: RMSE = {rmse:.3f} °C")

    experiment_results = mlflow.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(
            "london-weather-temperature-estimation"
        ).experiment_id]
    )
    sorted_results = experiment_results.sort_values(
        "metrics.rmse", ascending=True
    )
    print("\n=== EXPERIMENT RESULTS ===")
    print(sorted_results[["tags.mlflow.runName", "metrics.rmse"]])
    best_rmse = sorted_results.iloc[0]["metrics.rmse"]
    print(f"\nBest RMSE: {best_rmse:.3f} °C")
    print(f"Target (RMSE <= 3 °C) met: {best_rmse <= 3.0}")


if __name__ == "__main__":
    main()
