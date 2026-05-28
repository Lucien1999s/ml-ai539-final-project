# Gotham Cabs

This folder contains the modeling pipeline for the Gotham Cabs prediction task in the AI 539 final project.

## Task Overview

Gotham Cabs is a taxi trip duration prediction task. Given pickup time, passenger count, pickup location, and dropoff location, the goal is to predict the trip duration.

Target variable:

    duration

Raw training columns:

    pickup_datetime
    NumberOfPassengers
    duration
    pickup_x
    pickup_y
    dropoff_x
    dropoff_y

Raw test columns:

    pickup_datetime
    NumberOfPassengers
    pickup_x
    pickup_y
    dropoff_x
    dropoff_y

## Local Data Layout

Data files should be stored locally under:

    data/gotham/
    ├── Train.csv
    ├── TestFileTemplate.csv
    └── OfficialTest.csv

The data/ directory is not tracked by Git.

## Files

    gotham/
    ├── ablation.py
    ├── eda.py
    ├── predict.py
    ├── train.py
    ├── tune_xgboost.py
    ├── validate.py
    └── src/
        ├── __init__.py
        ├── data.py
        ├── features.py
        ├── metrics.py
        ├── models.py
        └── utils.py

## Feature Engineering

The pipeline derives additional features from the raw input columns:

- Time features: pickup hour, day of week, month, day, weekend flag, rush-hour flag, late-night flag.
- Cyclic time features: sine/cosine encodings for hour and day of week.
- Trip geometry features: dx, dy, absolute dx/dy, Euclidean distance, Manhattan distance, squared distance, log distance, direction angle.
- Spatial features: average route location, pickup/dropoff distance from origin, pickup/dropoff grid zones, route-zone hash.
- Passenger features: clipped passenger count and abnormal passenger-count flag.

## Setup

Install dependencies from the repository root:

    pip install -r requirements.txt

## Run EDA

    python gotham/eda.py \
      --train-path data/gotham/Train.csv \
      --target duration \
      --task-type regression

EDA outputs are saved under:

    gotham/outputs/eda/

## Validate a Candidate Model

Run holdout validation for a selected model:

    python gotham/validate.py \
      --train-path data/gotham/Train.csv \
      --target duration \
      --task-type regression \
      --model xgboost \
      --max-train-rows 300000

Supported regression model names:

    dummy_mean
    ridge
    random_forest
    extra_trees
    lightgbm
    xgboost

## Tune XGBoost

    python gotham/tune_xgboost.py \
      --train-path data/gotham/Train.csv \
      --target duration \
      --task-type regression \
      --max-train-rows 300000

Tuning outputs are saved under:

    gotham/outputs/tuning/

## Run Feature Ablation

    python gotham/ablation.py \
      --train-path data/gotham/Train.csv \
      --target duration \
      --task-type regression \
      --model xgboost \
      --max-train-rows 300000 \
      --output-dir gotham/outputs/ablation

Ablation outputs are saved under:

    gotham/outputs/ablation/

## Train Final Model

The final model is trained on the full Train.csv dataset after model comparison and tuning.

    python gotham/train.py \
      --train-path data/gotham/Train.csv \
      --target duration \
      --task-type regression \
      --sample-frac 0.10 \
      --max-sample-rows 100000 \
      --cv 3 \
      --final-model xgboost

Training outputs are saved under:

    gotham/outputs/
    ├── leaderboard.csv
    ├── metadata.json
    └── model.joblib

## Predict Official Test File

When the official test file is released, place it under:

    data/gotham/OfficialTest.csv

Then run:

    python gotham/predict.py \
      --test-path data/gotham/OfficialTest.csv \
      --prediction-column duration \
      --keep-test-columns \
      --output-path submissions/gotham_predictions.csv

## Check Prediction File

Before submission, inspect the output CSV:

    python - <<'PY'
    import pandas as pd

    df = pd.read_csv("submissions/gotham_predictions.csv")
    print(df.shape)
    print(df.head())
    print(df["duration"].describe())
    print("Negative predictions:", (df["duration"] < 0).sum())
    print("Missing predictions:", df["duration"].isna().sum())
    PY

The prediction file should contain the original test columns plus the predicted duration column.

## Notes

- Model comparison can be run on a subset of the data for speed.
- The final competition model should be trained on the full training dataset.
- The prediction script applies the same feature engineering used during training.
- Generated outputs, local data files, model files, and prediction CSV files are not tracked by Git.
