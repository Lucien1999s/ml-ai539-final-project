# Gotham Cabs

This folder contains the modeling pipeline for the Gotham Cabs prediction task in the AI 539 final project.

## Task Overview

Gotham Cabs is a taxi trip duration prediction task. Given pickup time, passenger count, pickup location, and dropoff location, the goal is to predict the trip duration.

The target variable is:

```text
duration
```

The input features in the raw training data are:

```text
pickup_datetime
NumberOfPassengers
pickup_x
pickup_y
dropoff_x
dropoff_y
```

## Local Data Layout

Data files should be stored locally under:

```text
data/gotham/
├── Train.csv
├── TestFileTemplate.csv
└── OfficialTest.csv
```

The `data/` directory is not tracked by Git.

## Files

```text
gotham/
├── eda.py
├── train.py
├── validate.py
├── tune_xgboost.py
├── predict.py
└── src/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── metrics.py
    ├── models.py
    └── utils.py
```

## Feature Engineering

The pipeline derives additional features from the raw input columns:

- Time features: pickup hour, day of week, month, day, weekend flag, rush-hour flag, late-night flag.
- Cyclic time features: sine/cosine encodings for hour and day of week.
- Trip geometry features: dx, dy, absolute dx/dy, Euclidean distance, Manhattan distance, squared distance, log distance, direction angle.
- Spatial features: average route location, pickup/dropoff distance from origin, pickup/dropoff grid zones, route-zone hash.
- Passenger features: clipped passenger count and abnormal passenger-count flag.

## Run EDA

```bash
python gotham/eda.py \
  --train-path data/gotham/Train.csv \
  --target duration \
  --task-type regression
```

EDA outputs are saved under:

```text
gotham/outputs/eda/
```

## Validate Models

Run holdout validation for a candidate model:

```bash
python gotham/validate.py \
  --train-path data/gotham/Train.csv \
  --target duration \
  --task-type regression \
  --model xgboost \
  --max-train-rows 300000
```

## Tune XGBoost

```bash
python gotham/tune_xgboost.py \
  --train-path data/gotham/Train.csv \
  --target duration \
  --task-type regression \
  --max-train-rows 300000
```

The best tuned model configuration was:

```text
XGBoostRegressor
n_estimators = 1200
learning_rate = 0.03
max_depth = 10
subsample = 0.85
colsample_bytree = 0.85
min_child_weight = 5
reg_alpha = 0.1
reg_lambda = 2.0
tree_method = hist
```

Validation performance of the best tuned XGBoost model:

```text
RMSE = 224.36 seconds
MAE  = 140.51 seconds
R²   = 0.8330
```

## Train Final Model

The final model is trained on the full `Train.csv` dataset.

```bash
python gotham/train.py \
  --train-path data/gotham/Train.csv \
  --target duration \
  --task-type regression \
  --sample-frac 0.10 \
  --max-sample-rows 100000 \
  --cv 3 \
  --final-model xgboost
```

Outputs:

```text
gotham/outputs/
├── leaderboard.csv
├── metadata.json
└── model.joblib
```

## Predict Official Test File

When the official test file is released, place it under:

```text
data/gotham/OfficialTest.csv
```

Then run:

```bash
python gotham/predict.py \
  --test-path data/gotham/OfficialTest.csv \
  --prediction-column duration \
  --keep-test-columns \
  --output-path submissions/gotham_predictions.csv
```

## Check Prediction File

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("submissions/gotham_predictions.csv")
print(df.shape)
print(df.head())
print(df["duration"].describe())
print("Negative predictions:", (df["duration"] < 0).sum())
print("Missing predictions:", df["duration"].isna().sum())
PY
```

## Notes

- Model comparison is performed on a subset of the data for speed.
- The final competition model is trained on the full training dataset.
- The prediction script automatically applies the same feature engineering to the official test file.
- Always inspect the output CSV before submitting.