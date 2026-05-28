# AI 539 Final Project

This repository contains the codebase for the AI 539 final project.

The project consists of two prediction tasks: Gotham Cabs and Botanist. Each task is organized in its own folder so that the two pipelines can be developed and tested independently.

At the current stage, the Gotham Cabs pipeline is implemented and ready for training, validation, tuning, and test prediction. The Botanist folder is reserved for the image classification task and will be completed separately.

## Repository Structure

```text
.
├── gotham/
│   ├── eda.py
│   ├── train.py
│   ├── validate.py
│   ├── tune_xgboost.py
│   ├── predict.py
│   └── src/
├── botanist/
│   ├── train.py
│   ├── predict.py
│   ├── notebooks/
│   ├── outputs/
│   └── src/
├── submissions/
├── README.md
└── requirements.txt
```

## Gotham Cabs

Gotham Cabs is a taxi trip duration prediction task.

Given pickup time, passenger count, pickup location, and dropoff location, the goal is to predict the trip duration. The target variable is:

```text
duration
```

The Gotham pipeline includes:

- Exploratory data analysis
- Feature engineering for time, distance, route geometry, spatial zones, and passenger count
- Baseline and tree-based model comparison
- Holdout validation
- XGBoost hyperparameter tuning
- Full-data final model training
- Test-file prediction pipeline

The final selected Gotham model is a tuned XGBoost regressor trained on the full Gotham training dataset.

Current validation result for the selected Gotham model:

```text
RMSE = 224.36 seconds
MAE  = 140.51 seconds
R²   = 0.8330
```

For detailed Gotham instructions, see:

```text
gotham/README.md
```

## Botanist

The Botanist folder is reserved for the image classification task.

Its implementation and final modeling details will be added separately.

## Local Data Layout

Large data files are not tracked in Git. Place local datasets under:

```text
data/
├── gotham/
│   ├── Train.csv
│   ├── TestFileTemplate.csv
│   └── OfficialTest.csv
└── botanist/
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Gotham Final Model

Train the final Gotham model:

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

Predict the official Gotham test file:

```bash
python gotham/predict.py \
  --test-path data/gotham/OfficialTest.csv \
  --prediction-column duration \
  --keep-test-columns \
  --output-path submissions/gotham_predictions.csv
```

## Notes

Large data files, generated outputs, model checkpoints, and prediction files are not tracked in Git.

The final report should describe the modeling decisions, feature engineering process, validation results, and final model selection for each task.