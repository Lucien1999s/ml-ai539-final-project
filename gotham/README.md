# Gotham Cabs

This folder contains the modeling pipeline for the Gotham Cabs prediction task in the AI 539 final project.

## Goal

The goal is to train machine learning models using `Train.csv`, compare multiple methods using validation or cross-validation, select a final model, retrain it on the full training data, and generate predictions for the official test file.

## Expected Local Data Layout

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
├── predict.py
└── src/
    ├── __init__.py
    ├── data.py
    ├── metrics.py
    ├── models.py
    └── utils.py
```

## 1. Run EDA

```bash
python gotham/eda.py \
  --train-path data/gotham/Train.csv \
  --target TARGET_COLUMN_NAME
```

If the target is the last column, `--target` can be omitted, but explicitly setting it is safer.

EDA outputs are saved under:

```text
gotham/outputs/eda/
```

## 2. Train and Compare Models

```bash
python gotham/train.py \
  --train-path data/gotham/Train.csv \
  --target TARGET_COLUMN_NAME \
  --sample-frac 0.15 \
  --max-sample-rows 150000 \
  --cv 5
```

Outputs:

```text
gotham/outputs/
├── leaderboard.csv
├── metadata.json
└── model.joblib
```

The leaderboard is based on a subset of the data for speed. The final saved model is trained on the full training data.

## 3. Predict Test File

Basic prediction:

```bash
python gotham/predict.py \
  --test-path data/gotham/TestFileTemplate.csv \
  --output-path submissions/gotham_predictions.csv
```

If the official submission format has an ID column:

```bash
python gotham/predict.py \
  --test-path data/gotham/OfficialTest.csv \
  --id-column ID_COLUMN_NAME \
  --output-path submissions/gotham_predictions.csv
```

If the instructor gives a template file with a column to fill:

```bash
python gotham/predict.py \
  --test-path data/gotham/OfficialTest.csv \
  --template-path data/gotham/TestFileTemplate.csv \
  --prediction-column TARGET_COLUMN_NAME \
  --output-path submissions/gotham_predictions.csv
```

## Notes

- Model comparison can use a fraction of the data for speed.
- The final competition model is trained on the entire training set.
- The prediction script automatically aligns official test columns to the training feature columns.
- Always inspect the output CSV before submitting.