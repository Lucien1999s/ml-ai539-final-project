# Botanist Plant Image Classification

This repository contains the code for the AI 539 Botanist plant image classification project. The task is to classify plant leaf images into 38 categories.

No external training images or pretrained weights are used. All models are trained only on the provided Botanist dataset.

## Overview

Three models are trained and compared:

- Basic CNN: a small convolutional neural network trained from scratch.
- EfficientNet-B0: the EfficientNet-B0 architecture trained from scratch.
- ConvNeXt-Tiny: the ConvNeXt-Tiny architecture trained from scratch.

In the final from-scratch experiment, Basic CNN achieved the best accuracy-efficiency tradeoff.

## Repository Structure

```text
.
|-- src/
|   |-- __init__.py
|   |-- data.py                  # Dataset loading, image lookup, transforms
|   |-- models.py                # Basic CNN, EfficientNet-B0, ConvNeXt-Tiny builders
|   |-- metrics.py               # Classification metric helpers
|   `-- utils.py                 # Device and random seed utilities
|-- train.py                     # Unified training entry point
|-- predict.py                   # Unified prediction entry point
|-- validate.py                  # Model comparison entry point
|-- eda.py                       # EDA figure generation
|-- CNN.py                       # Original Basic CNN training script
|-- train_efficientnet.py        # Original EfficientNet-B0 training script
|-- train_convnext.py            # Original ConvNeXt-Tiny training script
|-- compare_models.py            # Compare all trained models
|-- test_basic_cnn.py            # Basic CNN prediction script
|-- test_convnext.py             # ConvNeXt-Tiny prediction script
|-- main.py                      # Dataset checking helper
|-- requirements.txt
`-- README.md
```

Large files such as training images, CSV files, model checkpoints, PDFs, PNG figures, zip files, and generated reports are excluded by `.gitignore`.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

The dataset should be arranged as:

```text
Botanist_Training_Set.csv
TrainFiles/
```

The CSV file should contain:

- `FileName`: image file ID or image file name
- `Label`: class label from 1 to 38

## Dataset Check

```powershell
python .\main.py
```

## EDA

```powershell
python .\eda.py
```

This generates class distribution and sample-image figures in `eda_outputs/`.

## Training

Use the unified entry point:

```powershell
python .\train.py --model basic
python .\train.py --model efficientnet
python .\train.py --model convnext
```

These commands call the original training scripts and save the best checkpoint for each model.

## Model Comparison

After checkpoints are available, run:

```powershell
python .\validate.py
```

This runs `compare_models.py` and generates:

- `model_comparison_results.csv`
- `model_comparison_chart.png`
- `model_comparison_table.png`

## Prediction

Use the unified prediction script. Basic CNN is the recommended final model from the from-scratch experiment.

Single image:

```powershell
python .\predict.py --model basic --input .\example.jpg --output .\basic_predictions.csv
```

Folder of images:

```powershell
python .\predict.py --model basic --input .\TestFiles --output .\basic_predictions.csv
```

Teacher-provided CSV:

```powershell
python .\predict.py --model basic --csv .\Teacher_Test.csv --image-dir .\TestFiles --output .\basic_predictions.csv
```

The output CSV includes predicted label, confidence score, and top-k predictions. If the input CSV contains a `Label` column, the script also prints test accuracy.

## Notes

- Labels in the CSV are expected to be 1-based labels from 1 to 38.
- Training converts labels to 0-based class indices for PyTorch.
- Prediction converts model outputs back to 1-based labels.
- Basic CNN is the final recommended model because it achieved the highest validation accuracy with the smallest checkpoint and fastest inference.
