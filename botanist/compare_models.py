import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_tiny, efficientnet_b0

from CNN import BasicCNN


CSV_PATH = "Botanist_Training_Set.csv"
IMAGE_DIR = "TrainFiles"
FILE_COL = "FileName"
LABEL_COL = "Label"
NUM_CLASSES = 38
BATCH_SIZE = 32
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_CSV = "model_comparison_results.csv"
OUTPUT_CHART = "model_comparison_chart.png"
OUTPUT_TABLE = "model_comparison_table.png"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_path(image_dir, file_name):
    file_name = str(file_name)
    possible_names = [
        file_name,
        file_name + ".jpg",
        file_name + ".jpeg",
        file_name + ".png",
        file_name + ".JPG",
        file_name + ".JPEG",
        file_name + ".PNG",
    ]

    for name in possible_names:
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            return path

    return None


class BotanistDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        file_name = row[FILE_COL]
        label = int(row["class_index"])

        image_path = find_image_path(self.image_dir, file_name)
        if image_path is None:
            raise FileNotFoundError(f"Image not found: {file_name}")

        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def load_checkpoint(model, weight_path):
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


def build_basic_cnn():
    model = BasicCNN(num_classes=NUM_CLASSES)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    criterion = nn.CrossEntropyLoss()
    return model, transform, criterion


def build_efficientnet():
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return model, transform, criterion


def build_convnext():
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return model, transform, criterion


def evaluate_model(model, data_loader, criterion):
    total_loss = 0.0
    all_preds = []
    all_labels = []

    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    elapsed_seconds = time.perf_counter() - start
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "validation_loss": avg_loss,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "inference_seconds": elapsed_seconds,
        "images_per_second": len(data_loader.dataset) / elapsed_seconds,
    }


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def save_comparison_chart(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(results_df))
    width = 0.25
    accuracy_error = np.maximum(1.0 - results_df["accuracy"].to_numpy(), 1e-8)
    macro_f1_error = np.maximum(1.0 - results_df["macro_f1"].to_numpy(), 1e-8)
    weighted_f1_error = np.maximum(1.0 - results_df["weighted_f1"].to_numpy(), 1e-8)
    axes[0].bar(x - width, accuracy_error, width, label="1 - Accuracy")
    axes[0].bar(x, macro_f1_error, width, label="1 - Macro F1")
    axes[0].bar(x + width, weighted_f1_error, width, label="1 - Weighted F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results_df["model"], rotation=15, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_title("Validation Error Metrics")
    axes[0].set_ylabel("Error (log scale)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, results_df["parameters_m"], width, label="Parameters (M)")
    axes[1].bar(x + width / 2, results_df["checkpoint_mb"], width, label="Checkpoint (MB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df["model"], rotation=15, ha="right")
    axes[1].set_title("Model Size")
    axes[1].set_ylabel("Size")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_CHART, dpi=300)
    plt.close(fig)


def save_comparison_table(results_df):
    display_df = results_df[
        [
            "model",
            "accuracy",
            "macro_f1",
            "validation_loss",
            "parameters_m",
            "checkpoint_mb",
            "images_per_second",
        ]
    ].copy()

    display_df.columns = [
        "Model",
        "Accuracy",
        "Macro F1",
        "Val Loss",
        "Params (M)",
        "Checkpoint (MB)",
        "Images/Sec",
    ]

    for column in ["Accuracy", "Macro F1", "Val Loss", "Params (M)", "Checkpoint (MB)", "Images/Sec"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_TABLE, dpi=300)
    plt.close(fig)


def main():
    set_seed(RANDOM_SEED)
    print("Device:", DEVICE)

    df = pd.read_csv(CSV_PATH)
    df[FILE_COL] = df[FILE_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df["class_index"] = df[LABEL_COL] - 1

    _, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["class_index"],
    )

    model_specs = [
        {
            "model": "Basic CNN",
            "builder": build_basic_cnn,
            "weight_path": "botanist_basic_cnn_best.pth",
            "training_epochs": 15,
            "pretrained": "No",
        },
        {
            "model": "EfficientNet-B0",
            "builder": build_efficientnet,
            "weight_path": "botanist_efficientnet_b0_best.pth",
            "training_epochs": 10,
            "pretrained": "No",
        },
        {
            "model": "ConvNeXt-Tiny",
            "builder": build_convnext,
            "weight_path": "botanist_convnext_tiny_best.pth",
            "training_epochs": 12,
            "pretrained": "No",
        },
    ]

    results = []
    for spec in model_specs:
        print(f"\nEvaluating {spec['model']}...")
        model, transform, criterion = spec["builder"]()
        model = load_checkpoint(model, spec["weight_path"])

        dataset = BotanistDataset(val_df, IMAGE_DIR, transform)
        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        metrics = evaluate_model(model, data_loader, criterion)
        parameter_count = count_parameters(model)
        checkpoint_mb = os.path.getsize(spec["weight_path"]) / (1024 * 1024)

        row = {
            "model": spec["model"],
            "training_epochs": spec["training_epochs"],
            "pretrained": spec["pretrained"],
            "parameters": parameter_count,
            "parameters_m": parameter_count / 1_000_000,
            "checkpoint_mb": checkpoint_mb,
        }
        row.update(metrics)
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    save_comparison_chart(results_df)
    save_comparison_table(results_df)

    print("\nModel comparison results:")
    print(results_df.to_string(index=False))
    print("\nSaved files:")
    print(OUTPUT_CSV)
    print(OUTPUT_CHART)
    print(OUTPUT_TABLE)


if __name__ == "__main__":
    main()
