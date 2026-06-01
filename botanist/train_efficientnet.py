import os
import copy
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from torchvision import transforms
from torchvision.models import efficientnet_b0


CSV_PATH = "Botanist_Training_Set.csv"
IMAGE_DIR = "TrainFiles"

FILE_COL = "FileName"
LABEL_COL = "Label"

NUM_CLASSES = 38
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

USE_PRETRAINED = False

MODEL_SAVE_PATH = "botanist_efficientnet_b0_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, dataframe, image_dir, transform=None):
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

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def validate_one_epoch(model, val_loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or test EfficientNet-B0 on the Botanist dataset."
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to a trained model weight file. If set, skip training and run evaluation only."
    )
    return parser.parse_args()


def build_model(use_pretrained=False):
    if use_pretrained:
        raise ValueError(
            "External pretrained weights are disabled for this project. "
            "Use only the provided Botanist training dataset."
        )

    print("\nUsing EfficientNet-B0 architecture from scratch.")
    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model.to(DEVICE)


def load_model_weights(model, weight_path):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weight file not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print(f"\nModel weights loaded from: {weight_path}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("EfficientNet-B0 Training and Validation Loss")
    plt.legend()
    plt.savefig("efficientnet_loss_curve.png", dpi=300)
    plt.close()

    train_error_rates = np.maximum(1.0 - np.array(train_accs), 1e-8)
    val_error_rates = np.maximum(1.0 - np.array(val_accs), 1e-8)

    plt.figure()
    plt.plot(train_error_rates, label="Train Error Rate")
    plt.plot(val_error_rates, label="Validation Error Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate (1 - Accuracy)")
    plt.yscale("log")
    plt.title("EfficientNet-B0 Training and Validation Error Rate")
    plt.legend()
    plt.savefig("efficientnet_accuracy_curve.png", dpi=300)
    plt.close()


def evaluate_and_plot(model, val_loader, criterion):
    val_loss, val_acc, val_labels, val_preds = validate_one_epoch(
        model,
        val_loader,
        criterion
    )

    val_labels_original = [x + 1 for x in val_labels]
    val_preds_original = [x + 1 for x in val_preds]

    print("\nFinal validation loss:", val_loss)
    print("Final validation accuracy:", val_acc)

    print("\nClassification report:")
    print(classification_report(val_labels_original, val_preds_original))

    cm = confusion_matrix(val_labels_original, val_preds_original)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.title("EfficientNet-B0 Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.savefig("efficientnet_confusion_matrix.png", dpi=300)
    plt.close()

    return val_loss, val_acc


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    df = pd.read_csv(CSV_PATH)

    print("\nCSV loaded.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    df[FILE_COL] = df[FILE_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    df["class_index"] = df[LABEL_COL] - 1

    if df["class_index"].min() < 0 or df["class_index"].max() >= NUM_CLASSES:
        raise ValueError("Labels must be from 1 to 38.")

    print("\nChecking image files...")

    missing = []

    for file_name in df[FILE_COL]:
        if find_image_path(IMAGE_DIR, file_name) is None:
            missing.append(file_name)

    print("Missing image count:", len(missing))

    if len(missing) > 0:
        print("First 20 missing:", missing[:20])
        raise FileNotFoundError("Some images cannot be found.")

    print("All images found.")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["class_index"]
    )

    print("\nTrain size:", len(train_df))
    print("Validation size:", len(val_df))

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = BotanistDataset(val_df, IMAGE_DIR, val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.test is not None:
        model = build_model(use_pretrained=False)
        load_model_weights(model, args.test)
        evaluate_and_plot(model, val_loader, criterion)

        print("\nSaved files:")
        print("efficientnet_confusion_matrix.png")
        return

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = BotanistDataset(train_df, IMAGE_DIR, train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = build_model(USE_PRETRAINED)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("\nStart training EfficientNet-B0...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, val_labels, val_preds = validate_one_epoch(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.8f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_weights,
                "num_classes": NUM_CLASSES,
                "image_size": IMAGE_SIZE,
                "file_col": FILE_COL,
                "label_col": LABEL_COL,
                "use_pretrained": USE_PRETRAINED,
                "model_name": "efficientnet_b0"
            }, MODEL_SAVE_PATH)

            print("Best model saved.")

        print("-" * 50)

    print("\nTraining finished.")
    print("Best validation accuracy:", best_val_acc)

    model.load_state_dict(best_weights)

    evaluate_and_plot(model, val_loader, criterion)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    print("\nSaved files:")
    print(MODEL_SAVE_PATH)
    print("efficientnet_loss_curve.png")
    print("efficientnet_accuracy_curve.png")
    print("efficientnet_confusion_matrix.png")


if __name__ == "__main__":
    main()
