import os
import copy
import random
import pandas as pd
import numpy as np
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


# =========================
# 1. Settings
# =========================

CSV_PATH = "Botanist_Training_Set.csv"
IMAGE_DIR = "TrainFiles"

FILE_COL = "FileName"
LABEL_COL = "Label"

NUM_CLASSES = 38
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

MODEL_SAVE_PATH = "botanist_basic_cnn_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. Helper functions
# =========================

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


# =========================
# 3. Basic CNN model
# =========================

class BasicCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(BasicCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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


# =========================
# 4. Main training
# =========================

def main():
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

    print("\nLabel min:", df[LABEL_COL].min())
    print("Label max:", df[LABEL_COL].max())

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
        print("First 20 missing files:")
        print(missing[:20])
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

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = BotanistDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset = BotanistDataset(val_df, IMAGE_DIR, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = BasicCNN(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("\nStart training Basic CNN...\n")

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

        scheduler.step()

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
                "model_name": "basic_cnn"
            }, MODEL_SAVE_PATH)

            print("Best model saved.")

        print("-" * 50)

    print("\nTraining finished.")
    print("Best validation accuracy:", best_val_acc)

    model.load_state_dict(best_weights)

    val_loss, val_acc, val_labels, val_preds = validate_one_epoch(
        model,
        val_loader,
        criterion
    )

    val_labels_original = [x + 1 for x in val_labels]
    val_preds_original = [x + 1 for x in val_preds]

    print("\nFinal validation accuracy:", val_acc)

    print("\nClassification report:")
    print(classification_report(val_labels_original, val_preds_original))

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Basic CNN Training and Validation Loss")
    plt.legend()
    plt.savefig("basic_cnn_loss_curve.png", dpi=300)
    plt.show()

    train_error_rates = np.maximum(1.0 - np.array(train_accs), 1e-8)
    val_error_rates = np.maximum(1.0 - np.array(val_accs), 1e-8)

    plt.figure()
    plt.plot(train_error_rates, label="Train Error Rate")
    plt.plot(val_error_rates, label="Validation Error Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate (1 - Accuracy)")
    plt.yscale("log")
    plt.title("Basic CNN Training and Validation Error Rate")
    plt.legend()
    plt.savefig("basic_cnn_accuracy_curve.png", dpi=300)
    plt.show()

    cm = confusion_matrix(val_labels_original, val_preds_original)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.title("Basic CNN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.savefig("basic_cnn_confusion_matrix.png", dpi=300)
    plt.show()

    print("\nSaved files:")
    print(MODEL_SAVE_PATH)
    print("basic_cnn_loss_curve.png")
    print("basic_cnn_accuracy_curve.png")
    print("basic_cnn_confusion_matrix.png")


if __name__ == "__main__":
    main()
