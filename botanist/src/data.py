import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


FILE_COL = "FileName"
LABEL_COL = "Label"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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


def collect_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_paths = []
    for root, _, files in os.walk(input_path):
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, file_name))

    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {input_path}")

    return sorted(image_paths)


def load_training_dataframe(csv_path, num_classes=38):
    df = pd.read_csv(csv_path)
    df[FILE_COL] = df[FILE_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df["class_index"] = df[LABEL_COL] - 1

    if df["class_index"].min() < 0 or df["class_index"].max() >= num_classes:
        raise ValueError(f"Labels must be from 1 to {num_classes}.")

    return df


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


class TestImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_path


def build_basic_transform(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def build_imagenet_style_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

