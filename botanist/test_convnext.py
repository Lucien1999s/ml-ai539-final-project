import argparse
import os

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_tiny


NUM_CLASSES = 38
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEFAULT_WEIGHTS = "botanist_convnext_tiny_best.pth"
FILE_COL = "FileName"
LABEL_COL = "Label"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_model(weights_path):
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)

    return model.to(DEVICE).eval()


def load_csv_inputs(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    if FILE_COL not in df.columns:
        raise ValueError(f"CSV must contain a '{FILE_COL}' column.")

    image_paths = []
    for file_name in df[FILE_COL].astype(str):
        image_path = find_image_path(image_dir, file_name)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for CSV row: {file_name}")
        image_paths.append(image_path)

    return df, image_paths


def predict(model, image_paths, top_k):
    dataset = TestImageDataset(image_paths, build_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(DEVICE)
            probabilities = torch.softmax(model(images), dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

            for path, probs, indices in zip(paths, top_probs.cpu(), top_indices.cpu()):
                row = {
                    "image_path": path,
                    "predicted_label": int(indices[0].item()) + 1,
                    "confidence": float(probs[0].item()),
                }
                for rank, (prob, index) in enumerate(zip(probs, indices), start=1):
                    row[f"top{rank}_label"] = int(index.item()) + 1
                    row[f"top{rank}_confidence"] = float(prob.item())
                rows.append(row)

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run test predictions with the best ConvNeXt-Tiny Botanist model."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="A single image or a folder of test images.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV with FileName and optionally Label columns.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=".",
        help="Image folder used with --csv.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="Path to the trained ConvNeXt-Tiny checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="convnext_test_predictions.csv",
        help="Where to save predictions.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to save.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input is None and args.csv is None:
        raise ValueError("Use either --input for images/folder or --csv for a test CSV.")

    top_k = max(1, min(args.top_k, NUM_CLASSES))
    model = build_model(args.weights)

    source_df = None
    if args.csv is not None:
        source_df, image_paths = load_csv_inputs(args.csv, args.image_dir)
    else:
        image_paths = collect_image_paths(args.input)

    predictions = predict(model, image_paths, top_k)

    if source_df is not None:
        output_df = source_df.copy()
        output_df["image_path"] = predictions["image_path"]
        output_df["predicted_label"] = predictions["predicted_label"]
        output_df["confidence"] = predictions["confidence"]

        if LABEL_COL in output_df.columns:
            output_df[LABEL_COL] = output_df[LABEL_COL].astype(int)
            accuracy = (output_df[LABEL_COL] == output_df["predicted_label"]).mean()
            print(f"Test accuracy: {accuracy:.4f}")
    else:
        output_df = predictions

    output_df.to_csv(args.output, index=False)

    print(f"Device: {DEVICE}")
    print(f"Images tested: {len(output_df)}")
    print(f"Predictions saved to: {args.output}")
    print(output_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
