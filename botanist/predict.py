import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    FILE_COL,
    LABEL_COL,
    TestImageDataset,
    build_basic_transform,
    build_imagenet_style_transform,
    collect_image_paths,
    find_image_path,
)
from src.models import build_model, load_checkpoint
from src.utils import get_device


DEFAULT_WEIGHTS = {
    "basic": "botanist_basic_cnn_best.pth",
    "efficientnet": "botanist_efficientnet_b0_best.pth",
    "convnext": "botanist_convnext_tiny_best.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Botanist model predictions.")
    parser.add_argument("--model", choices=["basic", "efficientnet", "convnext"], default="basic")
    parser.add_argument("--input", type=str, default=None, help="Single image or folder of images.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV with FileName column.")
    parser.add_argument("--image-dir", type=str, default=".", help="Image folder used with --csv.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--output", type=str, default=None, help="Output prediction CSV path.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


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


def build_transform(model_name):
    if model_name == "basic":
        return build_basic_transform(image_size=128)
    return build_imagenet_style_transform(image_size=224)


def predict(model, image_paths, transform, device, top_k, batch_size):
    dataset = TestImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
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


def main():
    args = parse_args()
    if args.input is None and args.csv is None:
        raise ValueError("Use either --input for images/folder or --csv for a test CSV.")

    weights_path = args.weights or DEFAULT_WEIGHTS[args.model]
    output_path = args.output or f"{args.model}_predictions.csv"
    top_k = max(1, min(args.top_k, 38))

    device = get_device()
    model = build_model(args.model)
    model = load_checkpoint(model, weights_path, device)

    source_df = None
    if args.csv is not None:
        source_df, image_paths = load_csv_inputs(args.csv, args.image_dir)
    else:
        image_paths = collect_image_paths(args.input)

    predictions = predict(model, image_paths, build_transform(args.model), device, top_k, args.batch_size)

    if source_df is not None:
        output_df = source_df.copy()
        for column in predictions.columns:
            output_df[column] = predictions[column]
        if LABEL_COL in output_df.columns:
            output_df[LABEL_COL] = output_df[LABEL_COL].astype(int)
            accuracy = (output_df[LABEL_COL] == output_df["predicted_label"]).mean()
            print(f"Test accuracy: {accuracy:.4f}")
    else:
        output_df = predictions

    output_df.to_csv(output_path, index=False)
    print(f"Device: {device}")
    print(f"Images tested: {len(output_df)}")
    print(f"Predictions saved to: {output_path}")
    print(output_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

