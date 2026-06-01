import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.data import find_image_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate simple EDA figures for the Botanist dataset.")
    parser.add_argument("--csv", default="Botanist_Training_Set.csv")
    parser.add_argument("--image-dir", default="TrainFiles")
    parser.add_argument("--output-dir", default="eda_outputs")
    return parser.parse_args()


def save_label_distribution(df, output_dir):
    counts = df["Label"].value_counts().sort_index()
    plt.figure(figsize=(12, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "label_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def save_sample_grid(df, image_dir, output_dir):
    labels = list(range(1, 39))
    fig, axes = plt.subplots(4, 10, figsize=(12, 5.2))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    for idx, label in enumerate(labels):
        rows = df[df["Label"] == label]
        if rows.empty:
            continue
        image_path = find_image_path(image_dir, rows.iloc[0]["FileName"])
        if image_path is None:
            continue
        image = Image.open(image_path).convert("RGB")
        axes[idx].imshow(image)
        axes[idx].set_title(str(label), fontsize=8)
        axes[idx].axis("off")

    fig.suptitle("Example Leaf Images by Class Label", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = output_dir / "sample_images_by_class.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["FileName"] = df["FileName"].astype(str)
    df["Label"] = df["Label"].astype(int)

    counts = df["Label"].value_counts().sort_index()
    print("Rows:", len(df))
    print("Classes:", len(counts))
    print("Label range:", df["Label"].min(), "-", df["Label"].max())
    print("Min class count:", counts.min())
    print("Max class count:", counts.max())
    print("Imbalance ratio:", round(counts.max() / counts.min(), 2))

    print("Saved:", save_label_distribution(df, output_dir))
    print("Saved:", save_sample_grid(df, args.image_dir, output_dir))


if __name__ == "__main__":
    main()

