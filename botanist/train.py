import argparse
import runpy


TRAINING_SCRIPTS = {
    "basic": "CNN.py",
    "efficientnet": "train_efficientnet.py",
    "convnext": "train_convnext.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Botanist classification model.")
    parser.add_argument(
        "--model",
        choices=sorted(TRAINING_SCRIPTS),
        default="basic",
        help="Model to train.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runpy.run_path(TRAINING_SCRIPTS[args.model], run_name="__main__")


if __name__ == "__main__":
    main()

