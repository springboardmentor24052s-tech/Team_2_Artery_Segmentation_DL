
import csv
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v3"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v3"
    / "contact_sheets"
)

SAMPLES_PER_SHEET = 36
RANDOM_SEED = 42


def create_contact_sheet(split, label):

    folder = DATASET_ROOT / split / label
    csv_path = DATASET_ROOT / f"{split}_labels.csv"

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    rows = [
        row for row in rows
        if row["label"] == ("1" if label == "positive" else "0")
    ]

    random.seed(RANDOM_SEED)
    selected = random.sample(
        rows,
        min(SAMPLES_PER_SHEET, len(rows))
    )

    fig, axes = plt.subplots(6, 6, figsize=(14, 14))
    axes = axes.flatten()

    for ax, row in zip(axes, selected):

        image_path = folder / row["filename"]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            ax.set_title("Unreadable")
            ax.axis("off")
            continue

        ax.imshow(image, cmap="gray")

        if label == "negative":
            score = row.get("vessel_response_score", "")
            ax.set_title(f"Weak neg | Score {float(score):.1f}", fontsize=7)
        else:
            ax.set_title("Positive", fontsize=8)

        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(
        f"V3 {split.upper()} - {label.upper()} patches",
        fontsize=14
    )

    plt.tight_layout()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_ROOT
        / f"v3_{split}_{label}.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Saved:", output_path)
    print("Patches displayed:", len(selected))


def main():

    for split in ["train", "val", "test"]:
        for label in ["positive", "negative"]:
            create_contact_sheet(split, label)


if __name__ == "__main__":
    main()
