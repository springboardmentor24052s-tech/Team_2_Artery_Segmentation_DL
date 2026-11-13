
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
)

CSV_PATH = DIAGNOSTIC_DIR / "test_predictions_v2.csv"

OUTPUT_DIR = DIAGNOSTIC_DIR / "contact_sheets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# --------------------------------------------------
# Load prediction CSV
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

with open(CSV_PATH, "r", newline="") as file:
    reader = csv.DictReader(file)
    predictions = list(reader)


# --------------------------------------------------
# Select errors
# --------------------------------------------------

false_positives = [
    row for row in predictions
    if row["result"] == "FALSE_POSITIVE"
]

false_negatives = [
    row for row in predictions
    if row["result"] == "FALSE_NEGATIVE"
]

# Highest probability false positives
false_positives.sort(
    key=lambda row: float(row["predicted_probability"]),
    reverse=True
)

selected_fp = false_positives[:30]

# Lowest probability false negatives
false_negatives.sort(
    key=lambda row: float(row["predicted_probability"])
)

selected_fn = false_negatives[:20]


# --------------------------------------------------
# Contact sheet generation
# --------------------------------------------------

def create_contact_sheet(samples, class_folder, title, output_name):

    if not samples:
        print(f"No samples available for {title}")
        return

    columns = 5
    rows = int(np.ceil(len(samples) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, rows * 3)
    )

    axes = np.array(axes).reshape(-1)

    for index, sample in enumerate(samples):

        filename = sample["filename"]

        image_path = (
            DATASET_ROOT
            / "test"
            / class_folder
            / filename
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print("Unable to read:", image_path)
            axes[index].axis("off")
            continue

        probability = float(
            sample["predicted_probability"]
        )

        axes[index].imshow(
            image,
            cmap="gray"
        )

        axes[index].set_title(
            f"{filename}\nP={probability:.4f}",
            fontsize=8
        )

        axes[index].axis("off")

    for index in range(len(samples), len(axes)):
        axes[index].axis("off")

    fig.suptitle(title, fontsize=15)

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_name

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# --------------------------------------------------
# Generate false-positive contact sheet
# --------------------------------------------------

create_contact_sheet(
    selected_fp,
    "negative",
    "V2 TEST - TOP 30 FALSE POSITIVE NEGATIVE PATCHES",
    "v2_test_false_positives.png"
)


# --------------------------------------------------
# Generate false-negative contact sheet
# --------------------------------------------------

create_contact_sheet(
    selected_fn,
    "positive",
    "V2 TEST - FALSE NEGATIVE POSITIVE PATCHES",
    "v2_test_false_negatives.png"
)


# --------------------------------------------------
# Generate a random false-positive sample
# --------------------------------------------------

random_fp = random.sample(
    false_positives,
    min(30, len(false_positives))
)

create_contact_sheet(
    random_fp,
    "negative",
    "V2 TEST - RANDOM FALSE POSITIVE PATCHES",
    "v2_test_random_false_positives.png"
)


print("\nV2 error contact sheets generated.")
