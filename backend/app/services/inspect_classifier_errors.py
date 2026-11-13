
import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset"
)

CSV_PATH = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_prediction_probabilities.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_error_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions():

    with open(CSV_PATH, "r", newline="") as file:
        return list(csv.DictReader(file))


def create_contact_sheet(items, title, output_name):

    if not items:
        print("No samples found for:", title)
        return

    cell_width = 220
    cell_height = 175
    columns = 4
    rows = (len(items) + columns - 1) // columns

    sheet = np.full(
        (rows * cell_height + 50, columns * cell_width, 3),
        255,
        dtype=np.uint8
    )

    cv2.putText(
        sheet,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )

    for index, item in enumerate(items):

        split = item["split"]
        label = int(item["actual_label"])
        filename = item["filename"]
        probability = float(item["probability"])

        folder = "positive" if label == 1 else "negative"

        image_path = (
            DATASET_ROOT
            / split
            / folder
            / filename
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print("Could not read:", image_path)
            continue

        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        x = (index % columns) * cell_width
        y = (index // columns) * cell_height + 50

        sheet[
            y:y + 128,
            x:x + 128
        ] = image

        label_text = (
            f"{split} | {folder}"
        )

        probability_text = (
            f"Probability: {probability:.4f}"
        )

        cv2.putText(
            sheet,
            label_text,
            (x, y + 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 0, 0),
            1
        )

        cv2.putText(
            sheet,
            probability_text,
            (x, y + 163),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 0, 180),
            1
        )

    output_path = OUTPUT_DIR / output_name

    cv2.imwrite(str(output_path), sheet)

    print("Saved:", output_path)


def main():

    predictions = load_predictions()

    test_negative = [
        item for item in predictions
        if item["split"] == "test"
        and int(item["actual_label"]) == 0
    ]

    val_negative = [
        item for item in predictions
        if item["split"] == "val"
        and int(item["actual_label"]) == 0
    ]

    test_positive = [
        item for item in predictions
        if item["split"] == "test"
        and int(item["actual_label"]) == 1
    ]

    # Highest probability negatives = likely false positives
    test_negative.sort(
        key=lambda item: float(item["probability"]),
        reverse=True
    )

    val_negative.sort(
        key=lambda item: float(item["probability"]),
        reverse=True
    )

    # Lowest probability positives = likely false negatives
    test_positive.sort(
        key=lambda item: float(item["probability"])
    )

    create_contact_sheet(
        test_negative[:20],
        "TEST NEGATIVES - HIGHEST PREDICTIONS",
        "test_false_positive_candidates.png"
    )

    create_contact_sheet(
        val_negative[:20],
        "VALIDATION NEGATIVES - HIGHEST PREDICTIONS",
        "validation_hard_negatives.png"
    )

    create_contact_sheet(
        test_positive[:20],
        "TEST POSITIVES - LOWEST PREDICTIONS",
        "test_hard_positives.png"
    )


if __name__ == "__main__":
    main()
