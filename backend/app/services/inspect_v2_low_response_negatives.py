
import csv
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset_v2"
)

CSV_PATH = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
    / "test_predictions_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "classifier_v2_error_inspection"
    / "contact_sheets"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, "r", newline="") as file:
    rows = list(csv.DictReader(file))

# Select test negatives with low vessel response.
low_negatives = [
    row for row in rows
    if int(row["actual_label"]) == 0
    and row["negative_type"] == "low_vessel_response"
]

# Sort by predicted probability.
low_negatives.sort(
    key=lambda row: float(row["predicted_probability"])
)

# Select 36 evenly distributed examples.
indices = np.linspace(
    0,
    len(low_negatives) - 1,
    min(36, len(low_negatives)),
    dtype=int
)

selected = [low_negatives[i] for i in indices]

TILE_SIZE = 160
LABEL_HEIGHT = 45
GRID_COLUMNS = 6
GRID_ROWS = int(np.ceil(len(selected) / GRID_COLUMNS))

sheet = np.full(
    (
        GRID_ROWS * (TILE_SIZE + LABEL_HEIGHT),
        GRID_COLUMNS * TILE_SIZE,
        3
    ),
    255,
    dtype=np.uint8
)

for index, row in enumerate(selected):

    filename = row["filename"]

    image_path = (
        DATASET_DIR
        / "test"
        / "negative"
        / filename
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print("Could not read:", image_path)
        continue

    image = cv2.resize(image, (TILE_SIZE, TILE_SIZE))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    probability = float(row["predicted_probability"])

    x = (index % GRID_COLUMNS) * TILE_SIZE
    y = (index // GRID_COLUMNS) * (TILE_SIZE + LABEL_HEIGHT)

    sheet[
        y:y + TILE_SIZE,
        x:x + TILE_SIZE
    ] = image

    cv2.putText(
        sheet,
        filename[:22],
        (x + 3, y + TILE_SIZE + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1
    )

    cv2.putText(
        sheet,
        f"Probability: {probability:.4f}",
        (x + 3, y + TILE_SIZE + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 180),
        1
    )

output_path = (
    OUTPUT_DIR
    / "v2_test_low_response_negatives.png"
)

cv2.imwrite(str(output_path), sheet)

print("Selected patches:", len(selected))
print("Total low-response negatives:", len(low_negatives))
print("Saved contact sheet:", output_path)
