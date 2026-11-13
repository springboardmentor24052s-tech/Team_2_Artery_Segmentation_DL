
import json
import cv2
import numpy as np
from pathlib import Path

IMAGE_ID = "16c135650bb24194a3abfd836c03237c"

BRANCH_PATH = Path(
    "backend/results/branch_diameter_test.json"
)

GT_MASK_PATH = Path(
    "backend/results/arcade_masks/syntax/test/1_mask.png"
)

PRED_MASK_PATH = (
    Path("backend/results/api_predictions")
    / f"{IMAGE_ID}_cleaned_mask.png"
)

# Load branch data.
with open(BRANCH_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

# Load masks.
gt_mask = cv2.imread(
    str(GT_MASK_PATH), cv2.IMREAD_GRAYSCALE
)

pred_mask = cv2.imread(
    str(PRED_MASK_PATH), cv2.IMREAD_GRAYSCALE
)

if gt_mask is None:
    raise FileNotFoundError(GT_MASK_PATH)

if pred_mask is None:
    raise FileNotFoundError(PRED_MASK_PATH)

if gt_mask.shape != pred_mask.shape:
    raise ValueError(
        "Ground-truth and predicted masks have different dimensions."
    )

gt_binary = gt_mask > 0
pred_binary = pred_mask > 0

all_coordinates = []

print("\n========== BRANCH GROUND-TRUTH VALIDATION ==========")

for branch in data["branches"]:

    # Support both branch JSON structures.
    if "coordinates" in branch:
        coordinates = branch["coordinates"]

    elif "diameter_profile" in branch:
        coordinates = [
            [point["y"], point["x"]]
            for point in branch["diameter_profile"]
        ]

    else:
        raise KeyError(
            f"Coordinates missing for branch {branch.get('branch_id')}"
        )

    valid = [
        (int(y), int(x))
        for y, x in coordinates
        if 0 <= int(y) < gt_mask.shape[0]
        and 0 <= int(x) < gt_mask.shape[1]
    ]

    gt_hits = sum(
        gt_binary[y, x] for y, x in valid
    )

    pred_hits = sum(
        pred_binary[y, x] for y, x in valid
    )

    total = len(valid)

    gt_overlap = (
        100 * gt_hits / total if total else 0
    )

    pred_overlap = (
        100 * pred_hits / total if total else 0
    )

    all_coordinates.extend(valid)

    print(
        f"Branch {branch['branch_id']:2d} | "
        f"Pixels: {total:3d} | "
        f"GT overlap: {gt_overlap:6.2f}% | "
        f"Predicted overlap: {pred_overlap:6.2f}%"
    )

# Overall unique-pixel evaluation.
unique_coordinates = set(all_coordinates)

if not unique_coordinates:
    raise ValueError("No valid branch coordinates found.")

gt_total = sum(
    gt_binary[y, x]
    for y, x in unique_coordinates
)

pred_total = sum(
    pred_binary[y, x]
    for y, x in unique_coordinates
)

print("\n========== OVERALL ==========")
print("Unique branch pixels:", len(unique_coordinates))

print(
    "Ground-truth overlap:",
    round(100 * gt_total / len(unique_coordinates), 2),
    "%"
)

print(
    "Predicted-mask overlap:",
    round(100 * pred_total / len(unique_coordinates), 2),
    "%"
)

print("\nGround-truth branch validation completed.")
