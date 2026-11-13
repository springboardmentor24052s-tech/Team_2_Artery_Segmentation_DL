
import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

ANNOTATION_PATH = (
    PROJECT_ROOT
    / "backend/app/data/arcade/arcade/arcade/"
    / "stenosis/test/annotations/test.json"
)

PREDICTED_MASK_PATH = (
    PROJECT_ROOT
    / "backend/results/api_predictions/"
    / "15de43c0fbf34daea932512d447e7ffc_stenosis_mask.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend/results/stenosis_single_image_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD GROUND-TRUTH ANNOTATION
# --------------------------------------------------

with open(ANNOTATION_PATH, "r") as file:
    data = json.load(file)

image_info = next(
    (
        image for image in data["images"]
        if Path(image["file_name"]).name == "1.png"
    ),
    None
)

if image_info is None:
    raise ValueError("Image 1.png not found in annotation JSON.")

image_id = image_info["id"]

annotations = [
    annotation
    for annotation in data["annotations"]
    if annotation["image_id"] == image_id
    and annotation["category_id"] == 26
]

if not annotations:
    raise ValueError("No stenosis annotations found for 1.png.")


# --------------------------------------------------
# CREATE GROUND-TRUTH MASK
# --------------------------------------------------

height = image_info["height"]
width = image_info["width"]

ground_truth = np.zeros(
    (height, width),
    dtype=np.uint8
)

for annotation in annotations:

    segmentation = annotation.get("segmentation", [])

    if isinstance(segmentation, list):

        for polygon in segmentation:

            if len(polygon) < 6:
                continue

            points = np.array(
                polygon,
                dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(points).astype(np.int32)

            cv2.fillPoly(
                ground_truth,
                [points],
                255
            )


# --------------------------------------------------
# LOAD PREDICTED MASK
# --------------------------------------------------

if not PREDICTED_MASK_PATH.is_file():
    raise FileNotFoundError(
        f"Predicted mask not found: {PREDICTED_MASK_PATH}"
    )

predicted_image = cv2.imread(
    str(PREDICTED_MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

if predicted_image is None:
    raise ValueError("Unable to read predicted mask.")

if predicted_image.shape != ground_truth.shape:
    raise ValueError(
        "Predicted mask and ground-truth dimensions differ."
    )


# --------------------------------------------------
# BINARY MASKS
# --------------------------------------------------

gt = ground_truth > 0
pred = predicted_image > 0

intersection = np.logical_and(gt, pred).sum()

gt_pixels = gt.sum()
pred_pixels = pred.sum()

union = np.logical_or(gt, pred).sum()

true_positive = intersection
false_positive = np.logical_and(~gt, pred).sum()
false_negative = np.logical_and(gt, ~pred).sum()
true_negative = np.logical_and(~gt, ~pred).sum()


# --------------------------------------------------
# METRICS
# --------------------------------------------------

dice = (
    2 * intersection / (gt_pixels + pred_pixels)
    if gt_pixels + pred_pixels > 0
    else 0.0
)

iou = (
    intersection / union
    if union > 0
    else 0.0
)

precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive > 0
    else 0.0
)

recall = (
    true_positive / (true_positive + false_negative)
    if true_positive + false_negative > 0
    else 0.0
)

specificity = (
    true_negative / (true_negative + false_positive)
    if true_negative + false_positive > 0
    else 0.0
)


# --------------------------------------------------
# CREATE COMPARISON VISUALIZATION
# --------------------------------------------------

comparison = cv2.cvtColor(
    ground_truth,
    cv2.COLOR_GRAY2BGR
)

# Ground truth only: blue
# Prediction only: red
# Overlap: green

comparison[
    np.logical_and(gt, ~pred)
] = [255, 0, 0]

comparison[
    np.logical_and(~gt, pred)
] = [0, 0, 255]

comparison[
    np.logical_and(gt, pred)
] = [0, 255, 0]

comparison_path = OUTPUT_DIR / "mask_comparison.png"

cv2.imwrite(
    str(comparison_path),
    comparison
)

cv2.imwrite(
    str(OUTPUT_DIR / "ground_truth_mask.png"),
    ground_truth
)

cv2.imwrite(
    str(OUTPUT_DIR / "predicted_mask.png"),
    predicted_image
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n===================================")
print("STENOSIS SEGMENTATION EVALUATION")
print("===================================")

print("Image:", image_info["file_name"])
print("Image ID:", image_id)
print("Image dimensions:", width, "x", height)

print("\nPIXEL COUNTS")
print("Ground-truth pixels:", int(gt_pixels))
print("Predicted pixels:", int(pred_pixels))
print("Intersection:", int(intersection))

print("\nEVALUATION METRICS")
print(f"Dice Score:   {dice:.4f}")
print(f"IoU Score:    {iou:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"Specificity:  {specificity:.4f}")

print("\nCOMPARISON IMAGE")
print(comparison_path)

print("\nEvaluation completed.")
