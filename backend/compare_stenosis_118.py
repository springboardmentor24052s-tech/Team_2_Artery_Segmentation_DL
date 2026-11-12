
import json
import cv2
import numpy as np

from pathlib import Path
from app.services.segmentation import predict_stenosis

BASE_DIR = Path(__file__).resolve().parent

IMAGE_NAME = "41.png"
CATEGORY_ID = 26

IMAGE_DIR = (
    BASE_DIR / "app" / "data" / "arcade" / "arcade"
    / "arcade" / "stenosis" / "test" / "images"
)

ANNOTATION_FILE = (
    BASE_DIR / "app" / "data" / "arcade" / "arcade"
    / "arcade" / "stenosis" / "test" / "annotations"
    / "test.json"
)

OUTPUT_DIR = BASE_DIR / "results" / "comparison_41"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_path = IMAGE_DIR / IMAGE_NAME

if not image_path.exists():
    raise FileNotFoundError(image_path)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

image_info = next(
    (
        img for img in data["images"]
        if Path(img["file_name"]).name == IMAGE_NAME
    ),
    None
)

if image_info is None:
    raise ValueError(f"{IMAGE_NAME} not found in test.json")

image_id = image_info["id"]
height = image_info["height"]
width = image_info["width"]

# Read original image
original = cv2.imread(
    str(image_path), cv2.IMREAD_GRAYSCALE
)

if original is None:
    raise RuntimeError("Could not read original image.")

# Generate ground-truth mask
gt = np.zeros((height, width), dtype=np.uint8)

for ann in data["annotations"]:

    if ann["image_id"] != image_id:
        continue

    if ann["category_id"] != CATEGORY_ID:
        continue

    for polygon in ann.get("segmentation", []):

        if isinstance(polygon, list) and len(polygon) >= 6:
            points = np.array(
                polygon, dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(points).astype(np.int32)
            cv2.fillPoly(gt, [points], 255)

if not np.any(gt):
    raise RuntimeError("No ground-truth stenosis pixels found.")

# Run existing trained model
prediction = predict_stenosis(image_path)
pred = prediction["mask"]

if pred.shape != gt.shape:
    raise ValueError("Prediction and ground truth dimensions differ.")

# Create comparison overlay:
# Green = ground truth only
# Red = prediction only
# Yellow = overlap
comparison = np.zeros((height, width, 3), dtype=np.uint8)

gt_bool = gt > 0
pred_bool = pred > 0

comparison[gt_bool & ~pred_bool] = (0, 255, 0)
comparison[pred_bool & ~gt_bool] = (0, 0, 255)
comparison[gt_bool & pred_bool] = (0, 255, 255)

# Calculate metrics
tp = int(np.logical_and(gt_bool, pred_bool).sum())
fp = int(np.logical_and(~gt_bool, pred_bool).sum())
fn = int(np.logical_and(gt_bool, ~pred_bool).sum())

dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
iou = (tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0

# Save all images
cv2.imwrite(str(OUTPUT_DIR / "41_original.png"), original)
cv2.imwrite(str(OUTPUT_DIR / "41_ground_truth.png"), gt)
cv2.imwrite(str(OUTPUT_DIR / "41_prediction.png"), pred)
cv2.imwrite(str(OUTPUT_DIR / "41_comparison.png"), comparison)

print("\n========== IMAGE COMPARISON ==========")
print("Image:", IMAGE_NAME)
print("Image ID:", image_id)
print("Ground-truth pixels:", int(gt_bool.sum()))
print("Predicted pixels:", int(pred_bool.sum()))
print("True Positives:", tp)
print("False Positives:", fp)
print("False Negatives:", fn)
print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")
print("Output folder:", OUTPUT_DIR)
print("======================================")

print("\nGreen = Ground truth only")
print("Red = Prediction only")
print("Yellow = Overlapping pixels")
