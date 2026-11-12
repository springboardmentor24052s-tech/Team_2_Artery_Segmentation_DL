
import json
import csv
from pathlib import Path

import cv2
import numpy as np

from app.services.segmentation import predict_stenosis


# ------------------------------------------
# CONFIGURATION
# ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ANNOTATION_FILE = (
    BASE_DIR / "app" / "data" / "arcade"
    / "arcade" / "arcade" / "stenosis"
    / "test" / "annotations" / "test.json"
)

IMAGE_DIR = (
    BASE_DIR / "app" / "data" / "arcade"
    / "arcade" / "arcade" / "stenosis"
    / "test" / "images"
)

OUTPUT_DIR = BASE_DIR / "results" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_ID = 26

# Start with 10 images for a quick evaluation.
# Change to None later to evaluate all test images.
MAX_IMAGES = 10


# ------------------------------------------
# LOAD ANNOTATIONS
# ------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations_by_image = {}

for annotation in data["annotations"]:
    if annotation["category_id"] == CATEGORY_ID:
        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id, []
        ).append(annotation)


# ------------------------------------------
# CREATE GROUND-TRUTH MASK
# ------------------------------------------

def create_ground_truth(image_info):

    height = image_info["height"]
    width = image_info["width"]
    image_id = image_info["id"]

    mask = np.zeros((height, width), dtype=np.uint8)

    annotations = annotations_by_image.get(
        image_id, []
    )

    for annotation in annotations:

        segmentation = annotation.get(
            "segmentation", []
        )

        if isinstance(segmentation, list):

            for polygon in segmentation:

                if not isinstance(polygon, list):
                    continue

                if len(polygon) < 6:
                    continue

                points = np.array(
                    polygon,
                    dtype=np.float32
                ).reshape(-1, 2)

                points = np.round(points).astype(
                    np.int32
                )

                cv2.fillPoly(mask, [points], 1)

        else:
            raise ValueError(
                f"Unsupported segmentation format "
                f"for image ID {image_id}"
            )

    return mask > 0


# ------------------------------------------
# CALCULATE METRICS
# ------------------------------------------

def calculate_metrics(prediction, ground_truth):

    tp = int(
        np.logical_and(
            prediction, ground_truth
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction, ~ground_truth
        ).sum()
    )

    fn = int(
        np.logical_and(
            ~prediction, ground_truth
        ).sum()
    )

    dice_denominator = (
        prediction.sum() + ground_truth.sum()
    )

    union = np.logical_or(
        prediction, ground_truth
    ).sum()

    dice = (
        2 * tp / dice_denominator
        if dice_denominator else 1.0
    )

    iou = (
        tp / union
        if union else 1.0
    )

    return tp, fp, fn, dice, iou


# ------------------------------------------
# EVALUATE TEST IMAGES
# ------------------------------------------

results = []

images = sorted(
    data["images"],
    key=lambda item: item["file_name"]
)

if MAX_IMAGES is not None:
    images = images[:MAX_IMAGES]

print("\nStarting V3 stenosis evaluation...")
print("Images selected:", len(images))

for index, image_info in enumerate(images, start=1):

    filename = image_info["file_name"]
    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        print("Skipping missing image:", filename)
        continue

    try:

        # Ground truth
        gt = create_ground_truth(image_info)

        # Existing V3 model prediction
        prediction_result = predict_stenosis(
            image_path
        )

        pred = (
            prediction_result["mask"] > 0
        )

        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {filename}"
            )

        tp, fp, fn, dice, iou = calculate_metrics(
            pred, gt
        )

        results.append({
            "image": filename,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "dice": dice,
            "iou": iou
        })

        print(
            f"[{index}/{len(images)}] "
            f"{filename} | "
            f"Dice: {dice:.4f} | "
            f"IoU: {iou:.4f}"
        )

    except Exception as error:

        print(
            f"Error processing {filename}: {error}"
        )


# ------------------------------------------
# SAVE CSV RESULTS
# ------------------------------------------

csv_path = OUTPUT_DIR / "v3_stenosis_test_metrics.csv"

fieldnames = [
    "image", "tp", "fp", "fn", "dice", "iou"
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ------------------------------------------
# FINAL SUMMARY
# ------------------------------------------

if results:

    mean_dice = np.mean([
        result["dice"] for result in results
    ])

    mean_iou = np.mean([
        result["iou"] for result in results
    ])

    print("\n========== V3 EVALUATION SUMMARY ==========")
    print("Successfully evaluated:", len(results))
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU:  {mean_iou:.4f}")
    print("CSV saved at:", csv_path)
    print("===========================================")

else:

    print("No images were successfully evaluated.")
