
import sys
import json
import csv
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

DATASET_ROOT = (
    ROOT / "backend/app/data/arcade/arcade/arcade"
)

ANNOTATION_PATH = (
    DATASET_ROOT / "stenosis/val/annotations/val.json"
)

IMAGE_DIR = DATASET_ROOT / "stenosis/val/images"

OUTPUT_DIR = (
    ROOT / "backend/results/stenosis_validation_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# IMPORT TRAINED MODEL
# --------------------------------------------------

sys.path.insert(0, str(BACKEND))

from app.services.segmentation import predict_stenosis


# --------------------------------------------------
# LOAD COCO ANNOTATIONS
# --------------------------------------------------

with open(ANNOTATION_PATH, "r") as file:
    data = json.load(file)

images = data["images"]

stenosis_annotations = [
    ann for ann in data["annotations"]
    if ann["category_id"] == 26
]

annotations_by_image = {}

for ann in stenosis_annotations:
    annotations_by_image.setdefault(
        ann["image_id"], []
    ).append(ann)

print("\n======================================")
print("STENOSIS VALIDATION EVALUATION")
print("======================================")

print("Validation images:", len(images))
print("Stenosis annotations:", len(stenosis_annotations))


# --------------------------------------------------
# CREATE GROUND-TRUTH MASK
# --------------------------------------------------

def create_ground_truth(image_info):

    height = image_info["height"]
    width = image_info["width"]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    annotations = annotations_by_image.get(
        image_info["id"], []
    )

    for ann in annotations:

        segmentation = ann.get("segmentation", [])

        if not isinstance(segmentation, list):
            continue

        for polygon in segmentation:

            if len(polygon) < 6:
                continue

            points = np.array(
                polygon,
                dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(points).astype(np.int32)

            cv2.fillPoly(
                mask,
                [points],
                1
            )

    return mask > 0


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(gt, pred):

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


# --------------------------------------------------
# EVALUATE ALL VALIDATION IMAGES
# --------------------------------------------------

results = []

total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0

processed = 0
skipped = 0

for index, image_info in enumerate(images, start=1):

    filename = Path(image_info["file_name"]).name

    image_path = IMAGE_DIR / filename

    if not image_path.is_file():
        print("Skipping missing image:", filename)
        skipped += 1
        continue

    try:

        # Ground-truth mask
        gt = create_ground_truth(image_info)

        # Model prediction
        prediction = predict_stenosis(image_path)

        pred_mask = prediction["mask"]

        if pred_mask.shape != gt.shape:
            raise ValueError(
                f"Dimension mismatch for {filename}"
            )

        pred = pred_mask > 0

        # Calculate metrics
        metrics = calculate_metrics(gt, pred)

        # Accumulate counts
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_tn += metrics["tn"]

        results.append({
            "image": filename,
            "ground_truth_pixels": int(gt.sum()),
            "predicted_pixels": int(pred.sum()),
            **metrics
        })

        processed += 1

        if processed % 20 == 0:
            print(
                f"Processed {processed}/{len(images)} images"
            )

    except Exception as error:

        print(
            f"Error processing {filename}: {error}"
        )

        skipped += 1


# --------------------------------------------------
# MACRO AVERAGE METRICS
# --------------------------------------------------

if not results:
    raise RuntimeError(
        "No validation images were evaluated successfully."
    )

metric_names = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity"
]

macro_metrics = {
    name: float(
        np.mean([row[name] for row in results])
    )
    for name in metric_names
}


# --------------------------------------------------
# GLOBAL PIXEL METRICS
# --------------------------------------------------

global_metrics = calculate_metrics(
    np.concatenate([
        np.zeros(row["tn"], dtype=bool)
        for row in []
    ]) if False else np.array([False]),
    np.array([False])
)

# Calculate global metrics directly from accumulated counts.
tp = total_tp
fp = total_fp
fn = total_fn
tn = total_tn

global_dice = (
    2 * tp / (2 * tp + fp + fn)
    if (2 * tp + fp + fn) > 0 else 1.0
)

global_iou = (
    tp / (tp + fp + fn)
    if (tp + fp + fn) > 0 else 1.0
)

global_precision = (
    tp / (tp + fp)
    if (tp + fp) > 0 else 0.0
)

global_recall = (
    tp / (tp + fn)
    if (tp + fn) > 0 else 0.0
)

global_specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else 0.0
)


# --------------------------------------------------
# SAVE PER-IMAGE CSV
# --------------------------------------------------

csv_path = OUTPUT_DIR / "per_image_metrics.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# SAVE SUMMARY JSON
# --------------------------------------------------

summary = {
    "dataset": "ARCADE stenosis validation",
    "images_available": len(images),
    "images_processed": processed,
    "images_skipped": skipped,
    "annotations": len(stenosis_annotations),
    "macro_metrics": macro_metrics,
    "global_pixel_metrics": {
        "dice": global_dice,
        "iou": global_iou,
        "precision": global_precision,
        "recall": global_recall,
        "specificity": global_specificity
    }
}

summary_path = OUTPUT_DIR / "summary.json"

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)


# --------------------------------------------------
# PRINT FINAL RESULTS
# --------------------------------------------------

print("\n======================================")
print("VALIDATION RESULTS")
print("======================================")

print("Processed images:", processed)
print("Skipped images:", skipped)

print("\nMACRO AVERAGE METRICS")

for name, value in macro_metrics.items():
    print(f"{name.capitalize():15s}: {value:.4f}")

print("\nGLOBAL PIXEL METRICS")

print(f"Dice:        {global_dice:.4f}")
print(f"IoU:         {global_iou:.4f}")
print(f"Precision:   {global_precision:.4f}")
print(f"Recall:      {global_recall:.4f}")
print(f"Specificity: {global_specificity:.4f}")

print("\nFILES SAVED")
print("CSV:", csv_path)
print("Summary:", summary_path)

print("\nEvaluation completed.")
