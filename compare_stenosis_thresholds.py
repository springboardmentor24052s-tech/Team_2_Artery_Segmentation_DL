
import sys
import json
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

sys.path.insert(0, str(BACKEND))

from app.services.segmentation import predict_stenosis

DATASET_ROOT = (
    ROOT / "backend/app/data/arcade/arcade/arcade"
)

ANNOTATION_PATH = (
    DATASET_ROOT / "stenosis/val/annotations/val.json"
)

IMAGE_DIR = DATASET_ROOT / "stenosis/val/images"

OUTPUT_DIR = (
    ROOT / "backend/results/stenosis_threshold_experiment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]


# --------------------------------------------------
# LOAD ANNOTATIONS
# --------------------------------------------------

with open(ANNOTATION_PATH, "r") as file:
    data = json.load(file)

annotations_by_image = {}

for ann in data["annotations"]:
    if ann["category_id"] == 26:
        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)


# --------------------------------------------------
# CREATE GROUND-TRUTH MASK
# --------------------------------------------------

def create_ground_truth(image_info):

    gt = np.zeros(
        (image_info["height"], image_info["width"]),
        dtype=np.uint8
    )

    for ann in annotations_by_image.get(image_info["id"], []):

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

            cv2.fillPoly(gt, [points], 1)

    return gt > 0


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(gt, pred):

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator else 1.0
    )

    iou = (
        tp / iou_denominator
        if iou_denominator else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp else 0.0
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
# RUN MODEL ONCE PER IMAGE
# --------------------------------------------------

per_threshold = {t: [] for t in THRESHOLDS}

images = data["images"]

processed = 0

for index, image_info in enumerate(images, start=1):

    filename = Path(image_info["file_name"]).name
    image_path = IMAGE_DIR / filename

    if not image_path.is_file():
        print("Missing image:", filename)
        continue

    gt = create_ground_truth(image_info)

    prediction = predict_stenosis(image_path)

    probability_map = np.asarray(
        prediction["probability_map"],
        dtype=np.float32
    )

    if probability_map.shape != gt.shape:
        raise ValueError(
            f"Dimension mismatch: {filename}"
        )

    if not np.isfinite(probability_map).all():
        raise ValueError(
            f"Invalid probability values: {filename}"
        )

    if probability_map.min() < 0 or probability_map.max() > 1:
        raise ValueError(
            f"Probability map outside [0,1]: {filename}"
        )

    for threshold in THRESHOLDS:

        pred = probability_map >= threshold

        metrics = calculate_metrics(gt, pred)

        per_threshold[threshold].append({
            "image": filename,
            **metrics
        })

    processed += 1

    if processed % 20 == 0:
        print(f"Processed {processed}/{len(images)} images")


# --------------------------------------------------
# SUMMARIZE THRESHOLDS
# --------------------------------------------------

summary_rows = []

for threshold in THRESHOLDS:

    rows = per_threshold[threshold]

    if not rows:
        continue

    row = {
        "threshold": threshold,
        "images": len(rows)
    }

    for metric in [
        "dice", "iou", "precision",
        "recall", "specificity"
    ]:
        row[f"macro_{metric}"] = float(
            np.mean([item[metric] for item in rows])
        )

    tp = sum(item["tp"] for item in rows)
    fp = sum(item["fp"] for item in rows)
    fn = sum(item["fn"] for item in rows)
    tn = sum(item["tn"] for item in rows)

    row["global_dice"] = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 1.0
    )

    row["global_iou"] = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 1.0
    )

    row["global_precision"] = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    row["global_recall"] = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    row["global_specificity"] = (
        tn / (tn + fp)
        if tn + fp else 0.0
    )

    summary_rows.append(row)


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

csv_path = OUTPUT_DIR / "threshold_comparison.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n======================================")
print("THRESHOLD COMPARISON RESULTS")
print("======================================")

print("Processed images:", processed)

print(
    "\nThreshold | Dice | IoU | Precision | Recall | Specificity"
)

for row in summary_rows:

    print(
        f"{row['threshold']:.2f} | "
        f"{row['macro_dice']:.4f} | "
        f"{row['macro_iou']:.4f} | "
        f"{row['macro_precision']:.4f} | "
        f"{row['macro_recall']:.4f} | "
        f"{row['macro_specificity']:.4f}"
    )

best = max(
    summary_rows,
    key=lambda row: row["macro_dice"]
)

print("\nBest validation Dice threshold:", best["threshold"])
print("Best macro Dice:", best["macro_dice"])

print("\nCSV saved:", csv_path)

with open(OUTPUT_DIR / "threshold_summary.json", "w") as file:
    json.dump(summary_rows, file, indent=4)

print("Threshold experiment completed.")
