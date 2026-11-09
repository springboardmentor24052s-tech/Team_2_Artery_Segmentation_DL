
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import json
import csv

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = (
    BACKEND_ROOT / "results"
    / "stenosis_yolo_baseline" / "best.pt"
)

DATASET_ROOT = (
    BACKEND_ROOT / "results" / "stenosis_yolo_dataset"
)

IMAGE_DIR = DATASET_ROOT / "images" / "test"
LABEL_DIR = DATASET_ROOT / "labels" / "test"

OUTPUT_DIR = (
    BACKEND_ROOT / "results" / "nms_analysis"
    / "final_test_evaluation" / "localization_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 512
CONFIDENCE = 0.10
NMS_IOU = 0.30
MATCH_IOU = 0.50

IOU_BINS = [
    (0.00, 0.01, "zero_overlap"),
    (0.01, 0.10, "very_low"),
    (0.10, 0.25, "low"),
    (0.25, 0.50, "partial"),
    (0.50, 0.75, "good"),
    (0.75, 1.01, "high")
]


# --------------------------------------------------
# IOU CALCULATION
# --------------------------------------------------

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# --------------------------------------------------
# LOAD GROUND TRUTH
# --------------------------------------------------

def load_ground_truth(label_path):

    boxes = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, cx, cy, width, height = map(
                float, values
            )

            if int(class_id) != 0:
                continue

            x1 = (cx - width / 2) * IMAGE_SIZE
            y1 = (cy - height / 2) * IMAGE_SIZE
            x2 = (cx + width / 2) * IMAGE_SIZE
            y2 = (cy + height / 2) * IMAGE_SIZE

            boxes.append([x1, y1, x2, y2])

    return boxes


# --------------------------------------------------
# GET IOU CATEGORY
# --------------------------------------------------

def get_iou_category(iou):

    for lower, upper, name in IOU_BINS:

        if lower <= iou < upper:
            return name

    return "high"


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted([
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in [
            ".png", ".jpg", ".jpeg"
        ]
    ])

    if not image_paths:
        raise RuntimeError("No test images found.")

    # One record for every ground-truth object
    gt_iou_records = []

    # One record for every prediction
    prediction_iou_records = []

    image_summaries = []

    print(f"Test images: {len(image_paths)}")
    print("Starting localization analysis...")

    for index, image_path in enumerate(image_paths, start=1):

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        gt_boxes = load_ground_truth(label_path)

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device="cpu",
            max_det=300,
            verbose=False
        )[0]

        predictions = []

        for box in result.boxes:

            predictions.append({
                "box": box.xyxy[0].cpu().tolist(),
                "confidence": float(box.conf[0].cpu())
            })

        # Find best prediction for every GT box
        for gt_index, gt_box in enumerate(gt_boxes):

            best_iou = 0.0
            best_prediction = None

            for pred_index, prediction in enumerate(predictions):

                iou = calculate_iou(
                    gt_box,
                    prediction["box"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_prediction = pred_index

            gt_iou_records.append({
                "image": image_path.name,
                "gt_index": gt_index,
                "best_iou": round(best_iou, 4),
                "category": get_iou_category(best_iou),
                "matched_at_0_50": best_iou >= MATCH_IOU,
                "best_prediction_index": best_prediction
            })

        # Find best ground truth for every prediction
        for pred_index, prediction in enumerate(predictions):

            best_iou = 0.0
            best_gt = None

            for gt_index, gt_box in enumerate(gt_boxes):

                iou = calculate_iou(
                    prediction["box"],
                    gt_box
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_index

            prediction_iou_records.append({
                "image": image_path.name,
                "prediction_index": pred_index,
                "confidence": round(
                    prediction["confidence"], 4
                ),
                "best_iou": round(best_iou, 4),
                "category": get_iou_category(best_iou),
                "matched_at_0_50": best_iou >= MATCH_IOU,
                "nearest_gt_index": best_gt
            })

        image_summaries.append({
            "image": image_path.name,
            "ground_truth_count": len(gt_boxes),
            "prediction_count": len(predictions)
        })

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    # --------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------

    gt_categories = Counter(
        record["category"]
        for record in gt_iou_records
    )

    prediction_categories = Counter(
        record["category"]
        for record in prediction_iou_records
    )

    total_gt = len(gt_iou_records)
    total_predictions = len(prediction_iou_records)

    gt_matched = sum(
        record["matched_at_0_50"]
        for record in gt_iou_records
    )

    prediction_matched = sum(
        record["matched_at_0_50"]
        for record in prediction_iou_records
    )

    mean_best_gt_iou = (
        sum(r["best_iou"] for r in gt_iou_records) / total_gt
        if total_gt else 0
    )

    mean_best_prediction_iou = (
        sum(r["best_iou"] for r in prediction_iou_records)
        / total_predictions
        if total_predictions else 0
    )

    summary = {
        "configuration": {
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "matching_iou": MATCH_IOU
        },
        "images": len(image_paths),
        "ground_truth_analysis": {
            "total": total_gt,
            "matched_at_0_50": gt_matched,
            "missed_at_0_50": total_gt - gt_matched,
            "mean_best_iou": round(mean_best_gt_iou, 4),
            "iou_distribution": dict(gt_categories)
        },
        "prediction_analysis": {
            "total": total_predictions,
            "matched_at_0_50": prediction_matched,
            "unmatched_at_0_50": (
                total_predictions - prediction_matched
            ),
            "mean_best_iou": round(
                mean_best_prediction_iou, 4
            ),
            "iou_distribution": dict(prediction_categories)
        }
    }

    # --------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------

    report = {
        "summary": summary,
        "ground_truth_records": gt_iou_records,
        "prediction_records": prediction_iou_records
    }

    json_path = OUTPUT_DIR / "localization_analysis.json"

    with open(json_path, "w") as file:
        json.dump(report, file, indent=4)

    # --------------------------------------------------
    # SAVE GROUND TRUTH CSV
    # --------------------------------------------------

    gt_csv = OUTPUT_DIR / "ground_truth_iou_analysis.csv"

    with open(gt_csv, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "gt_index",
                "best_iou",
                "category",
                "matched_at_0_50",
                "best_prediction_index"
            ]
        )

        writer.writeheader()
        writer.writerows(gt_iou_records)

    # --------------------------------------------------
    # SAVE PREDICTION CSV
    # --------------------------------------------------

    pred_csv = OUTPUT_DIR / "prediction_iou_analysis.csv"

    with open(pred_csv, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "prediction_index",
                "confidence",
                "best_iou",
                "category",
                "matched_at_0_50",
                "nearest_gt_index"
            ]
        )

        writer.writeheader()
        writer.writerows(prediction_iou_records)

    # --------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------

    print("\nLOCALIZATION ANALYSIS SUMMARY")
    print("=" * 55)

    print("\nGround-truth analysis:")
    print(json.dumps(summary["ground_truth_analysis"], indent=4))

    print("\nPrediction analysis:")
    print(json.dumps(summary["prediction_analysis"], indent=4))

    print("\nFiles saved:")
    print(json_path)
    print(gt_csv)
    print(pred_csv)


if __name__ == "__main__":
    main()
