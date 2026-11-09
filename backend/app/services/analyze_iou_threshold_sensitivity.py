
from pathlib import Path
from ultralytics import YOLO
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
    / "final_test_evaluation" / "iou_sensitivity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 512
CONFIDENCE = 0.10
NMS_IOU = 0.30

MATCHING_THRESHOLDS = [
    0.30, 0.40, 0.50, 0.60, 0.70
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
# ONE-TO-ONE MATCHING
# --------------------------------------------------

def match_predictions(gt_boxes, pred_boxes, threshold):

    candidates = []

    for gt_index, gt in enumerate(gt_boxes):

        for pred_index, pred in enumerate(pred_boxes):

            iou = calculate_iou(
                gt,
                pred["box"]
            )

            if iou >= threshold:

                candidates.append(
                    (iou, gt_index, pred_index)
                )

    # Highest IoU pairs first
    candidates.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()

    for iou, gt_index, pred_index in candidates:

        if (
            gt_index not in matched_gt
            and pred_index not in matched_predictions
        ):

            matched_gt.add(gt_index)
            matched_predictions.add(pred_index)

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in [
            ".png", ".jpg", ".jpeg"
        ]
    ])

    if not image_paths:
        raise RuntimeError("No test images found.")

    # Generate predictions only once
    # Matching IoU changes; confidence and NMS stay fixed.

    cached_data = []

    print("Generating predictions...")

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

        pred_boxes = []

        for box in result.boxes:

            pred_boxes.append({
                "box": box.xyxy[0].cpu().tolist(),
                "confidence": float(box.conf[0].cpu())
            })

        cached_data.append({
            "image": image_path.name,
            "ground_truth": gt_boxes,
            "predictions": pred_boxes
        })

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    # Evaluate the cached predictions at each threshold

    results = []

    for threshold in MATCHING_THRESHOLDS:

        print(f"\nMatching IoU threshold: {threshold}")

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for item in cached_data:

            tp, fp, fn = match_predictions(
                item["ground_truth"],
                item["predictions"],
                threshold
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp else 0
        )

        recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        row = {
            "matching_iou": threshold,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

        results.append(row)

        print(json.dumps(row, indent=4))

    # Save results

    report = {
        "model": str(MODEL_PATH),
        "images": len(image_paths),
        "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU,
        "matching_results": results
    }

    json_path = (
        OUTPUT_DIR / "iou_threshold_sensitivity_report.json"
    )

    with open(json_path, "w") as file:
        json.dump(report, file, indent=4)

    csv_path = (
        OUTPUT_DIR / "iou_threshold_sensitivity.csv"
    )

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "matching_iou",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "f1_score"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nIoU sensitivity analysis completed.")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
