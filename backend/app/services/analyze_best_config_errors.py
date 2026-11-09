
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import json
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = (
    BACKEND_ROOT / "results" / "stenosis_yolo_baseline" / "best.pt"
)

DATASET_ROOT = (
    BACKEND_ROOT / "results" / "stenosis_yolo_dataset"
)

IMAGE_DIR = DATASET_ROOT / "images" / "test"
LABEL_DIR = DATASET_ROOT / "labels" / "test"

OUTPUT_DIR = (
    BACKEND_ROOT / "results" / "nms_analysis"
    / "final_test_evaluation" / "error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
CONFIDENCE = 0.10
NMS_IOU = 0.30
MATCH_IOU = 0.50

# COCO-style box-area categories, based on pixel area
SMALL_LIMIT = 32 ** 2
MEDIUM_LIMIT = 96 ** 2


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def box_area(box):
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return width * height


def get_size_category(box):
    area = box_area(box)

    if area < SMALL_LIMIT:
        return "small"
    elif area < MEDIUM_LIMIT:
        return "medium"
    return "large"


def load_ground_truth(label_path):
    boxes = []

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, cx, cy, width, height = map(float, values)

            if int(class_id) != 0:
                continue

            x1 = (cx - width / 2) * IMAGE_SIZE
            y1 = (cy - height / 2) * IMAGE_SIZE
            x2 = (cx + width / 2) * IMAGE_SIZE
            y2 = (cy + height / 2) * IMAGE_SIZE

            boxes.append([x1, y1, x2, y2])

    return boxes


def match_predictions(gt_boxes, pred_boxes):
    candidates = []

    for gt_index, gt in enumerate(gt_boxes):
        for pred_index, pred in enumerate(pred_boxes):
            iou = calculate_iou(gt, pred["box"])

            if iou >= MATCH_IOU:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matched_pairs = []

    for iou, gt_index, pred_index in candidates:
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            matched_pairs.append((gt_index, pred_index, iou))

    missed_gt = [
        index for index in range(len(gt_boxes))
        if index not in matched_gt
    ]

    false_positive_predictions = [
        index for index in range(len(pred_boxes))
        if index not in matched_pred
    ]

    return matched_pairs, missed_gt, false_positive_predictions


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )

    if not image_paths:
        raise RuntimeError("No test images found.")

    totals = {
        "TP": 0,
        "FP": 0,
        "FN": 0
    }

    size_stats = {
        category: {
            "ground_truth": 0,
            "matched": 0,
            "missed": 0
        }
        for category in ["small", "medium", "large"]
    }

    error_images = []
    fp_confidences = []
    image_error_rows = []

    print(f"Analysing {len(image_paths)} test images...")
    print(f"Confidence={CONFIDENCE}, NMS IoU={NMS_IOU}")

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

        matched_pairs, missed_gt, false_positives = (
            match_predictions(gt_boxes, pred_boxes)
        )

        tp = len(matched_pairs)
        fp = len(false_positives)
        fn = len(missed_gt)

        totals["TP"] += tp
        totals["FP"] += fp
        totals["FN"] += fn

        # Count ground-truth objects by size
        for gt_index, gt in enumerate(gt_boxes):
            category = get_size_category(gt)
            size_stats[category]["ground_truth"] += 1

            if gt_index in {pair[0] for pair in matched_pairs}:
                size_stats[category]["matched"] += 1
            else:
                size_stats[category]["missed"] += 1

        # False-positive confidence values
        for pred_index in false_positives:
            fp_confidences.append(
                pred_boxes[pred_index]["confidence"]
            )

        if fp > 0 or fn > 0:
            error_images.append({
                "image": image_path.name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "total_errors": fp + fn
            })

        image_error_rows.append({
            "image": image_path.name,
            "ground_truth": len(gt_boxes),
            "predictions": len(pred_boxes),
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    precision = (
        totals["TP"] / (totals["TP"] + totals["FP"])
        if totals["TP"] + totals["FP"] else 0
    )

    recall = (
        totals["TP"] / (totals["TP"] + totals["FN"])
        if totals["TP"] + totals["FN"] else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    for category in size_stats:
        gt_count = size_stats[category]["ground_truth"]
        matched = size_stats[category]["matched"]

        size_stats[category]["recall"] = (
            round(matched / gt_count, 4)
            if gt_count else 0
        )

    error_images.sort(
        key=lambda item: item["total_errors"],
        reverse=True
    )

    report = {
        "configuration": {
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "matching_iou": MATCH_IOU
        },
        "images": len(image_paths),
        "overall": {
            **totals,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "size_wise_analysis": size_stats,
        "false_positive_confidence": {
            "count": len(fp_confidences),
            "average": (
                round(sum(fp_confidences) / len(fp_confidences), 4)
                if fp_confidences else 0
            )
        },
        "images_with_errors": len(error_images),
        "top_20_error_images": error_images[:20]
    }

    json_path = OUTPUT_DIR / "test_error_analysis.json"

    with open(json_path, "w") as file:
        json.dump(report, file, indent=4)

    csv_path = OUTPUT_DIR / "image_wise_errors.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image", "ground_truth", "predictions",
                "TP", "FP", "FN"
            ]
        )
        writer.writeheader()
        writer.writerows(image_error_rows)

    print("\nTEST ERROR ANALYSIS")
    print("=" * 50)
    print(json.dumps(report, indent=4))
    print(f"\nJSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
