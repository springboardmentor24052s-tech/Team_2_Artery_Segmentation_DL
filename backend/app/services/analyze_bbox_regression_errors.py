
from pathlib import Path
from ultralytics import YOLO
import json
import csv
import math
import statistics

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
    / "final_test_evaluation" / "bbox_regression_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 512
CONFIDENCE = 0.10
NMS_IOU = 0.30
MATCH_IOU = 0.50


# --------------------------------------------------
# BOX UTILITIES
# --------------------------------------------------

def box_area(box):
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return width * height


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = box_area(box1)
    area2 = box_area(box2)

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def get_box_dimensions(box):

    width = box[2] - box[0]
    height = box[3] - box[1]

    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2

    return {
        "width": width,
        "height": height,
        "center_x": center_x,
        "center_y": center_y,
        "area": width * height
    }


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

def match_boxes(gt_boxes, predictions):

    candidates = []

    for gt_index, gt_box in enumerate(gt_boxes):

        for pred_index, prediction in enumerate(predictions):

            iou = calculate_iou(
                gt_box,
                prediction["box"]
            )

            if iou >= MATCH_IOU:

                candidates.append(
                    (iou, gt_index, pred_index)
                )

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()
    matched_pairs = []

    for iou, gt_index, pred_index in candidates:

        if (
            gt_index not in matched_gt
            and pred_index not in matched_predictions
        ):

            matched_gt.add(gt_index)
            matched_predictions.add(pred_index)

            matched_pairs.append(
                (gt_index, pred_index, iou)
            )

    return matched_pairs


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

    records = []

    unmatched_gt = 0
    unmatched_predictions = 0

    print(f"Test images: {len(image_paths)}")
    print("Starting bounding-box regression analysis...")

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

        matched_pairs = match_boxes(
            gt_boxes,
            predictions
        )

        matched_gt = {
            pair[0] for pair in matched_pairs
        }

        matched_pred = {
            pair[1] for pair in matched_pairs
        }

        unmatched_gt += len(gt_boxes) - len(matched_gt)
        unmatched_predictions += (
            len(predictions) - len(matched_pred)
        )

        for gt_index, pred_index, iou in matched_pairs:

            gt_box = gt_boxes[gt_index]
            pred_box = predictions[pred_index]["box"]

            gt = get_box_dimensions(gt_box)
            pred = get_box_dimensions(pred_box)

            center_dx = pred["center_x"] - gt["center_x"]
            center_dy = pred["center_y"] - gt["center_y"]

            center_distance = math.sqrt(
                center_dx ** 2 + center_dy ** 2
            )

            width_error = pred["width"] - gt["width"]
            height_error = pred["height"] - gt["height"]

            width_ratio = (
                pred["width"] / gt["width"]
                if gt["width"] > 0 else 0
            )

            height_ratio = (
                pred["height"] / gt["height"]
                if gt["height"] > 0 else 0
            )

            area_ratio = (
                pred["area"] / gt["area"]
                if gt["area"] > 0 else 0
            )

            records.append({
                "image": image_path.name,
                "gt_index": gt_index,
                "prediction_index": pred_index,
                "confidence": predictions[pred_index]["confidence"],
                "iou": round(iou, 4),
                "gt_width": round(gt["width"], 2),
                "pred_width": round(pred["width"], 2),
                "width_error": round(width_error, 2),
                "width_ratio": round(width_ratio, 4),
                "gt_height": round(gt["height"], 2),
                "pred_height": round(pred["height"], 2),
                "height_error": round(height_error, 2),
                "height_ratio": round(height_ratio, 4),
                "center_dx": round(center_dx, 2),
                "center_dy": round(center_dy, 2),
                "center_distance": round(center_distance, 2),
                "area_ratio": round(area_ratio, 4)
            })

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    def average(field):

        values = [record[field] for record in records]

        return (
            round(statistics.mean(values), 4)
            if values else 0
        )

    def median(field):

        values = [record[field] for record in records]

        return (
            round(statistics.median(values), 4)
            if values else 0
        )

    summary = {
        "images": len(image_paths),
        "matched_pairs": len(records),
        "unmatched_ground_truth": unmatched_gt,
        "unmatched_predictions": unmatched_predictions,
        "mean_iou": average("iou"),
        "mean_center_distance": average("center_distance"),
        "median_center_distance": median("center_distance"),
        "mean_width_error": average("width_error"),
        "mean_height_error": average("height_error"),
        "mean_width_ratio": average("width_ratio"),
        "mean_height_ratio": average("height_ratio"),
        "mean_area_ratio": average("area_ratio")
    }

    # Directional statistics
    summary["width_bias"] = {
        "predictions_wider": sum(
            r["width_error"] > 0 for r in records
        ),
        "predictions_narrower": sum(
            r["width_error"] < 0 for r in records
        ),
        "equal": sum(
            r["width_error"] == 0 for r in records
        )
    }

    summary["height_bias"] = {
        "predictions_taller": sum(
            r["height_error"] > 0 for r in records
        ),
        "predictions_shorter": sum(
            r["height_error"] < 0 for r in records
        ),
        "equal": sum(
            r["height_error"] == 0 for r in records
        )
    }

    # --------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------

    report = {
        "configuration": {
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "matching_iou": MATCH_IOU
        },
        "summary": summary,
        "matched_box_records": records
    }

    json_path = OUTPUT_DIR / "bbox_regression_analysis.json"

    with open(json_path, "w") as file:
        json.dump(report, file, indent=4)

    csv_path = OUTPUT_DIR / "bbox_regression_records.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()) if records else [
                "image"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nBOUNDING-BOX REGRESSION ANALYSIS")
    print("=" * 55)
    print(json.dumps(summary, indent=4))

    print("\nReports saved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
