
import json
import csv
from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_baseline"
    / "best.pt"
)

DATASET_ROOT = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
)

TEST_IMAGES = DATASET_ROOT / "images" / "test"
TEST_LABELS = DATASET_ROOT / "labels" / "test"

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "nms_analysis"
    / "final_test_evaluation"
    / "error_categorization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.10
NMS_IOU = 0.30
MATCH_IOU = 0.50
IMAGE_SIZE = 512
MAX_DETECTIONS = 300


# ============================================================
# BOUNDING BOX UTILITIES
# ============================================================

def yolo_to_xyxy(label, image_width, image_height):
    """Convert normalized YOLO coordinates to pixel coordinates."""

    class_id, x_center, y_center, width, height = map(float, label)

    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height

    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    """Calculate intersection over union."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )
    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(label_path, image_width, image_height):
    """Read ground-truth YOLO labels."""

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            boxes.append(
                yolo_to_xyxy(
                    values,
                    image_width,
                    image_height
                )
            )

    return boxes


# ============================================================
# ONE-TO-ONE MATCHING
# ============================================================

def match_predictions(ground_truth, predictions):
    """
    Match predictions to ground truth using descending IoU.

    Each ground-truth box and prediction can be matched once.
    """

    possible_matches = []

    for gt_index, gt_box in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):

            iou = calculate_iou(gt_box, pred["box"])

            if iou >= MATCH_IOU:
                possible_matches.append(
                    (iou, gt_index, pred_index)
                )

    possible_matches.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()
    matches = []

    for iou, gt_index, pred_index in possible_matches:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        matches.append({
            "gt_index": gt_index,
            "pred_index": pred_index,
            "iou": round(iou, 4)
        })

    return matched_gt, matched_predictions, matches


# ============================================================
# ERROR CATEGORIZATION
# ============================================================

def categorize_image_errors(ground_truth, predictions):

    matched_gt, matched_predictions, matches = match_predictions(
        ground_truth,
        predictions
    )

    missed_gt_indices = [
        i for i in range(len(ground_truth))
        if i not in matched_gt
    ]

    unmatched_pred_indices = [
        i for i in range(len(predictions))
        if i not in matched_predictions
    ]

    missed_regions = []

    for gt_index in missed_gt_indices:

        gt_box = ground_truth[gt_index]

        best_iou = 0.0

        for pred in predictions:
            best_iou = max(
                best_iou,
                calculate_iou(gt_box, pred["box"])
            )

        if best_iou == 0:
            category = "missed_no_overlap"
        elif best_iou < 0.25:
            category = "missed_very_low_overlap"
        else:
            category = "missed_partial_overlap"

        missed_regions.append({
            "ground_truth_index": gt_index,
            "category": category,
            "best_iou": round(best_iou, 4)
        })

    false_positives = []

    for pred_index in unmatched_pred_indices:

        prediction = predictions[pred_index]

        best_iou = 0.0

        for gt_box in ground_truth:
            best_iou = max(
                best_iou,
                calculate_iou(prediction["box"], gt_box)
            )

        if best_iou == 0:
            category = "false_positive_no_overlap"
        elif best_iou < 0.25:
            category = "false_positive_low_overlap"
        else:
            category = "false_positive_partial_overlap"

        false_positives.append({
            "prediction_index": pred_index,
            "category": category,
            "confidence": round(
                prediction["confidence"], 4
            ),
            "best_iou": round(best_iou, 4)
        })

    return {
        "matches": matches,
        "missed_regions": missed_regions,
        "false_positives": false_positives
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("Loading YOLO model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"Test image folder not found: {TEST_IMAGES}"
        )

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        list(TEST_IMAGES.glob("*.png"))
        + list(TEST_IMAGES.glob("*.jpg"))
        + list(TEST_IMAGES.glob("*.jpeg"))
    )

    if not image_paths:
        raise RuntimeError("No test images found.")

    print(f"Test images: {len(image_paths)}")
    print("Starting error categorization...")

    total = Counter()
    image_results = []

    for index, image_path in enumerate(image_paths, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: Could not read {image_path.name}")
            continue

        image_height, image_width = image.shape[:2]

        label_path = TEST_LABELS / (
            image_path.stem + ".txt"
        )

        ground_truth = load_ground_truth(
            label_path,
            image_width,
            image_height
        )

        result = model.predict(
            source=str(image_path),
            conf=CONFIDENCE,
            iou=NMS_IOU,
            imgsz=IMAGE_SIZE,
            max_det=MAX_DETECTIONS,
            device="cpu",
            verbose=False
        )[0]

        predictions = []

        if result.boxes is not None:

            for box in result.boxes:

                coordinates = (
                    box.xyxy[0].cpu().numpy().tolist()
                )

                confidence = float(
                    box.conf[0].cpu().item()
                )

                predictions.append({
                    "box": coordinates,
                    "confidence": confidence
                })

        categorized = categorize_image_errors(
            ground_truth,
            predictions
        )

        matches = categorized["matches"]
        missed = categorized["missed_regions"]
        false_positives = categorized["false_positives"]

        total["images"] += 1
        total["ground_truth"] += len(ground_truth)
        total["predictions"] += len(predictions)
        total["matched"] += len(matches)
        total["missed"] += len(missed)
        total["false_positives"] += len(false_positives)

        for item in missed:
            total[item["category"]] += 1

        for item in false_positives:
            total[item["category"]] += 1

        if len(predictions) > len(matches):
            total["images_with_unmatched_predictions"] += 1

        if missed:
            total["images_with_missed_regions"] += 1

        image_results.append({
            "image": image_path.name,
            "ground_truth_count": len(ground_truth),
            "prediction_count": len(predictions),
            "matched_count": len(matches),
            "missed_regions": missed,
            "false_positives": false_positives
        })

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "configuration": {
            "model": "YOLOv8n V1",
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "matching_iou": MATCH_IOU,
            "image_size": IMAGE_SIZE
        },
        "overall": {
            "images": total["images"],
            "ground_truth": total["ground_truth"],
            "predictions": total["predictions"],
            "matched": total["matched"],
            "missed": total["missed"],
            "false_positives": total["false_positives"]
        },
        "missed_region_categories": {
            "no_overlap": total["missed_no_overlap"],
            "very_low_overlap": total[
                "missed_very_low_overlap"
            ],
            "partial_overlap": total[
                "missed_partial_overlap"
            ]
        },
        "false_positive_categories": {
            "no_overlap": total[
                "false_positive_no_overlap"
            ],
            "low_overlap": total[
                "false_positive_low_overlap"
            ],
            "partial_overlap": total[
                "false_positive_partial_overlap"
            ]
        },
        "image_level": {
            "images_with_missed_regions": total[
                "images_with_missed_regions"
            ],
            "images_with_unmatched_predictions": total[
                "images_with_unmatched_predictions"
            ]
        }
    }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    json_path = OUTPUT_DIR / "error_categorization_report.json"

    with open(json_path, "w") as file:
        json.dump(
            {
                "summary": summary,
                "image_results": image_results
            },
            file,
            indent=4
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = OUTPUT_DIR / "image_error_categories.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "image",
            "ground_truth_count",
            "prediction_count",
            "matched_count",
            "missed_count",
            "false_positive_count",
            "missed_no_overlap",
            "missed_very_low_overlap",
            "missed_partial_overlap",
            "fp_no_overlap",
            "fp_low_overlap",
            "fp_partial_overlap"
        ])

        for item in image_results:

            missed_counts = Counter(
                x["category"] for x in item["missed_regions"]
            )

            fp_counts = Counter(
                x["category"] for x in item["false_positives"]
            )

            writer.writerow([
                item["image"],
                item["ground_truth_count"],
                item["prediction_count"],
                item["matched_count"],
                len(item["missed_regions"]),
                len(item["false_positives"]),
                missed_counts["missed_no_overlap"],
                missed_counts["missed_very_low_overlap"],
                missed_counts["missed_partial_overlap"],
                fp_counts["false_positive_no_overlap"],
                fp_counts["false_positive_low_overlap"],
                fp_counts["false_positive_partial_overlap"]
            ])

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nSTENOSIS DETECTION ERROR CATEGORIZATION")
    print("=" * 55)

    print(json.dumps(summary, indent=4))

    print("\nReports saved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
