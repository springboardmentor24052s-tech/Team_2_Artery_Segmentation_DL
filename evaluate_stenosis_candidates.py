
import json
import csv
from pathlib import Path

import numpy as np

from app.services.segmentation import predict_stenosis
from app.services.candidate_extraction import (
    extract_stenosis_candidates
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_DIR = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
    / "test"
)

IMAGE_DIR = TEST_DIR / "images"
ANNOTATION_FILE = TEST_DIR / "annotations" / "test.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_candidate_evaluation"
)

MIN_AREA = 10
IOU_THRESHOLD = 0.50


# --------------------------------------------------
# BOUNDING BOX IOU
# --------------------------------------------------

def calculate_iou(box_a, box_b):
    """
    Calculate IoU between two boxes.

    Box format:
    [x, y, width, height]
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)

    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# ONE-TO-ONE GREEDY MATCHING
# --------------------------------------------------

def match_candidates(predictions, ground_truth, threshold):
    """
    Match predicted candidates with ground-truth boxes.

    Each prediction and each annotation can be matched
    at most once.

    Matching is performed using descending IoU.
    """

    possible_matches = []

    for pred_index, prediction in enumerate(predictions):

        pred_box = prediction["bounding_box"]

        pred_box = [
            pred_box["x"],
            pred_box["y"],
            pred_box["width"],
            pred_box["height"]
        ]

        for gt_index, gt_box in enumerate(ground_truth):

            iou = calculate_iou(pred_box, gt_box)

            if iou >= threshold:
                possible_matches.append(
                    (iou, pred_index, gt_index)
                )

    possible_matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    matched_predictions = set()
    matched_ground_truth = set()

    matches = []

    for iou, pred_index, gt_index in possible_matches:

        if pred_index in matched_predictions:
            continue

        if gt_index in matched_ground_truth:
            continue

        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)

        matches.append({
            "prediction_index": pred_index,
            "ground_truth_index": gt_index,
            "iou": round(float(iou), 4)
        })

    true_positives = len(matches)

    false_positives = (
        len(predictions) - true_positives
    )

    false_negatives = (
        len(ground_truth) - true_positives
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matches": matches
    }


# --------------------------------------------------
# LOAD COCO ANNOTATIONS
# --------------------------------------------------

def load_ground_truth():

    if not ANNOTATION_FILE.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    stenosis_category_ids = {
        category_id
        for category_id, name in categories.items()
        if name.lower() == "stenosis"
    }

    if not stenosis_category_ids:
        raise ValueError(
            "Stenosis category was not found in annotations."
        )

    image_records = {
        image["id"]: image
        for image in data["images"]
    }

    ground_truth = {
        image_id: []
        for image_id in image_records
    }

    for annotation in data["annotations"]:

        if (
            annotation["category_id"]
            not in stenosis_category_ids
        ):
            continue

        x, y, width, height = annotation["bbox"]

        ground_truth[annotation["image_id"]].append(
            [
                float(x),
                float(y),
                float(width),
                float(height)
            ]
        )

    return image_records, ground_truth


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_records, ground_truth = load_ground_truth()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_predictions = 0
    total_annotations = 0

    image_results = []

    all_matched_ious = []

    print("\nStarting V4 candidate evaluation...")
    print("Model: Attention U-Net V4")
    print("Test images:", len(image_records))
    print("IoU threshold:", IOU_THRESHOLD)
    print("Minimum component area:", MIN_AREA)

    # Process images in dataset order.
    for image_id, image_info in sorted(
        image_records.items()
    ):

        image_name = image_info["file_name"]

        image_path = IMAGE_DIR / image_name

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Test image not found: {image_path}"
            )

        # Ground-truth stenosis boxes.
        gt_boxes = ground_truth[image_id]

        # V4 prediction.
        prediction = predict_stenosis(image_path)

        # Extract connected-component candidates.
        extraction = extract_stenosis_candidates(
            probability_map=prediction["probability_map"],
            binary_mask=prediction["mask"],
            min_area=MIN_AREA
        )

        candidates = extraction["candidates"]

        # Match predictions against annotations.
        matching = match_candidates(
            predictions=candidates,
            ground_truth=gt_boxes,
            threshold=IOU_THRESHOLD
        )

        tp = matching["true_positives"]
        fp = matching["false_positives"]
        fn = matching["false_negatives"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_predictions += len(candidates)
        total_annotations += len(gt_boxes)

        for match in matching["matches"]:
            all_matched_ious.append(match["iou"])

        image_results.append({
            "image_id": image_id,
            "file_name": image_name,
            "ground_truth_count": len(gt_boxes),
            "predicted_candidate_count": len(candidates),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "matches": matching["matches"]
        })

        if len(image_results) % 25 == 0:
            print(
                f"Processed {len(image_results)} "
                f"/ {len(image_records)} images"
            )

    # --------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    mean_matched_iou = (
        float(np.mean(all_matched_ious))
        if all_matched_ious
        else 0.0
    )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    report = {
        "model": "Attention U-Net V4",
        "dataset": "ARCADE stenosis test set",
        "evaluation_type": "Bounding-box candidate matching",
        "test_images": len(image_records),
        "ground_truth_annotations": total_annotations,
        "predicted_candidates": total_predictions,
        "minimum_component_area": MIN_AREA,
        "iou_threshold": IOU_THRESHOLD,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "mean_matched_iou": round(mean_matched_iou, 4),
        "images": image_results
    }

    report_path = (
        OUTPUT_DIR / "v4_candidate_evaluation.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # Per-image CSV report.
    csv_path = OUTPUT_DIR / "v4_candidate_evaluation.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image_id",
            "file_name",
            "ground_truth_count",
            "predicted_candidate_count",
            "true_positives",
            "false_positives",
            "false_negatives"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in image_results:

            writer.writerow({
                key: result[key]
                for key in fieldnames
            })

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("V4 CANDIDATE EVALUATION RESULTS")
    print("=" * 55)

    print("Test images:", len(image_records))
    print("Ground-truth annotations:", total_annotations)
    print("Predicted candidates:", total_predictions)

    print("\nTrue positives:", total_tp)
    print("False positives:", total_fp)
    print("False negatives:", total_fn)

    print("\nPrecision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1_score, 4))
    print("Mean matched IoU:", round(mean_matched_iou, 4))

    print("\nReports saved:")
    print(report_path)
    print(csv_path)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
