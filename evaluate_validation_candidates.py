
import json
from pathlib import Path

import numpy as np

from app.services.segmentation import predict_stenosis
from app.services.candidate_extraction import extract_stenosis_candidates


PROJECT_ROOT = Path(__file__).resolve().parent

VAL_DIR = (
    PROJECT_ROOT / "backend/app/data/arcade/arcade/arcade/stenosis/val"
)

ANNOTATION_FILE = VAL_DIR / "annotations/val.json"

OUTPUT_DIR = (
    PROJECT_ROOT / "backend/results/stenosis_candidate_evaluation"
)

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]
MIN_AREA = 10
IOU_THRESHOLD = 0.50


def calculate_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection

    return intersection / union if union > 0 else 0.0


def match_candidates(predictions, ground_truth):
    possible = []

    for pi, pred in enumerate(predictions):
        b = pred["bounding_box"]
        pred_box = [b["x"], b["y"], b["width"], b["height"]]

        for gi, gt in enumerate(ground_truth):
            iou = calculate_iou(pred_box, gt)

            if iou >= IOU_THRESHOLD:
                possible.append((iou, pi, gi))

    possible.sort(reverse=True)

    used_predictions = set()
    used_ground_truth = set()
    matched_ious = []

    for iou, pi, gi in possible:
        if pi in used_predictions or gi in used_ground_truth:
            continue

        used_predictions.add(pi)
        used_ground_truth.add(gi)
        matched_ious.append(iou)

    tp = len(matched_ious)

    return {
        "tp": tp,
        "fp": len(predictions) - tp,
        "fn": len(ground_truth) - tp,
        "matched_ious": matched_ious
    }


def main():
    with open(ANNOTATION_FILE, encoding="utf-8") as f:
        data = json.load(f)

    categories = {
        c["id"]: c["name"] for c in data["categories"]
    }

    stenosis_ids = {
        cid for cid, name in categories.items()
        if name.lower() == "stenosis"
    }

    image_records = {
        image["id"]: image for image in data["images"]
    }

    ground_truth = {
        image_id: [] for image_id in image_records
    }

    for ann in data["annotations"]:
        if ann["category_id"] in stenosis_ids:
            ground_truth[ann["image_id"]].append(
                [float(v) for v in ann["bbox"]]
            )

    totals = {
        t: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "predictions": 0,
            "annotations": 0,
            "matched_ious": []
        }
        for t in THRESHOLDS
    }

    images = sorted(image_records.items())

    print("Validation candidate evaluation")
    print("Images:", len(images))
    print("Thresholds:", THRESHOLDS)

    for index, (image_id, info) in enumerate(images, 1):
        image_path = VAL_DIR / "images" / info["file_name"]

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        prediction = predict_stenosis(image_path)
        probability = prediction["probability_map"]

        gt_boxes = ground_truth[image_id]

        for threshold in THRESHOLDS:
            binary_mask = (
                probability >= threshold
            ).astype(np.uint8) * 255

            extraction = extract_stenosis_candidates(
                probability_map=probability,
                binary_mask=binary_mask,
                min_area=MIN_AREA
            )

            candidates = extraction["candidates"]

            result = match_candidates(candidates, gt_boxes)

            current = totals[threshold]
            current["tp"] += result["tp"]
            current["fp"] += result["fp"]
            current["fn"] += result["fn"]
            current["predictions"] += len(candidates)
            current["annotations"] += len(gt_boxes)
            current["matched_ious"].extend(
                result["matched_ious"]
            )

        if index % 25 == 0:
            print(f"Processed {index}/{len(images)}")

    report = {}

    print("\nVALIDATION CANDIDATE RESULTS")
    print("-" * 100)
    print(
        f"{'Threshold':<12}{'Candidates':<14}"
        f"{'TP':<8}{'FP':<10}{'FN':<10}"
        f"{'Precision':<13}{'Recall':<12}"
        f"{'F1':<12}{'Mean IoU':<12}"
    )

    for threshold, values in totals.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        mean_iou = (
            float(np.mean(values["matched_ious"]))
            if values["matched_ious"] else 0
        )

        report[str(threshold)] = {
            "predicted_candidates": values["predictions"],
            "ground_truth_annotations": values["annotations"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mean_matched_iou": round(mean_iou, 4)
        }

        print(
            f"{threshold:<12.2f}"
            f"{values['predictions']:<14}"
            f"{tp:<8}{fp:<10}{fn:<10}"
            f"{precision:<13.4f}{recall:<12.4f}"
            f"{f1:<12.4f}{mean_iou:<12.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "validation_candidate_comparison.json"

    output_file.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\nReport saved:", output_file)


if __name__ == "__main__":
    main()
