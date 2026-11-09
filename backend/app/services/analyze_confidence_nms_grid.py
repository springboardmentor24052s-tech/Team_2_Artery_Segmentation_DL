
from pathlib import Path
from ultralytics import YOLO
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = BACKEND_ROOT / "results" / "stenosis_yolo_baseline" / "best.pt"
DATASET_ROOT = BACKEND_ROOT / "results" / "stenosis_yolo_dataset"

IMAGE_DIR = DATASET_ROOT / "images" / "val"
LABEL_DIR = DATASET_ROOT / "labels" / "val"

OUTPUT_DIR = BACKEND_ROOT / "results" / "nms_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
MATCH_IOU = 0.50
CONFIDENCE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
NMS_THRESHOLDS = [0.30, 0.40, 0.50]


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

    for iou, gt_index, pred_index in candidates:
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )

    if not image_paths:
        raise RuntimeError("No validation images found.")

    ground_truth = {}

    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        ground_truth[image_path.name] = load_ground_truth(label_path)

    report = {}

    total_combinations = (
        len(CONFIDENCE_THRESHOLDS) * len(NMS_THRESHOLDS)
    )

    combination_number = 0

    for confidence in CONFIDENCE_THRESHOLDS:
        for nms_iou in NMS_THRESHOLDS:

            combination_number += 1

            print(
                f"\n[{combination_number}/{total_combinations}] "
                f"Confidence={confidence}, NMS IoU={nms_iou}"
            )

            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_predictions = 0

            for image_path in image_paths:

                result = model.predict(
                    source=str(image_path),
                    imgsz=IMAGE_SIZE,
                    conf=confidence,
                    iou=nms_iou,
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

                tp, fp, fn = match_predictions(
                    ground_truth[image_path.name],
                    pred_boxes
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_predictions += len(pred_boxes)

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

            key = f"conf_{confidence}_nms_{nms_iou}"

            report[key] = {
                "confidence": confidence,
                "nms_iou": nms_iou,
                "images": len(image_paths),
                "predictions": total_predictions,
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }

            print(
                f"TP={total_tp}, FP={total_fp}, FN={total_fn}"
            )
            print(
                f"Precision={precision:.4f}, "
                f"Recall={recall:.4f}, F1={f1:.4f}"
            )

    best_key = max(
        report,
        key=lambda key: report[key]["f1_score"]
    )

    report["best_combination"] = {
        "configuration": best_key,
        **report[best_key]
    }

    report_path = OUTPUT_DIR / "confidence_nms_grid_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 55)
    print("BEST VALIDATION CONFIGURATION")
    print("=" * 55)
    print(json.dumps(report["best_combination"], indent=4))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
