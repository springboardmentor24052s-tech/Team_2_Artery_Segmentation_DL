
from pathlib import Path
from ultralytics import YOLO
import json

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
    BACKEND_ROOT / "results" / "nms_analysis" / "final_test_evaluation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.10
NMS_IOU = 0.30
IMAGE_SIZE = 512
MATCH_IOU = 0.50


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

    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError("Test image or label directory not found.")

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )

    if not image_paths:
        raise RuntimeError("No test images found.")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predictions = 0
    total_ground_truth = 0

    print(f"Test images: {len(image_paths)}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"NMS IoU: {NMS_IOU}")

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

        tp, fp, fn = match_predictions(gt_boxes, pred_boxes)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_predictions += len(pred_boxes)
        total_ground_truth += len(gt_boxes)

        if index % 25 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

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

    report = {
        "model": str(MODEL_PATH),
        "dataset_split": "test",
        "images": len(image_paths),
        "confidence_threshold": CONFIDENCE,
        "nms_iou_threshold": NMS_IOU,
        "matching_iou_threshold": MATCH_IOU,
        "ground_truth_boxes": total_ground_truth,
        "predictions": total_predictions,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

    report_path = OUTPUT_DIR / "best_config_test_metrics.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nFINAL TEST RESULTS")
    print("=" * 45)
    print(json.dumps(report, indent=4))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
