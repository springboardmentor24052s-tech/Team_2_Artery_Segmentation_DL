
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_baseline"
    / "best.pt"
)

IMAGE_DIR = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "images"
    / "val"
)

LABEL_DIR = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "labels"
    / "val"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "stenosis_confidence_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.70

CONFIDENCE_THRESHOLDS = [
    0.05, 0.10, 0.15, 0.25, 0.40, 0.50
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

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

    if union <= 0:
        return 0.0

    return intersection / union


def match_predictions(ground_truth, predictions):

    matched_gt = set()

    true_positives = 0

    sorted_predictions = sorted(
        predictions,
        key=lambda item: item["confidence"],
        reverse=True
    )

    for prediction in sorted_predictions:

        best_iou = 0.0
        best_index = None

        for index, gt in enumerate(ground_truth):

            if index in matched_gt:
                continue

            iou = calculate_iou(
                prediction["box"],
                gt
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if (
            best_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matched_gt.add(best_index)
            true_positives += 1

    false_positives = len(predictions) - true_positives

    false_negatives = len(ground_truth) - true_positives

    return true_positives, false_positives, false_negatives


# ==================================================
# 3. VERIFY PATHS
# ==================================================

for name, path in {
    "Model": MODEL_PATH,
    "Images": IMAGE_DIR,
    "Labels": LABEL_DIR
}.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

# ==================================================
# 4. LOAD MODEL
# ==================================================

print("\nLoading YOLO V1 baseline model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")

# ==================================================
# 5. LOAD VALIDATION DATA
# ==================================================

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise RuntimeError("No validation images found.")

# Cache all ground-truth boxes and predictions.
# Predict once at the lowest threshold, then reuse
# predictions for all higher thresholds.

validation_data = []

print("\nGenerating predictions...")

for image_index, image_path in enumerate(image_paths, start=1):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label: {label_path}"
        )

    ground_truth = []

    for line in label_path.read_text().splitlines():

        values = line.strip().split()

        if not values:
            continue

        if len(values) != 5:
            raise ValueError(
                f"Invalid label: {label_path}"
            )

        class_id, x, y, w, h = map(float, values)

        if class_id != 0:
            raise ValueError(
                f"Unexpected class ID: {class_id}"
            )

        x1 = (x - w / 2) * IMAGE_SIZE
        y1 = (y - h / 2) * IMAGE_SIZE
        x2 = (x + w / 2) * IMAGE_SIZE
        y2 = (y + h / 2) * IMAGE_SIZE

        ground_truth.append([x1, y1, x2, y2])

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=min(CONFIDENCE_THRESHOLDS),
        iou=NMS_IOU_THRESHOLD,
        device="cpu",
        max_det=300,
        verbose=False
    )

    predictions = []

    if results and results[0].boxes is not None:

        for box in results[0].boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            confidence = float(
                box.conf[0].cpu().item()
            )

            predictions.append({
                "box": coordinates,
                "confidence": confidence
            })

    validation_data.append({
        "image": image_path.name,
        "ground_truth": ground_truth,
        "predictions": predictions
    })

    if image_index % 25 == 0:
        print(
            f"Processed {image_index}/{len(image_paths)} images"
        )

# ==================================================
# 6. EVALUATE EACH CONFIDENCE THRESHOLD
# ==================================================

all_ground_truth = sum(
    len(item["ground_truth"])
    for item in validation_data
)

report_lines = []

report_lines.append(
    "YOLO V1 CONFIDENCE THRESHOLD ANALYSIS"
)

report_lines.append("=" * 60)

report_lines.append(
    f"Validation images: {len(validation_data)}"
)

report_lines.append(
    f"Ground-truth boxes: {all_ground_truth}"
)

report_lines.append(
    f"IoU threshold: {IOU_THRESHOLD}"
)

report_lines.append("")

print("\n" + "=" * 70)
print("CONFIDENCE THRESHOLD COMPARISON")
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'TP':<10}"
    f"{'FP':<10}"
    f"{'FN':<10}"
    f"{'Recall':<12}"
    f"{'Precision':<12}"
    f"{'F1':<10}"
)

for threshold in CONFIDENCE_THRESHOLDS:

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for item in validation_data:

        ground_truth = item["ground_truth"]

        predictions = [
            prediction
            for prediction in item["predictions"]
            if prediction["confidence"] >= threshold
        ]

        tp, fp, fn = match_predictions(
            ground_truth,
            predictions
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    print(
        f"{threshold:<12.2f}"
        f"{total_tp:<10}"
        f"{total_fp:<10}"
        f"{total_fn:<10}"
        f"{recall * 100:<12.2f}"
        f"{precision * 100:<12.2f}"
        f"{f1 * 100:<10.2f}"
    )

    report_lines.append(
        f"\nConfidence threshold: {threshold:.2f}"
    )

    report_lines.append(f"TP: {total_tp}")
    report_lines.append(f"FP: {total_fp}")
    report_lines.append(f"FN: {total_fn}")
    report_lines.append(f"Precision: {precision:.4f}")
    report_lines.append(f"Recall: {recall:.4f}")
    report_lines.append(f"F1: {f1:.4f}")

# ==================================================
# 7. SAVE REPORT
# ==================================================

report_path = (
    OUTPUT_DIR
    / "v1_confidence_threshold_analysis.txt"
)

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS COMPLETED")
print("=" * 70)

print("Report saved at:")
print(report_path)
