
from pathlib import Path
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
    / "stenosis_size_confidence_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.70

CONFIDENCE_THRESHOLDS = [0.10, 0.15]

SMALL_AREA = 32 ** 2
MEDIUM_AREA = 96 ** 2

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

    return intersection / union if union > 0 else 0.0


def get_size_category(area):

    if area < SMALL_AREA:
        return "small"

    elif area < MEDIUM_AREA:
        return "medium"

    return "large"


def match_predictions(ground_truth, predictions):

    matched_gt = set()

    results = {
        "small": {"tp": 0, "fn": 0},
        "medium": {"tp": 0, "fn": 0},
        "large": {"tp": 0, "fn": 0}
    }

    tp = 0
    fp = 0

    predictions = sorted(
        predictions,
        key=lambda item: item["confidence"],
        reverse=True
    )

    for prediction in predictions:

        best_iou = 0.0
        best_index = None

        for index, gt in enumerate(ground_truth):

            if index in matched_gt:
                continue

            iou = calculate_iou(
                prediction["box"],
                gt["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if (
            best_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matched_gt.add(best_index)

            category = ground_truth[best_index]["category"]

            results[category]["tp"] += 1
            tp += 1

        else:
            fp += 1

    for index, gt in enumerate(ground_truth):

        if index not in matched_gt:

            results[gt["category"]]["fn"] += 1

    return results, tp, fp, len(ground_truth) - tp


# ==================================================
# 3. VALIDATE PATHS
# ==================================================

for name, path in {
    "Model": MODEL_PATH,
    "Validation images": IMAGE_DIR,
    "Validation labels": LABEL_DIR
}.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

print("\nAll paths verified.")

# ==================================================
# 4. LOAD MODEL
# ==================================================

print("\nLoading YOLO V1 baseline...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")

# ==================================================
# 5. LOAD VALIDATION DATA AND PREDICT
# ==================================================

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

validation_data = []

print("\nGenerating validation predictions...")

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

        area = (
            max(0, x2 - x1) *
            max(0, y2 - y1)
        )

        ground_truth.append({
            "box": [x1, y1, x2, y2],
            "category": get_size_category(area)
        })

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

            predictions.append({
                "box": (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                ),
                "confidence": float(
                    box.conf[0].cpu().item()
                )
            })

    validation_data.append({
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

report = []

for threshold in CONFIDENCE_THRESHOLDS:

    total_stats = {
        "small": {"tp": 0, "fn": 0},
        "medium": {"tp": 0, "fn": 0},
        "large": {"tp": 0, "fn": 0}
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for item in validation_data:

        filtered_predictions = [
            prediction
            for prediction in item["predictions"]
            if prediction["confidence"] >= threshold
        ]

        size_results, tp, fp, fn = match_predictions(
            item["ground_truth"],
            filtered_predictions
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for category in ["small", "medium", "large"]:

            total_stats[category]["tp"] += (
                size_results[category]["tp"]
            )

            total_stats[category]["fn"] += (
                size_results[category]["fn"]
            )

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

    print("\n" + "=" * 60)
    print(f"CONFIDENCE THRESHOLD: {threshold:.2f}")
    print("=" * 60)

    print(f"Overall precision: {precision * 100:.2f}%")
    print(f"Overall recall: {recall * 100:.2f}%")
    print(f"Overall F1: {f1 * 100:.2f}%")
    print("False positives:", total_fp)

    report.append(f"\nCONFIDENCE: {threshold:.2f}")
    report.append(f"Precision: {precision * 100:.2f}%")
    report.append(f"Recall: {recall * 100:.2f}%")
    report.append(f"F1: {f1 * 100:.2f}%")
    report.append(f"False positives: {total_fp}")

    for category in ["small", "medium", "large"]:

        tp = total_stats[category]["tp"]
        fn = total_stats[category]["fn"]

        gt = tp + fn

        size_recall = (
            tp / gt * 100
            if gt else 0
        )

        print(f"\n{category.upper()} BOXES")
        print("Ground truth:", gt)
        print("Detected:", tp)
        print("Missed:", fn)
        print(f"Recall: {size_recall:.2f}%")

        report.append(f"\n{category.upper()}")
        report.append(f"Ground truth: {gt}")
        report.append(f"Detected: {tp}")
        report.append(f"Missed: {fn}")
        report.append(f"Recall: {size_recall:.2f}%")

# ==================================================
# 7. SAVE REPORT
# ==================================================

report_path = (
    OUTPUT_DIR
    / "v1_size_by_confidence_report.txt"
)

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("SIZE-WISE CONFIDENCE ANALYSIS COMPLETED")
print("=" * 60)

print("Report saved at:")
print(report_path)
