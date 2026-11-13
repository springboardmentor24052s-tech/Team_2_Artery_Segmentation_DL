
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

# Correct YOLO V1 baseline model path
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
    / "stenosis_size_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.70

# Bounding-box area categories
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

    if union <= 0:
        return 0.0

    return intersection / union


def get_size_category(area):

    if area < SMALL_AREA:
        return "small"

    elif area < MEDIUM_AREA:
        return "medium"

    return "large"


# ==================================================
# 3. VALIDATE PATHS
# ==================================================

required_paths = {
    "Model": MODEL_PATH,
    "Validation images": IMAGE_DIR,
    "Validation labels": LABEL_DIR
}

for name, path in required_paths.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

print("\nAll required paths verified.")

# ==================================================
# 4. LOAD YOLO V1 MODEL
# ==================================================

print("\nLoading YOLO V1 baseline model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")

# ==================================================
# 5. INITIALIZE STATISTICS
# ==================================================

stats = {
    "small": {
        "ground_truth": 0,
        "matched": 0
    },
    "medium": {
        "ground_truth": 0,
        "matched": 0
    },
    "large": {
        "ground_truth": 0,
        "matched": 0
    }
}

false_positives = 0
total_images = 0
images_without_annotations = 0

# ==================================================
# 6. GET VALIDATION IMAGES
# ==================================================

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise RuntimeError("No validation images found.")

print("\nStarting validation error analysis...")
print("Total validation images:", len(image_paths))

# ==================================================
# 7. PROCESS EACH IMAGE
# ==================================================

for image_path in image_paths:

    total_images += 1

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    ground_truth_boxes = []

    # ----------------------------------------------
    # Read ground-truth annotations
    # ----------------------------------------------

    for line in label_path.read_text().splitlines():

        values = line.strip().split()

        if not values:
            continue

        if len(values) != 5:
            raise ValueError(
                f"Invalid YOLO annotation in {label_path}: {line}"
            )

        class_id, x, y, w, h = map(float, values)

        if class_id != 0:
            raise ValueError(
                f"Unexpected class ID {class_id} in {label_path}"
            )

        # Convert normalized YOLO coordinates to pixels
        x1 = (x - w / 2) * IMAGE_SIZE
        y1 = (y - h / 2) * IMAGE_SIZE
        x2 = (x + w / 2) * IMAGE_SIZE
        y2 = (y + h / 2) * IMAGE_SIZE

        area = (
            max(0, x2 - x1) *
            max(0, y2 - y1)
        )

        category = get_size_category(area)

        ground_truth_boxes.append({
            "box": [x1, y1, x2, y2],
            "category": category,
            "matched": False
        })

        stats[category]["ground_truth"] += 1

    if not ground_truth_boxes:
        images_without_annotations += 1

    # ----------------------------------------------
    # Run YOLO prediction
    # ----------------------------------------------

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
        device="cpu",
        verbose=False
    )

    predicted_boxes = []

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

            predicted_boxes.append({
                "box": coordinates,
                "confidence": confidence
            })

    # ----------------------------------------------
    # Sort predictions by confidence
    # ----------------------------------------------

    predicted_boxes.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    # ----------------------------------------------
    # Greedy one-to-one IoU matching
    # ----------------------------------------------

    for prediction in predicted_boxes:

        best_iou = 0.0
        best_gt = None

        for gt in ground_truth_boxes:

            if gt["matched"]:
                continue

            iou_score = calculate_iou(
                prediction["box"],
                gt["box"]
            )

            if iou_score > best_iou:
                best_iou = iou_score
                best_gt = gt

        if (
            best_gt is not None
            and best_iou >= IOU_THRESHOLD
        ):

            best_gt["matched"] = True

            stats[
                best_gt["category"]
            ]["matched"] += 1

        else:
            false_positives += 1

    # ----------------------------------------------
    # Progress
    # ----------------------------------------------

    if total_images % 25 == 0:

        print(
            f"Processed {total_images}/"
            f"{len(image_paths)} images"
        )

# ==================================================
# 8. CALCULATE RESULTS
# ==================================================

total_gt = sum(
    stats[c]["ground_truth"]
    for c in stats
)

total_matched = sum(
    stats[c]["matched"]
    for c in stats
)

total_missed = total_gt - total_matched

overall_recall = (
    total_matched / total_gt * 100
    if total_gt > 0 else 0
)

# ==================================================
# 9. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 60)
print("YOLO V1 VALIDATION ERROR ANALYSIS BY BOX SIZE")
print("=" * 60)

print("\nModel:", MODEL_PATH)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("IoU threshold:", IOU_THRESHOLD)

print("\nTOTAL IMAGES:", total_images)

print(
    "Images without annotations:",
    images_without_annotations
)

print("\nSIZE-WISE DETECTION RESULTS")

for category in ["small", "medium", "large"]:

    gt_count = stats[category]["ground_truth"]
    matched = stats[category]["matched"]

    missed = gt_count - matched

    recall = (
        matched / gt_count * 100
        if gt_count > 0 else 0
    )

    print(f"\n{category.upper()} BOXES")

    print("Ground-truth boxes:", gt_count)
    print("Successfully detected:", matched)
    print("Missed boxes:", missed)
    print(f"Recall: {recall:.2f}%")

print("\nFALSE POSITIVES:", false_positives)

print("\nOVERALL RESULTS")

print("Total ground-truth boxes:", total_gt)
print("Total matched boxes:", total_matched)
print("Total missed boxes:", total_missed)

print(f"Overall recall: {overall_recall:.2f}%")

# ==================================================
# 10. SAVE REPORT
# ==================================================

report_path = (
    OUTPUT_DIR / "v1_size_error_analysis.txt"
)

with report_path.open("w", encoding="utf-8") as file:

    file.write(
        "YOLO V1 VALIDATION ERROR ANALYSIS\n"
    )

    file.write("=" * 55 + "\n")

    file.write(f"Model: {MODEL_PATH}\n")
    file.write(f"Images: {total_images}\n")
    file.write(
        f"Confidence threshold: {CONFIDENCE_THRESHOLD}\n"
    )
    file.write(f"IoU threshold: {IOU_THRESHOLD}\n")

    for category in ["small", "medium", "large"]:

        gt = stats[category]["ground_truth"]
        matched = stats[category]["matched"]
        missed = gt - matched

        recall = (
            matched / gt * 100
            if gt > 0 else 0
        )

        file.write(f"\n{category.upper()} BOXES\n")
        file.write(f"Ground truth: {gt}\n")
        file.write(f"Matched: {matched}\n")
        file.write(f"Missed: {missed}\n")
        file.write(f"Recall: {recall:.2f}%\n")

    file.write(f"\nFalse positives: {false_positives}\n")
    file.write(f"Total missed: {total_missed}\n")
    file.write(f"Overall recall: {overall_recall:.2f}%\n")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Report saved at:")
print(report_path)
