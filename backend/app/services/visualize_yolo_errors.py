
from pathlib import Path
import cv2
from ultralytics import YOLO

# ==================================================
# 1. PATHS
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
    / "stenosis_visual_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
CONFIDENCE = 0.10
IOU_THRESHOLD = 0.50
NMS_IOU = 0.70
MAX_IMAGES = 20

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# ==================================================
# 2. IOU FUNCTION
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
        max(0, box2[3] - box1[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ==================================================
# 3. CHECK PATHS
# ==================================================

for name, path in {
    "Model": MODEL_PATH,
    "Images": IMAGE_DIR,
    "Labels": LABEL_DIR
}.items():

    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

# ==================================================
# 4. LOAD MODEL
# ==================================================

print("\nLoading YOLO V1 model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded.")

# ==================================================
# 5. SELECT VALIDATION IMAGES
# ==================================================

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise RuntimeError("No validation images found.")

# ==================================================
# 6. GENERATE VISUAL COMPARISONS
# ==================================================

saved_count = 0

for image_path in image_paths:

    if saved_count >= MAX_IMAGES:
        break

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label: {label_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path.name)
        continue

    ground_truth = []

    # Read ground-truth annotations
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

        x1 = int((x - w / 2) * IMAGE_SIZE)
        y1 = int((y - h / 2) * IMAGE_SIZE)
        x2 = int((x + w / 2) * IMAGE_SIZE)
        y2 = int((y + h / 2) * IMAGE_SIZE)

        ground_truth.append([x1, y1, x2, y2])

    # Run YOLO prediction
    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=NMS_IOU,
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

    # Sort predictions by confidence
    predictions.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    # Match predictions with ground truth
    matched_gt = set()
    matched_predictions = set()

    for pred_index, prediction in enumerate(predictions):

        best_iou = 0
        best_gt_index = None

        for gt_index, gt_box in enumerate(ground_truth):

            if gt_index in matched_gt:
                continue

            iou_score = calculate_iou(
                prediction["box"],
                gt_box
            )

            if iou_score > best_iou:
                best_iou = iou_score
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matched_gt.add(best_gt_index)
            matched_predictions.add(pred_index)

    # Select images containing useful errors
    missed_count = len(ground_truth) - len(matched_gt)

    false_positive_count = (
        len(predictions) - len(matched_predictions)
    )

    if missed_count == 0 and false_positive_count == 0:
        continue

    # Draw ground truth in green
    for index, gt_box in enumerate(ground_truth):

        x1, y1, x2, y2 = gt_box

        if index in matched_gt:
            color = (0, 255, 0)
            label = "GT - MATCHED"
        else:
            color = (0, 255, 255)
            label = "GT - MISSED"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    # Draw predictions in red
    for index, prediction in enumerate(predictions):

        x1, y1, x2, y2 = map(
            int,
            prediction["box"]
        )

        confidence = prediction["confidence"]

        if index in matched_predictions:
            color = (255, 0, 0)
            label = f"DETECTED {confidence:.2f}"
        else:
            color = (0, 0, 255)
            label = f"FALSE POSITIVE {confidence:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            image,
            label,
            (x1, min(image.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    # Add summary text
    cv2.putText(
        image,
        f"GT: {len(ground_truth)} | Pred: {len(predictions)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_comparison.png"
    )

    cv2.imwrite(str(output_path), image)

    saved_count += 1

    print(
        f"Saved {saved_count}: {output_path.name} "
        f"| Missed: {missed_count} "
        f"| False positives: {false_positive_count}"
    )

# ==================================================
# 7. FINAL STATUS
# ==================================================

print("\n" + "=" * 60)
print("VISUAL ERROR ANALYSIS COMPLETED")
print("=" * 60)

print("Images saved:", saved_count)
print("Output folder:", OUTPUT_DIR)
