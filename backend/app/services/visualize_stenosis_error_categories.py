
import json
import random
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
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
    / "error_visual_gallery"
)

CONFIDENCE = 0.10
NMS_IOU = 0.30
MATCH_IOU = 0.50
IMAGE_SIZE = 512
MAX_DETECTIONS = 300

SAMPLES_PER_CATEGORY = 20
RANDOM_SEED = 42


# ============================================================
# UTILITIES
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def load_ground_truth(label_path, width, height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            _, xc, yc, w, h = map(float, values)

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            boxes.append([x1, y1, x2, y2])

    return boxes


def match_boxes(ground_truth, predictions):

    possible_matches = []

    for gi, gt in enumerate(ground_truth):

        for pi, pred in enumerate(predictions):

            iou = calculate_iou(gt, pred["box"])

            if iou >= MATCH_IOU:
                possible_matches.append((iou, gi, pi))

    possible_matches.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gi, pi in possible_matches:

        if gi in matched_gt or pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)

        matches.append((gi, pi, iou))

    return matched_gt, matched_pred, matches


# ============================================================
# DRAWING
# ============================================================

def draw_box(image, box, color, label, thickness=2):

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    text_y = max(18, y1 - 7)

    cv2.putText(
        image,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA
    )


def save_visualization(
    image,
    image_name,
    ground_truth,
    predictions,
    matched_gt,
    matched_pred,
    category,
    output_folder
):

    canvas = image.copy()

    # Ground-truth boxes
    for index, box in enumerate(ground_truth):

        if index in matched_gt:

            draw_box(
                canvas,
                box,
                (0, 255, 0),
                f"GT MATCHED {index}"
            )

        else:

            draw_box(
                canvas,
                box,
                (0, 255, 255),
                f"GT MISSED {index}"
            )

    # Prediction boxes
    for index, prediction in enumerate(predictions):

        box = prediction["box"]
        confidence = prediction["confidence"]

        if index in matched_pred:

            draw_box(
                canvas,
                box,
                (255, 0, 0),
                f"PRED MATCHED {confidence:.2f}"
            )

        else:

            draw_box(
                canvas,
                box,
                (0, 0, 255),
                f"FALSE POS {confidence:.2f}"
            )

    # Add information panel
    panel_height = 65

    panel = canvas.copy()

    cv2.rectangle(
        panel,
        (0, 0),
        (canvas.shape[1], panel_height),
        (20, 20, 20),
        -1
    )

    canvas = cv2.addWeighted(
        panel,
        0.75,
        canvas,
        0.25,
        0
    )

    cv2.putText(
        canvas,
        f"Image: {image_name}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        canvas,
        f"Category: {category}",
        (10, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        canvas,
        f"GT: {len(ground_truth)} | Pred: {len(predictions)} | "
        f"Matched: {len(matched_gt)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / image_name

    cv2.imwrite(str(output_path), canvas)

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        list(TEST_IMAGES.glob("*.png"))
        + list(TEST_IMAGES.glob("*.jpg"))
        + list(TEST_IMAGES.glob("*.jpeg"))
    )

    if not image_paths:
        raise RuntimeError("No test images found.")

    categories = defaultdict(list)

    print(f"Test images: {len(image_paths)}")
    print("Generating predictions and categorizing images...")

    for index, image_path in enumerate(image_paths, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = TEST_LABELS / f"{image_path.stem}.txt"

        ground_truth = load_ground_truth(
            label_path,
            width,
            height
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

                predictions.append({
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf[0].cpu().item())
                })

        matched_gt, matched_pred, matches = match_boxes(
            ground_truth,
            predictions
        )

        missed_indices = [
            i for i in range(len(ground_truth))
            if i not in matched_gt
        ]

        unmatched_pred_indices = [
            i for i in range(len(predictions))
            if i not in matched_pred
        ]

        # Category 1: missed regions
        if missed_indices:

            best_overlap = max(
                (
                    calculate_iou(ground_truth[gi], pred["box"])
                    for gi in missed_indices
                    for pred in predictions
                ),
                default=0.0
            )

            if best_overlap == 0:
                categories["missed_no_overlap"].append(image_path)

            elif best_overlap < 0.25:
                categories["missed_very_low_overlap"].append(image_path)

            else:
                categories["missed_partial_overlap"].append(image_path)

        # Category 2: false positives
        if unmatched_pred_indices:

            best_overlap = max(
                (
                    calculate_iou(
                        predictions[pi]["box"],
                        gt
                    )
                    for pi in unmatched_pred_indices
                    for gt in ground_truth
                ),
                default=0.0
            )

            if best_overlap == 0:
                categories["false_positive_no_overlap"].append(image_path)

            elif best_overlap < 0.25:
                categories["false_positive_low_overlap"].append(image_path)

            else:
                categories["false_positive_partial_overlap"].append(image_path)

        # Category 3: localization failures
        if any(
            0 < calculate_iou(ground_truth[gi], pred["box"]) < MATCH_IOU
            for gi in missed_indices
            for pred in predictions
        ):
            categories["localization_failure"].append(image_path)

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_paths)}")

        # Cache image-level details for final visualization
        image_details[image_path.name] = {
            "image": image,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "matched_gt": matched_gt,
            "matched_pred": matched_pred
        }

    # --------------------------------------------------------
    # SELECT AND SAVE REPRESENTATIVE IMAGES
    # --------------------------------------------------------

    manifest = {}

    for category, paths in categories.items():

        unique_paths = list(dict.fromkeys(paths))

        random.shuffle(unique_paths)

        selected = unique_paths[:SAMPLES_PER_CATEGORY]

        category_dir = OUTPUT_DIR / category

        category_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        for image_path in selected:

            details = image_details[image_path.name]

            output_path = save_visualization(
                details["image"],
                image_path.name,
                details["ground_truth"],
                details["predictions"],
                details["matched_gt"],
                details["matched_pred"],
                category,
                category_dir
            )

            saved.append(str(output_path))

        manifest[category] = {
            "available_images": len(unique_paths),
            "saved_images": len(saved),
            "files": saved
        }

    manifest_path = OUTPUT_DIR / "visual_gallery_manifest.json"

    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=4)

    print("\nVISUAL GALLERY GENERATED")
    print("=" * 50)

    for category, details in manifest.items():

        print(
            f"{category}: "
            f"{details['saved_images']} images saved "
            f"(available: {details['available_images']})"
        )

    print("\nOutput folder:")
    print(OUTPUT_DIR)


# Cache for image-level data
image_details = {}


if __name__ == "__main__":
    main()
