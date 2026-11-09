
from pathlib import Path
import random
import math

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

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

IMAGE_DIR = DATASET_ROOT / "images" / "val"
LABEL_DIR = DATASET_ROOT / "labels" / "val"

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "nms_analysis"
    / "visual_comparisons"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 512
CONFIDENCE = 0.05
NMS_THRESHOLDS = [0.30, 0.50, 0.70]

# Number of images to visualize
NUMBER_OF_IMAGES = 12

# Fixed seed for reproducible image selection
RANDOM_SEED = 42

# --------------------------------------------------
# 3. LOAD GROUND-TRUTH LABELS
# --------------------------------------------------

def load_ground_truth(label_path):

    boxes = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, cx, cy, width, height = map(
                float, values
            )

            if int(class_id) != 0:
                continue

            x1 = (cx - width / 2) * IMAGE_SIZE
            y1 = (cy - height / 2) * IMAGE_SIZE
            x2 = (cx + width / 2) * IMAGE_SIZE
            y2 = (cy + height / 2) * IMAGE_SIZE

            boxes.append([x1, y1, x2, y2])

    return boxes


# --------------------------------------------------
# 4. DRAW GROUND-TRUTH AND PREDICTIONS
# --------------------------------------------------

def draw_boxes(image, ground_truth, predictions, threshold):

    output = image.copy()

    # Ground-truth boxes: GREEN
    for box in ground_truth:

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            "GT",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # Prediction boxes: RED
    for prediction in predictions:

        x1, y1, x2, y2 = map(int, prediction["box"])
        confidence = prediction["confidence"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            output,
            f"Pred {confidence:.2f}",
            (x1, min(output.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

    title = (
        f"NMS IoU: {threshold:.2f} | "
        f"Predictions: {len(predictions)}"
    )

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 32),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        output,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    return output


# --------------------------------------------------
# 5. RUN PREDICTIONS FOR EACH NMS THRESHOLD
# --------------------------------------------------

def get_predictions(model, image_path, nms_iou):

    result = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=nms_iou,
        device="cpu",
        max_det=300,
        verbose=False
    )[0]

    predictions = []

    for box in result.boxes:

        coordinates = box.xyxy[0].cpu().tolist()
        confidence = float(box.conf[0].cpu())

        predictions.append({
            "box": coordinates,
            "confidence": confidence
        })

    return predictions


# --------------------------------------------------
# 6. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Loading V1 YOLO model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation images not found: {IMAGE_DIR}"
        )

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in [
            ".png", ".jpg", ".jpeg"
        ]
    ])

    if not image_paths:
        raise RuntimeError("No validation images found.")

    # Select reproducible sample
    random.seed(RANDOM_SEED)

    selected_images = random.sample(
        image_paths,
        min(NUMBER_OF_IMAGES, len(image_paths))
    )

    print(f"Selected images: {len(selected_images)}")

    for index, image_path in enumerate(selected_images, start=1):

        print(
            f"\nProcessing {index}/{len(selected_images)}: "
            f"{image_path.name}"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            print("Skipping unreadable image.")
            continue

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        ground_truth = load_ground_truth(label_path)

        comparison_images = []

        for threshold in NMS_THRESHOLDS:

            predictions = get_predictions(
                model,
                image_path,
                threshold
            )

            comparison = draw_boxes(
                image,
                ground_truth,
                predictions,
                threshold
            )

            comparison_images.append(comparison)

            print(
                f"NMS={threshold:.2f}: "
                f"{len(predictions)} predictions"
            )

        # Combine three panels horizontally
        final_image = cv2.hconcat(comparison_images)

        output_path = (
            OUTPUT_DIR
            / f"{index:02d}_{image_path.stem}_nms_comparison.png"
        )

        cv2.imwrite(
            str(output_path),
            final_image
        )

        print(f"Saved: {output_path}")

    print("\nNMS visual comparison completed.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
