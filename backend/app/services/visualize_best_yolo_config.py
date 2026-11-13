
from pathlib import Path
import random
import cv2
from ultralytics import YOLO

# Project paths
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
    / "best_config_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best configuration from validation analysis
IMAGE_SIZE = 512
CONFIDENCE = 0.10
NMS_IOU = 0.30

NUMBER_OF_IMAGES = 20
RANDOM_SEED = 42


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


def draw_comparison(image, ground_truth, predictions):

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

    # Model predictions: RED
    for prediction in predictions:

        x1, y1, x2, y2 = map(
            int, prediction["box"]
        )

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

    # Header
    header = (
        f"Confidence: {CONFIDENCE} | "
        f"NMS IoU: {NMS_IOU} | "
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
        header,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    return output


def main():

    print("Loading V1 YOLO model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(IMAGE_DIR)

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in [
            ".png", ".jpg", ".jpeg"
        ]
    ])

    random.seed(RANDOM_SEED)

    selected_images = random.sample(
        image_paths,
        min(NUMBER_OF_IMAGES, len(image_paths))
    )

    print(f"Selected images: {len(selected_images)}")

    for index, image_path in enumerate(
        selected_images, start=1
    ):

        print(
            f"Processing {index}/{len(selected_images)}: "
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

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device="cpu",
            max_det=300,
            verbose=False
        )[0]

        predictions = []

        for box in result.boxes:

            predictions.append({
                "box": box.xyxy[0].cpu().tolist(),
                "confidence": float(box.conf[0].cpu())
            })

        final_image = draw_comparison(
            image,
            ground_truth,
            predictions
        )

        output_path = (
            OUTPUT_DIR
            / f"{index:02d}_{image_path.stem}_best_config.png"
        )

        cv2.imwrite(
            str(output_path),
            final_image
        )

        print(
            f"Ground truth: {len(ground_truth)}, "
            f"Predictions: {len(predictions)}"
        )

    print("\nVisualization completed.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
