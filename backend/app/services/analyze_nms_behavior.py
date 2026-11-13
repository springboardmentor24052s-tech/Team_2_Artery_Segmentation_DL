
from pathlib import Path
from ultralytics import YOLO
import json

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = BACKEND_ROOT / "results" / "stenosis_yolo_baseline" / "best.pt"

IMAGE_DIR = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "images"
    / "val"
)

OUTPUT_DIR = BACKEND_ROOT / "results" / "nms_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis settings
IMAGE_SIZE = 512
CONFIDENCE = 0.05

# Test different NMS IoU thresholds
NMS_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

def main():
    print("Loading V1 YOLO model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Validation image folder not found: {IMAGE_DIR}")

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(IMAGE_DIR.glob("*"))

    image_paths = [
        path for path in image_paths
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    print(f"Validation images found: {len(image_paths)}")

    results_summary = {}

    for nms_iou in NMS_THRESHOLDS:
        print(f"\nRunning NMS IoU = {nms_iou}")

        total_predictions = 0
        images_with_predictions = 0

        for image_path in image_paths:
            results = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=nms_iou,
                device="cpu",
                max_det=300,
                verbose=False
            )

            result = results[0]
            prediction_count = len(result.boxes)

            total_predictions += prediction_count

            if prediction_count > 0:
                images_with_predictions += 1

        results_summary[str(nms_iou)] = {
            "nms_iou": nms_iou,
            "confidence": CONFIDENCE,
            "total_images": len(image_paths),
            "total_predictions": total_predictions,
            "images_with_predictions": images_with_predictions,
            "average_predictions_per_image": (
                total_predictions / len(image_paths)
                if image_paths else 0
            )
        }

        print(f"Total predictions: {total_predictions}")
        print(f"Images with predictions: {images_with_predictions}")
        print(
            "Average predictions per image:",
            round(results_summary[str(nms_iou)][
                "average_predictions_per_image"
            ], 2)
        )

    report_path = OUTPUT_DIR / "nms_prediction_count_report.json"

    with open(report_path, "w") as file:
        json.dump(results_summary, file, indent=4)

    print("\nNMS analysis completed.")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
