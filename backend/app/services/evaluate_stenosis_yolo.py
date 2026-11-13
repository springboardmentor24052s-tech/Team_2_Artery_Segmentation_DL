
from pathlib import Path
from ultralytics import YOLO
import json

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_training"
    / "yolov8n_stenosis"
    / "weights"
    / "best.pt"
)

DATASET_YAML = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "data.yaml"
)

OUTPUT_DIR = BACKEND_ROOT / "results" / "stenosis_yolo_test_evaluation"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_YAML}")

    print("Loading trained YOLOv8n model...")
    model = YOLO(str(MODEL_PATH))

    print("\nEvaluating on untouched TEST dataset...")

    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=512,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name="test_results",
        exist_ok=True
    )

    results = {
        "model": "YOLOv8n",
        "evaluation_split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / "test_metrics.json"

    with open(report_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\n========== TEST RESULTS ==========")
    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nEvaluation report saved at:")
    print(report_path)


if __name__ == "__main__":
    main()
