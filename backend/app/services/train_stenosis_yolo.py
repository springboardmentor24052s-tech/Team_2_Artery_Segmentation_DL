
from pathlib import Path
import torch
from ultralytics import YOLO

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_YAML = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "data.yaml"
)

OUTPUT_DIR = BACKEND_ROOT / "results" / "stenosis_yolo_training"

# --------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

EPOCHS = 50
IMAGE_SIZE = 512
BATCH_SIZE = 4
PATIENCE = 10
WORKERS = 0

DEVICE = "cpu"

# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("ARTERYAI - YOLO STENOSIS DETECTION TRAINING")
    print("=" * 60)

    print("YOLO Model:", MODEL_NAME)
    print("Dataset:", DATASET_YAML)
    print("Device:", DEVICE)
    print("CUDA Available:", torch.cuda.is_available())

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pretrained YOLOv8 Nano model
    model = YOLO(MODEL_NAME)

    print("\nStarting training...\n")

    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="yolov8n_stenosis",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        plots=True,
        save=True,
        verbose=True
    )

    print("\nTraining completed.")

    best_model = (
        OUTPUT_DIR
        / "yolov8n_stenosis"
        / "weights"
        / "best.pt"
    )

    print("Best model:", best_model)
    print("Model exists:", best_model.exists())

    if not best_model.exists():
        raise FileNotFoundError("Best model was not generated.")

    print("\nRunning validation evaluation...")

    trained_model = YOLO(str(best_model))

    metrics = trained_model.val(
        data=str(DATASET_YAML),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS
    )

    print("\nVALIDATION RESULTS")
    print("-" * 40)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\nTraining and validation completed successfully.")


if __name__ == "__main__":
    main()
