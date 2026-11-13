
from pathlib import Path
from ultralytics import YOLO
import torch

# ==========================================
# ARTERYAI - YOLO STENOSIS DETECTION V2
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_YAML = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "data.yaml"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_training_v2"
)

MODEL_NAME = "yolov8n.pt"

EPOCHS = 60
IMAGE_SIZE = 512
BATCH_SIZE = 4
PATIENCE = 15
WORKERS = 0
DEVICE = "cpu"


def main():

    print("=" * 60)
    print("ARTERYAI - YOLO STENOSIS DETECTION V2")
    print("=" * 60)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    print("Dataset:", DATASET_YAML)
    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)

    print("\nLoading pretrained YOLOv8n model...")
    model = YOLO(MODEL_NAME)

    print("\nStarting Experiment 2 training...")

    model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        patience=PATIENCE,
        pretrained=True,

        # Reduced augmentation
        mosaic=0.5,
        close_mosaic=10,
        fliplr=0.0,
        flipud=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        degrees=0.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,

        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,

        auto_augment=None,
        erasing=0.0,

        plots=True,
        save=True,
        verbose=True,

        project=str(OUTPUT_DIR),
        name="yolov8n_stenosis_v2",
        exist_ok=True
    )

    print("\nTraining completed.")

    best_model_path = (
        OUTPUT_DIR
        / "yolov8n_stenosis_v2"
        / "weights"
        / "best.pt"
    )

    print("Best model:", best_model_path)
    print("Model exists:", best_model_path.exists())

    if not best_model_path.exists():
        raise FileNotFoundError("Best model was not generated.")

    print("\nRunning validation evaluation...")

    best_model = YOLO(str(best_model_path))

    metrics = best_model.val(
        data=str(DATASET_YAML),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        project=str(OUTPUT_DIR),
        name="v2_validation",
        exist_ok=True
    )

    print("\n========== V2 VALIDATION RESULTS ==========")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\nExperiment 2 completed successfully.")


if __name__ == "__main__":
    main()
