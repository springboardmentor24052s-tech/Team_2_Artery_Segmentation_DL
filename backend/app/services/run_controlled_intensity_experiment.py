
import json
import shutil
from pathlib import Path

import yaml
import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT / "backend" / "results"
    / "stenosis_yolo_baseline" / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT / "backend" / "results"
    / "stenosis_yolo_dataset" / "data.yaml"
)

DATASET_ROOT = (
    PROJECT_ROOT / "backend" / "results"
    / "stenosis_yolo_dataset"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "backend" / "results"
    / "controlled_intensity_experiment"
)

QUANTIZED_ROOT = OUTPUT_ROOT / "quantized_dataset"

IMG_SIZE = 512
BATCH_SIZE = 4
DEVICE = "cpu"


def prepare_quantized_validation():
    source_images = DATASET_ROOT / "images" / "val"
    source_labels = DATASET_ROOT / "labels" / "val"

    target_images = QUANTIZED_ROOT / "images" / "val"
    target_labels = QUANTIZED_ROOT / "labels" / "val"

    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in source_images.iterdir()
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    print(f"\nPreparing quantized validation images: {len(image_paths)}")

    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.uint8)

        # Map each pixel to the nearest lower even intensity.
        quantized = (pixels // 2) * 2

        output_image = Image.fromarray(quantized.astype(np.uint8))
        output_image.save(target_images / image_path.name)

        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing validation label: {label_path}"
            )

        shutil.copy2(
            label_path,
            target_labels / label_path.name
        )

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    if len(list(target_images.iterdir())) != len(image_paths):
        raise RuntimeError("Quantized image count verification failed.")

    print("Quantized validation dataset prepared successfully.")

    return target_images


def create_quantized_yaml(quantized_images):
    with open(DATA_YAML, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    data["path"] = str(DATASET_ROOT.resolve())
    data["val"] = str(quantized_images.resolve())

    yaml_path = QUANTIZED_ROOT / "data_quantized.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    return yaml_path


def evaluate_model(model, yaml_path, experiment_name):
    print("\n" + "=" * 65)
    print(f"EVALUATING: {experiment_name}")
    print("=" * 65)

    results = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(OUTPUT_ROOT / "evaluation"),
        name=experiment_name,
        exist_ok=True,
        verbose=True
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map)
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    return metrics


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    print("Loading existing YOLO V1 model...")
    model = YOLO(str(MODEL_PATH))

    # Evaluate original validation data.
    original_metrics = evaluate_model(
        model,
        DATA_YAML,
        "original_validation"
    )

    # Prepare separate quantized validation data.
    quantized_images = prepare_quantized_validation()

    quantized_yaml = create_quantized_yaml(quantized_images)

    # Evaluate the same model on quantized validation data.
    quantized_metrics = evaluate_model(
        model,
        quantized_yaml,
        "quantized_validation"
    )

    comparison = {}

    for metric in original_metrics:
        original = original_metrics[metric]
        quantized = quantized_metrics[metric]

        comparison[metric] = {
            "original": original,
            "quantized": quantized,
            "difference": quantized - original
        }

    report = {
        "experiment": "Original vs 7-bit quantized validation",
        "model": str(MODEL_PATH),
        "quantization": "floor(pixel / 2) * 2",
        "original_metrics": original_metrics,
        "quantized_metrics": quantized_metrics,
        "comparison": comparison
    }

    report_path = OUTPUT_ROOT / "controlled_experiment_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for metric, values in comparison.items():
        print(
            f"{metric}: "
            f"Original={values['original']:.5f}, "
            f"Quantized={values['quantized']:.5f}, "
            f"Difference={values['difference']:+.5f}"
        )

    print(f"\nReport saved: {report_path}")
    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
