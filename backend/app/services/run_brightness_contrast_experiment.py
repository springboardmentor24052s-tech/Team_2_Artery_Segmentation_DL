
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BACKEND_ROOT = PROJECT_ROOT / "backend"

MODEL_PATH = (
    BACKEND_ROOT / "results"
    / "stenosis_yolo_baseline" / "best.pt"
)

DATASET_ROOT = (
    BACKEND_ROOT / "results"
    / "stenosis_yolo_dataset"
)

DATA_YAML = DATASET_ROOT / "data.yaml"

OUTPUT_ROOT = (
    BACKEND_ROOT / "results"
    / "brightness_contrast_experiment"
)

TRAIN_MEAN = 0.4777 * 255
TRAIN_STD = 0.15189 * 255

IMG_SIZE = 512
BATCH_SIZE = 4


def transform_image(image, condition):
    pixels = image.astype(np.float32)

    if condition == "brightness":
        current_mean = float(pixels.mean())
        transformed = pixels + (TRAIN_MEAN - current_mean)

    elif condition == "contrast":
        current_mean = float(pixels.mean())
        current_std = float(pixels.std())

        if current_std < 1e-6:
            transformed = pixels
        else:
            transformed = (
                (pixels - current_mean)
                * (TRAIN_STD / current_std)
                + current_mean
            )

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return np.clip(
        np.rint(transformed), 0, 255
    ).astype(np.uint8)


def prepare_dataset(condition):
    source_images = DATASET_ROOT / "images" / "val"
    source_labels = DATASET_ROOT / "labels" / "val"

    target_root = OUTPUT_ROOT / condition
    target_images = target_root / "images" / "val"
    target_labels = target_root / "labels" / "val"

    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(source_images.glob("*.png"))

    if len(image_paths) != 199:
        raise RuntimeError(
            f"Expected 199 validation images, found {len(image_paths)}"
        )

    print(f"\nPreparing {condition} dataset...")

    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(
            str(image_path), cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        transformed = transform_image(image, condition)

        output_path = target_images / image_path.name

        if not cv2.imwrite(str(output_path), transformed):
            raise RuntimeError(
                f"Cannot save image: {output_path}"
            )

        label_path = source_labels / (
            image_path.stem + ".txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        shutil.copy2(
            label_path,
            target_labels / label_path.name
        )

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    yaml_data = {
        "path": str(target_root.resolve()),
        "train": str(
            (DATASET_ROOT / "images" / "train").resolve()
        ),
        "val": "images/val",
        "test": str(
            (DATASET_ROOT / "images" / "test").resolve()
        ),
        "names": {0: "stenosis"},
        "nc": 1
    }

    yaml_path = target_root / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            yaml_data, file, sort_keys=False
        )

    print(f"{condition} dataset prepared.")

    return yaml_path


def evaluate(model, yaml_path, name):
    print("\n" + "=" * 65)
    print(f"EVALUATING: {name}")
    print("=" * 65)

    result = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=0,
        plots=True,
        project=str(OUTPUT_ROOT / "evaluations"),
        name=name,
        exist_ok=True,
        verbose=True
    )

    return {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map)
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for required in [MODEL_PATH, DATA_YAML]:
        if not required.exists():
            raise FileNotFoundError(required)

    model = YOLO(str(MODEL_PATH))

    results = {}

    results["original"] = evaluate(
        model, DATA_YAML, "original"
    )

    for condition in ["brightness", "contrast"]:
        yaml_path = prepare_dataset(condition)

        results[condition] = evaluate(
            model, yaml_path, condition
        )

    comparison = {}

    for condition in ["brightness", "contrast"]:
        comparison[condition] = {}

        for metric in results["original"]:
            baseline = results["original"][metric]
            changed = results[condition][metric]

            comparison[condition][metric] = {
                "original": baseline,
                "transformed": changed,
                "difference": changed - baseline
            }

    report = {
        "experiment": "Brightness and contrast normalization",
        "model": str(MODEL_PATH),
        "train_mean_reference": TRAIN_MEAN,
        "train_std_reference": TRAIN_STD,
        "results": results,
        "comparison": comparison
    }

    report_path = (
        OUTPUT_ROOT
        / "brightness_contrast_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for condition in ["brightness", "contrast"]:
        print(f"\n{condition.upper()}")

        for metric, values in comparison[condition].items():
            print(
                f"{metric}: "
                f"Original={values['original']:.5f}, "
                f"Transformed={values['transformed']:.5f}, "
                f"Difference={values['difference']:+.5f}"
            )

    print(f"\nReport saved: {report_path}")
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
