
from pathlib import Path
import json
import csv

import cv2
import numpy as np


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_ROOT = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
)

IMAGE_SIZE = 512

SPLITS = ["train", "val", "test"]

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "yolo_preprocessing_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def find_images(split):

    image_dir = DATASET_ROOT / "images" / split

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    return sorted([
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp"
        }
    ])


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def preprocess_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None

    original_height, original_width = image.shape[:2]

    # Convert BGR to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to YOLO training image size
    resized = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    normalized = resized.astype(
        np.float32
    ) / 255.0

    return {
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": resized.shape[1],
        "processed_height": resized.shape[0],
        "brightness": float(np.mean(normalized)),
        "contrast_std": float(np.std(normalized)),
        "dynamic_range": float(
            np.percentile(normalized, 90)
            - np.percentile(normalized, 10)
        ),
        "unique_levels": int(
            len(np.unique(resized))
        ),
        "dark_pixels_pct": float(
            np.mean(normalized < 0.10) * 100
        ),
        "bright_pixels_pct": float(
            np.mean(normalized > 0.90) * 100
        )
    }


# ============================================================
# PROCESS DATASET
# ============================================================

def analyze_split(split):

    image_paths = find_images(split)

    records = []
    failed = 0

    print(f"\nProcessing {split.upper()}...")

    for index, path in enumerate(image_paths, start=1):

        result = preprocess_image(path)

        if result is None:
            failed += 1
            continue

        result["image"] = path.name
        result["split"] = split

        records.append(result)

        if index % 100 == 0:
            print(
                f"{split}: {index}/{len(image_paths)}"
            )

    return records, failed, len(image_paths)


# ============================================================
# SUMMARY
# ============================================================

def summarize(records):

    metrics = [
        "brightness",
        "contrast_std",
        "dynamic_range",
        "unique_levels",
        "dark_pixels_pct",
        "bright_pixels_pct"
    ]

    summary = {}

    for split in SPLITS:

        split_records = [
            record
            for record in records
            if record["split"] == split
        ]

        split_summary = {
            "image_count": len(split_records)
        }

        for metric in metrics:

            values = np.array(
                [record[metric] for record in split_records],
                dtype=float
            )

            if len(values) == 0:
                split_summary[metric] = None
                continue

            split_summary[metric] = {
                "mean": round(float(np.mean(values)), 5),
                "median": round(float(np.median(values)), 5),
                "std": round(float(np.std(values)), 5),
                "min": round(float(np.min(values)), 5),
                "max": round(float(np.max(values)), 5)
            }

        summary[split] = split_summary

    return summary


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    if not records:
        return

    output_path = (
        OUTPUT_DIR / "preprocessing_image_records.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nCSV saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("YOLO PREPROCESSING COMPARISON")
    print("=" * 65)

    print(f"\nDataset: {DATASET_ROOT}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")

    all_records = []
    failure_report = {}

    for split in SPLITS:

        records, failed, total = analyze_split(split)

        all_records.extend(records)

        failure_report[split] = {
            "total_images": total,
            "processed_images": len(records),
            "failed_images": failed
        }

        print(
            f"{split.upper()} complete: "
            f"{len(records)} processed, "
            f"{failed} failed"
        )

    if not all_records:
        raise RuntimeError(
            "No images were successfully processed."
        )

    summary = summarize(all_records)

    save_csv(all_records)

    summary["processing_status"] = failure_report

    report_path = (
        OUTPUT_DIR / "preprocessing_comparison_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPARISON RESULTS")
    print("=" * 65)

    for split in SPLITS:

        values = summary[split]

        print(f"\n{split.upper()}")
        print("-" * 40)

        print(f"Images: {values['image_count']}")

        for metric in [
            "brightness",
            "contrast_std",
            "dynamic_range",
            "unique_levels"
        ]:

            metric_values = values[metric]

            if metric_values is not None:
                print(
                    f"{metric}: "
                    f"mean={metric_values['mean']}, "
                    f"median={metric_values['median']}"
                )

    print("\nProcessing status:")

    for split, values in failure_report.items():
        print(f"{split.upper()}: {values}")

    print("\nReport saved:")
    print(report_path)
    print(
        OUTPUT_DIR / "preprocessing_image_records.csv"
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
