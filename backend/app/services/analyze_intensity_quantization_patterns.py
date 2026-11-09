
from pathlib import Path
import json
import csv

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_ROOT = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "images"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "intensity_quantization_patterns"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

# Analyze a maximum of 100 images per split for
# detailed per-image pattern analysis.
MAX_IMAGES_PER_SPLIT = 100


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(split):

    image_dir = DATASET_ROOT / split

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    paths = sorted([
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp"
        }
    ])

    return paths


# ============================================================
# LOAD IMAGE
# ============================================================

def load_grayscale(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    return image


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(path, split):

    image = load_grayscale(path)

    if image is None:
        return None

    pixels = image.ravel()

    histogram = np.bincount(
        pixels,
        minlength=256
    )

    occupied_levels = np.where(histogram > 0)[0]

    missing_levels = 256 - len(occupied_levels)

    gaps = np.diff(occupied_levels)

    if len(gaps) > 0:

        mean_gap = float(np.mean(gaps))
        max_gap = int(np.max(gaps))
        median_gap = float(np.median(gaps))

    else:

        mean_gap = 0
        max_gap = 0
        median_gap = 0

    probabilities = histogram / len(pixels)

    nonzero_probabilities = probabilities[
        probabilities > 0
    ]

    entropy = float(
        -np.sum(
            nonzero_probabilities
            * np.log2(nonzero_probabilities)
        )
    )

    # Fraction of pixels represented by the 10 most
    # frequent intensity levels.
    top_10_fraction = float(
        np.sum(np.sort(histogram)[-10:]) / len(pixels)
    )

    return {
        "image": path.name,
        "split": split,
        "unique_levels": int(len(occupied_levels)),
        "missing_levels": int(missing_levels),
        "mean_intensity_gap": round(mean_gap, 5),
        "median_intensity_gap": round(median_gap, 5),
        "maximum_intensity_gap": max_gap,
        "entropy_bits": round(entropy, 5),
        "top_10_intensity_fraction": round(
            top_10_fraction, 5
        )
    }, histogram


# ============================================================
# ANALYZE SPLIT
# ============================================================

def analyze_split(split):

    paths = find_images(split)

    if not paths:
        raise RuntimeError(
            f"No images found in {split}"
        )

    selected_paths = paths[:MAX_IMAGES_PER_SPLIT]

    records = []
    combined_histogram = np.zeros(
        256,
        dtype=np.float64
    )

    failed = 0

    for index, path in enumerate(
        selected_paths,
        start=1
    ):

        result = analyze_image(path, split)

        if result is None:
            failed += 1
            continue

        record, histogram = result

        records.append(record)
        combined_histogram += histogram

        if index % 25 == 0:
            print(
                f"{split}: {index}/{len(selected_paths)}"
            )

    if not records:
        raise RuntimeError(
            f"No readable images in {split}"
        )

    return {
        "records": records,
        "histogram": combined_histogram,
        "total_images": len(paths),
        "analyzed_images": len(records),
        "failed_images": failed
    }


# ============================================================
# SPLIT SUMMARY
# ============================================================

def summarize(records):

    metrics = [
        "unique_levels",
        "missing_levels",
        "mean_intensity_gap",
        "median_intensity_gap",
        "maximum_intensity_gap",
        "entropy_bits",
        "top_10_intensity_fraction"
    ]

    summary = {}

    for split in SPLITS:

        split_records = [
            record
            for record in records
            if record["split"] == split
        ]

        summary[split] = {
            "image_count": len(split_records)
        }

        for metric in metrics:

            values = np.array(
                [r[metric] for r in split_records],
                dtype=float
            )

            summary[split][metric] = {
                "mean": round(
                    float(np.mean(values)), 5
                ),
                "median": round(
                    float(np.median(values)), 5
                ),
                "std": round(
                    float(np.std(values)), 5
                )
            }

    return summary


# ============================================================
# SAVE HISTOGRAM
# ============================================================

def save_histograms(split_results):

    report = {}

    for split, result in split_results.items():

        histogram = result["histogram"]

        total_pixels = np.sum(histogram)

        normalized = histogram / total_pixels

        report[split] = {
            str(index): round(
                float(value), 10
            )
            for index, value in enumerate(normalized)
        }

    output_path = (
        OUTPUT_DIR / "normalized_intensity_histograms.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(f"\nHistograms saved: {output_path}")


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    if not records:
        return

    output_path = (
        OUTPUT_DIR / "quantization_pattern_records.csv"
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

    print(f"CSV saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("INTENSITY QUANTIZATION PATTERN ANALYSIS")
    print("=" * 65)

    print(
        f"\nImages analyzed per split: "
        f"maximum {MAX_IMAGES_PER_SPLIT}"
    )

    split_results = {}
    all_records = []

    for split in SPLITS:

        print(f"\nAnalyzing {split.upper()}...")

        result = analyze_split(split)

        split_results[split] = result

        all_records.extend(result["records"])

        print(
            f"Total images: {result['total_images']}"
        )

        print(
            f"Analyzed: {result['analyzed_images']}"
        )

        print(
            f"Failed: {result['failed_images']}"
        )

    summary = summarize(all_records)

    save_csv(all_records)

    save_histograms(split_results)

    report = {
        "sampling_limit_per_split": MAX_IMAGES_PER_SPLIT,
        "summary": summary,
        "processing": {
            split: {
                "total_images": result["total_images"],
                "analyzed_images": result["analyzed_images"],
                "failed_images": result["failed_images"]
            }
            for split, result in split_results.items()
        }
    }

    report_path = (
        OUTPUT_DIR / "quantization_pattern_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("QUANTIZATION ANALYSIS RESULTS")
    print("=" * 65)

    for split, values in summary.items():

        print(f"\n{split.upper()}")
        print("-" * 45)

        for metric, stats in values.items():

            if metric == "image_count":
                print(f"Images: {stats}")
                continue

            print(
                f"{metric}: "
                f"mean={stats['mean']}, "
                f"median={stats['median']}"
            )

    print("\nReports saved:")
    print(report_path)
    print(
        OUTPUT_DIR / "quantization_pattern_records.csv"
    )
    print(
        OUTPUT_DIR / "normalized_intensity_histograms.json"
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
