
import json
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_ROOT = (
    BACKEND_ROOT
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "image_quality_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# IMAGE QUALITY METRICS
# ============================================================

def calculate_image_metrics(image_path):

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    height, width = image.shape

    # Normalize intensity values to 0-1
    normalized = image.astype(np.float32) / 255.0

    # Brightness
    mean_brightness = float(np.mean(normalized))

    # Contrast indicator: standard deviation of intensity
    contrast_std = float(np.std(normalized))

    # Intensity range
    min_intensity = float(np.min(normalized))
    max_intensity = float(np.max(normalized))

    # Percentiles
    p10 = float(np.percentile(normalized, 10))
    p50 = float(np.percentile(normalized, 50))
    p90 = float(np.percentile(normalized, 90))

    # Dynamic intensity range
    dynamic_range = p90 - p10

    # Percentage of very dark and very bright pixels
    dark_pixel_percentage = float(
        np.mean(normalized < 0.10) * 100
    )

    bright_pixel_percentage = float(
        np.mean(normalized > 0.90) * 100
    )

    return {
        "image": image_path.name,
        "width": width,
        "height": height,
        "mean_brightness": mean_brightness,
        "contrast_std": contrast_std,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "dynamic_range": dynamic_range,
        "dark_pixel_percentage": dark_pixel_percentage,
        "bright_pixel_percentage": bright_pixel_percentage
    }


# ============================================================
# ANALYZE DATASET SPLITS
# ============================================================

def analyze_split(split):

    split_dir = DATASET_ROOT / split

    image_dir = split_dir / "images"

    if not image_dir.exists():
        image_dir = split_dir

    image_paths = sorted([
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print(f"\nAnalyzing {split.upper()} split...")
    print(f"Images found: {len(image_paths)}")

    records = []

    failed_images = []

    for index, image_path in enumerate(image_paths, start=1):

        metrics = calculate_image_metrics(image_path)

        if metrics is None:
            failed_images.append(image_path.name)
            continue

        metrics["split"] = split

        records.append(metrics)

        if index % 100 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    return records, failed_images


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_summary(records):

    metric_names = [
        "mean_brightness",
        "contrast_std",
        "dynamic_range",
        "dark_pixel_percentage",
        "bright_pixel_percentage"
    ]

    summary = {
        "image_count": len(records)
    }

    for metric in metric_names:

        values = np.array(
            [record[metric] for record in records],
            dtype=float
        )

        if len(values) == 0:
            continue

        summary[metric] = {
            "mean": round(float(np.mean(values)), 5),
            "median": round(float(np.median(values)), 5),
            "std": round(float(np.std(values)), 5),
            "min": round(float(np.min(values)), 5),
            "max": round(float(np.max(values)), 5),
            "p25": round(float(np.percentile(values, 25)), 5),
            "p75": round(float(np.percentile(values, 75)), 5)
        }

    return summary


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    output_path = OUTPUT_DIR / "image_quality_records.csv"

    if not records:
        print("No records available for CSV.")
        return

    fieldnames = list(records[0].keys())

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nCSV saved: {output_path}")


# ============================================================
# VISUALIZATION
# ============================================================

def generate_plots(records):

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("mean_brightness", "Mean Brightness"),
        ("contrast_std", "Contrast Standard Deviation"),
        ("dynamic_range", "Dynamic Intensity Range"),
        ("dark_pixel_percentage", "Dark Pixel Percentage")
    ]

    split_colors = {
        "train": "steelblue",
        "val": "darkorange",
        "test": "seagreen"
    }

    # Individual distributions
    for metric, title in metrics:

        plt.figure(figsize=(9, 5))

        for split in SPLITS:

            values = [
                record[metric]
                for record in records
                if record["split"] == split
            ]

            if values:

                plt.hist(
                    values,
                    bins=30,
                    alpha=0.45,
                    label=split.upper(),
                    color=split_colors[split]
                )

        plt.title(f"{title} Distribution Across Dataset Splits")
        plt.xlabel(title)
        plt.ylabel("Number of Images")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()

        filename = f"{metric}_distribution.png"

        plt.savefig(
            plot_dir / filename,
            dpi=150
        )

        plt.close()

    # Box plots for split comparison
    for metric, title in metrics:

        data = [
            [
                record[metric]
                for record in records
                if record["split"] == split
            ]
            for split in SPLITS
        ]

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            data,
            tick_labels=["Train", "Validation", "Test"],
            showmeans=True
        )

        plt.title(f"{title}: Split Comparison")
        plt.ylabel(title)
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()

        plt.savefig(
            plot_dir / f"{metric}_split_comparison.png",
            dpi=150
        )

        plt.close()

    print(f"\nGraphs saved in: {plot_dir}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STENOSIS IMAGE QUALITY ANALYSIS")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}"
        )

    all_records = []

    failed_report = {}

    for split in SPLITS:

        records, failed = analyze_split(split)

        all_records.extend(records)

        failed_report[split] = failed

    # Summary
    split_summaries = {}

    for split in SPLITS:

        split_records = [
            record for record in all_records
            if record["split"] == split
        ]

        split_summaries[split] = calculate_summary(
            split_records
        )

    report = {
        "dataset": "ARCADE Stenosis",
        "total_successfully_processed": len(all_records),
        "total_failed": sum(
            len(items) for items in failed_report.values()
        ),
        "split_summaries": split_summaries,
        "failed_images": failed_report
    }

    # Save JSON
    json_path = OUTPUT_DIR / "image_quality_summary.json"

    with open(json_path, "w") as file:
        json.dump(report, file, indent=4)

    # Save CSV
    save_csv(all_records)

    # Save plots
    generate_plots(all_records)

    # Print summary
    print("\nIMAGE QUALITY SUMMARY")
    print("=" * 60)

    for split, summary in split_summaries.items():

        print(f"\n{split.upper()}")

        print(
            f"Images: {summary['image_count']}"
        )

        for metric in [
            "mean_brightness",
            "contrast_std",
            "dynamic_range",
            "dark_pixel_percentage"
        ]:

            if metric in summary:

                print(
                    f"{metric}: "
                    f"mean={summary[metric]['mean']}, "
                    f"median={summary[metric]['median']}"
                )

    print("\nFailed images:", report["total_failed"])

    print("\nReports saved:")
    print(json_path)
    print(OUTPUT_DIR / "image_quality_records.csv")


if __name__ == "__main__":
    main()
