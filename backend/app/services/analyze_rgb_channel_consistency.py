
from pathlib import Path
import json
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_ROOT = (
    BACKEND_ROOT / "app" / "data" / "arcade"
    / "arcade" / "arcade" / "stenosis"
)

OUTPUT_DIR = (
    BACKEND_ROOT / "results" / "rgb_channel_analysis"
)

PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def get_image_paths(split):

    split_dir = DATASET_ROOT / split
    image_dir = split_dir / "images"

    if not image_dir.exists():
        image_dir = split_dir

    return sorted([
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ])


# ============================================================
# CHANNEL ANALYSIS
# ============================================================

def analyze_image(image_path, split):

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        return None

    record = {
        "image": image_path.name,
        "split": split,
        "shape": str(image.shape),
        "dtype": str(image.dtype),
        "image_type": "unknown"
    }

    if image.ndim == 2:

        record.update({
            "image_type": "grayscale",
            "unique_r_g": None,
            "unique_g_b": None,
            "unique_r_b": None,
            "mean_abs_rg": None,
            "mean_abs_gb": None,
            "mean_abs_rb": None,
            "identical_channels": None,
            "grayscale_conversion_difference": None
        })

        return record

    if image.ndim != 3 or image.shape[2] < 3:
        record["image_type"] = "unsupported"
        return record

    # OpenCV loads color images in BGR order.
    b, g, r = cv2.split(image[:, :, :3])

    # Channel equality
    identical_rg = np.array_equal(r, g)
    identical_gb = np.array_equal(g, b)
    identical_rb = np.array_equal(r, b)

    # Mean absolute channel differences
    mean_abs_rg = float(np.mean(
        np.abs(r.astype(np.float32) - g.astype(np.float32))
    ))

    mean_abs_gb = float(np.mean(
        np.abs(g.astype(np.float32) - b.astype(np.float32))
    ))

    mean_abs_rb = float(np.mean(
        np.abs(r.astype(np.float32) - b.astype(np.float32))
    ))

    # Convert to grayscale using OpenCV
    converted = cv2.cvtColor(
        image[:, :, :3],
        cv2.COLOR_BGR2GRAY
    )

    # Compare grayscale conversion with each original channel
    diff_r = float(np.mean(
        np.abs(converted.astype(np.float32) - r.astype(np.float32))
    ))

    diff_g = float(np.mean(
        np.abs(converted.astype(np.float32) - g.astype(np.float32))
    ))

    diff_b = float(np.mean(
        np.abs(converted.astype(np.float32) - b.astype(np.float32))
    ))

    record.update({
        "image_type": "RGB",
        "unique_r_g": identical_rg,
        "unique_g_b": identical_gb,
        "unique_r_b": identical_rb,
        "mean_abs_rg": round(mean_abs_rg, 5),
        "mean_abs_gb": round(mean_abs_gb, 5),
        "mean_abs_rb": round(mean_abs_rb, 5),
        "identical_channels": (
            identical_rg and identical_gb and identical_rb
        ),
        "grayscale_conversion_difference": round(
            float(np.mean(
                np.abs(
                    converted.astype(np.float32)
                    - r.astype(np.float32)
                )
            )), 5
        ),
        "gray_vs_r": round(diff_r, 5),
        "gray_vs_g": round(diff_g, 5),
        "gray_vs_b": round(diff_b, 5)
    })

    return record


# ============================================================
# ANALYZE ALL SPLITS
# ============================================================

def analyze_all_splits():

    all_records = []

    failed_images = []

    for split in SPLITS:

        paths = get_image_paths(split)

        print(f"\nAnalyzing {split.upper()}")
        print(f"Images: {len(paths)}")

        for index, path in enumerate(paths, start=1):

            record = analyze_image(path, split)

            if record is None:
                failed_images.append({
                    "split": split,
                    "image": path.name
                })
                continue

            all_records.append(record)

            if index % 100 == 0:
                print(f"Processed {index}/{len(paths)}")

    return all_records, failed_images


# ============================================================
# SUMMARY
# ============================================================

def summarize(records):

    summary = {}

    for split in SPLITS:

        split_records = [
            r for r in records if r["split"] == split
        ]

        rgb_records = [
            r for r in split_records
            if r["image_type"] == "RGB"
        ]

        grayscale_records = [
            r for r in split_records
            if r["image_type"] == "grayscale"
        ]

        identical = [
            r for r in rgb_records
            if r["identical_channels"] is True
        ]

        summary[split] = {
            "total_images": len(split_records),
            "rgb_images": len(rgb_records),
            "grayscale_images": len(grayscale_records),
            "unsupported_images": sum(
                r["image_type"] == "unsupported"
                for r in split_records
            ),
            "rgb_with_identical_channels": len(identical),
            "rgb_with_different_channels": (
                len(rgb_records) - len(identical)
            )
        }

        if rgb_records:

            for metric in [
                "mean_abs_rg",
                "mean_abs_gb",
                "mean_abs_rb",
                "grayscale_conversion_difference"
            ]:

                values = np.array([
                    r[metric] for r in rgb_records
                ], dtype=float)

                summary[split][metric] = {
                    "mean": round(float(np.mean(values)), 5),
                    "median": round(float(np.median(values)), 5),
                    "maximum": round(float(np.max(values)), 5)
                }

    return summary


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    csv_path = OUTPUT_DIR / "rgb_channel_records.csv"

    fields = [
        "image",
        "split",
        "shape",
        "dtype",
        "image_type",
        "unique_r_g",
        "unique_g_b",
        "unique_r_b",
        "mean_abs_rg",
        "mean_abs_gb",
        "mean_abs_rb",
        "identical_channels",
        "grayscale_conversion_difference",
        "gray_vs_r",
        "gray_vs_g",
        "gray_vs_b"
    ]

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)

    print(f"\nCSV saved: {csv_path}")


# ============================================================
# VISUALIZATION
# ============================================================

def generate_plots(records):

    rgb_records = [
        r for r in records
        if r["image_type"] == "RGB"
    ]

    if not rgb_records:
        print("No RGB images available for plotting.")
        return

    metrics = [
        ("mean_abs_rg", "R-G Difference"),
        ("mean_abs_gb", "G-B Difference"),
        ("mean_abs_rb", "R-B Difference"),
        (
            "grayscale_conversion_difference",
            "Grayscale Conversion Difference"
        )
    ]

    for metric, title in metrics:

        values = [
            r[metric] for r in rgb_records
        ]

        plt.figure(figsize=(9, 5))

        plt.hist(values, bins=30)

        plt.title(f"{title} Across RGB Training Images")
        plt.xlabel("Mean Absolute Pixel Difference")
        plt.ylabel("Number of Images")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / f"{metric}_distribution.png",
            dpi=150
        )

        plt.close()

    print(f"Plots saved: {PLOT_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RGB CHANNEL CONSISTENCY ANALYSIS")
    print("=" * 60)

    records, failed = analyze_all_splits()

    summary = summarize(records)

    save_csv(records)

    generate_plots(records)

    report = {
        "dataset": "ARCADE Stenosis",
        "total_processed": len(records),
        "total_failed": len(failed),
        "split_summaries": summary,
        "failed_images": failed
    }

    json_path = OUTPUT_DIR / "rgb_channel_summary.json"

    with open(json_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nRGB CHANNEL ANALYSIS SUMMARY")
    print("=" * 60)

    for split, result in summary.items():

        print(f"\n{split.upper()}")

        for key, value in result.items():
            print(f"{key}: {value}")

    print("\nFailed images:", len(failed))

    print("\nReports saved:")
    print(json_path)
    print(OUTPUT_DIR / "rgb_channel_records.csv")


if __name__ == "__main__":
    main()
