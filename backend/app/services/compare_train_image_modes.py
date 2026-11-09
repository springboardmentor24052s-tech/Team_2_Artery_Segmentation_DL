
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

TRAIN_DIR = (
    BACKEND_ROOT
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
    / "train"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "train_image_mode_comparison"
)

PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


# ============================================================
# FIND IMAGES BY MODE
# ============================================================

def find_images():

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory does not exist: {TRAIN_DIR}"
        )

    image_paths = sorted(
        path
        for path in TRAIN_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "annotation" not in str(path.parent).lower()
    )

    rgb_images = []
    grayscale_images = []
    unreadable_images = []

    for path in image_paths:

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            unreadable_images.append(path)
            continue

        if image.ndim == 2:
            grayscale_images.append(path)

        elif image.ndim == 3 and image.shape[2] == 3:
            rgb_images.append(path)

        elif image.ndim == 3 and image.shape[2] == 4:
            # Handle images with an alpha channel.
            rgb_images.append(path)

        else:
            unreadable_images.append(path)

    return rgb_images, grayscale_images, unreadable_images


# ============================================================
# IMAGE METRICS
# ============================================================

def analyze_image(path, group):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        return None

    if image.ndim == 3:

        if image.shape[2] == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY
            )

        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

    gray = image

    if gray.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got {gray.dtype}: {path}"
        )

    normalized = gray.astype(np.float32) / 255.0

    return {
        "image": path.name,
        "group": group,
        "brightness": float(np.mean(normalized)),
        "contrast_std": float(np.std(normalized)),
        "dynamic_range": float(
            np.percentile(normalized, 90)
            - np.percentile(normalized, 10)
        ),
        "unique_levels": int(len(np.unique(gray))),
        "dark_pixels_pct": float(
            np.mean(normalized < 0.10) * 100
        ),
        "bright_pixels_pct": float(
            np.mean(normalized > 0.90) * 100
        ),
        "p10": float(np.percentile(normalized, 10)),
        "p50": float(np.percentile(normalized, 50)),
        "p90": float(np.percentile(normalized, 90))
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize(records):

    summary = {}

    metrics = [
        "brightness",
        "contrast_std",
        "dynamic_range",
        "unique_levels",
        "dark_pixels_pct",
        "bright_pixels_pct"
    ]

    for group in ["RGB_stored", "grayscale_mode"]:

        group_records = [
            record
            for record in records
            if record["group"] == group
        ]

        group_summary = {
            "image_count": len(group_records)
        }

        for metric in metrics:

            values = np.array(
                [record[metric] for record in group_records],
                dtype=float
            )

            if len(values) == 0:
                group_summary[metric] = None
                continue

            group_summary[metric] = {
                "mean": round(float(np.mean(values)), 5),
                "median": round(float(np.median(values)), 5),
                "std": round(float(np.std(values)), 5),
                "min": round(float(np.min(values)), 5),
                "max": round(float(np.max(values)), 5)
            }

        summary[group] = group_summary

    return summary


# ============================================================
# CONVERT IMAGE TO GRAYSCALE
# ============================================================

def read_as_grayscale(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        return None

    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# ============================================================
# HISTOGRAM COMPARISON
# ============================================================

def generate_histogram(rgb_paths, gray_paths):

    rgb_hist = np.zeros(256, dtype=np.float64)
    gray_hist = np.zeros(256, dtype=np.float64)

    rgb_pixels = 0
    gray_pixels = 0

    for path in rgb_paths:

        gray = read_as_grayscale(path)

        if gray is None:
            continue

        rgb_hist += np.bincount(
            gray.ravel(),
            minlength=256
        )

        rgb_pixels += gray.size

    for path in gray_paths:

        gray = read_as_grayscale(path)

        if gray is None:
            continue

        gray_hist += np.bincount(
            gray.ravel(),
            minlength=256
        )

        gray_pixels += gray.size

    if rgb_pixels == 0 or gray_pixels == 0:
        print(
            "\nHistogram skipped: one image group has no readable pixels."
        )
        return

    rgb_hist /= rgb_pixels
    gray_hist /= gray_pixels

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.arange(256) / 255,
        rgb_hist,
        label="RGB-stored grayscale",
        linewidth=1.6
    )

    plt.plot(
        np.arange(256) / 255,
        gray_hist,
        label="Grayscale mode",
        linewidth=1.6
    )

    plt.title(
        "Training Image Intensity Distribution by Storage Mode"
    )
    plt.xlabel("Normalized Pixel Intensity")
    plt.ylabel("Proportion of Pixels")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_path = (
        PLOT_DIR / "train_mode_intensity_histogram.png"
    )

    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Histogram saved: {output_path}")


# ============================================================
# METRIC BOXPLOTS
# ============================================================

def generate_boxplots(records):

    metrics = [
        ("brightness", "Mean Brightness"),
        ("contrast_std", "Contrast Standard Deviation"),
        ("dynamic_range", "Dynamic Range"),
        ("unique_levels", "Unique Intensity Levels")
    ]

    for metric, title in metrics:

        rgb_values = [
            record[metric]
            for record in records
            if record["group"] == "RGB_stored"
        ]

        gray_values = [
            record[metric]
            for record in records
            if record["group"] == "grayscale_mode"
        ]

        if not rgb_values or not gray_values:
            print(
                f"Skipping {metric} boxplot: one group is empty."
            )
            continue

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            [rgb_values, gray_values],
            tick_labels=[
                "RGB-stored",
                "Grayscale mode"
            ],
            showmeans=True
        )

        plt.title(
            f"{title}: Training Image Comparison"
        )
        plt.ylabel(title)
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()

        output_path = (
            PLOT_DIR / f"{metric}_comparison.png"
        )

        plt.savefig(output_path, dpi=160)
        plt.close()

        print(f"Boxplot saved: {output_path}")


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    if not records:
        print("CSV skipped: no records available.")
        return

    path = OUTPUT_DIR / "train_mode_image_records.csv"

    with open(path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"CSV saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TRAINING IMAGE MODE COMPARISON")
    print("=" * 60)

    print(f"\nDataset directory:\n{TRAIN_DIR}")

    rgb_paths, gray_paths, unreadable_paths = find_images()

    print(f"\nRGB-stored images: {len(rgb_paths)}")
    print(f"Grayscale-mode images: {len(gray_paths)}")
    print(f"Unreadable/unsupported images: {len(unreadable_paths)}")

    if not rgb_paths and not gray_paths:
        raise RuntimeError(
            "No images were found. Check the dataset directory."
        )

    records = []

    for index, path in enumerate(rgb_paths, start=1):

        result = analyze_image(path, "RGB_stored")

        if result is not None:
            records.append(result)

        if index % 100 == 0:
            print(
                f"RGB processed: {index}/{len(rgb_paths)}"
            )

    for index, path in enumerate(gray_paths, start=1):

        result = analyze_image(path, "grayscale_mode")

        if result is not None:
            records.append(result)

        if index % 100 == 0:
            print(
                f"Grayscale processed: {index}/{len(gray_paths)}"
            )

    if not records:
        raise RuntimeError(
            "Images were found but none could be analyzed."
        )

    summary = summarize(records)

    save_csv(records)

    generate_histogram(rgb_paths, gray_paths)

    generate_boxplots(records)

    report_path = (
        OUTPUT_DIR / "train_mode_comparison_summary.json"
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

    print("\n" + "=" * 60)
    print("TRAINING IMAGE MODE COMPARISON RESULTS")
    print("=" * 60)

    for group, values in summary.items():

        print(f"\n{group}")
        print(f"Images: {values['image_count']}")

        for metric in [
            "brightness",
            "contrast_std",
            "dynamic_range",
            "unique_levels"
        ]:

            metric_values = values[metric]

            if metric_values is None:
                print(f"{metric}: No data")
                continue

            print(
                f"{metric}: "
                f"mean={metric_values['mean']}, "
                f"median={metric_values['median']}"
            )

    print("\nReports saved:")
    print(report_path)
    print(OUTPUT_DIR / "train_mode_image_records.csv")
    print(PLOT_DIR)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
