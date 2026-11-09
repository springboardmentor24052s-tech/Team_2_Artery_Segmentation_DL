
import json
import csv
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS AND CONFIGURATION
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
    / "intensity_quantization_analysis"
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
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        return None

    original_dtype = str(image.dtype)

    original_bit_depth = image.dtype.itemsize * 8

    if image.ndim == 2:
        grayscale = image

    elif image.ndim == 3:

        if image.shape[2] == 4:
            grayscale = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY
            )
        else:
            grayscale = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

    else:
        return None

    height, width = grayscale.shape

    unique_values = np.unique(grayscale)

    unique_count = len(unique_values)

    minimum = int(np.min(grayscale))
    maximum = int(np.max(grayscale))

    # Calculate intensity gaps
    gaps = np.diff(unique_values)

    missing_levels = (
        max(0, maximum - minimum + 1 - unique_count)
    )

    mean_intensity = float(np.mean(grayscale))

    std_intensity = float(np.std(grayscale))

    total_pixels = grayscale.size

    minimum_percentage = float(
        np.mean(grayscale == minimum) * 100
    )

    maximum_percentage = float(
        np.mean(grayscale == maximum) * 100
    )

    # Most frequent pixel values
    counts = np.bincount(
        grayscale.astype(np.int64).ravel()
    )

    top_indices = np.argsort(counts)[-5:][::-1]

    top_values = [
        {
            "intensity": int(value),
            "count": int(counts[value]),
            "percentage": round(
                counts[value] / total_pixels * 100, 4
            )
        }
        for value in top_indices
        if counts[value] > 0
    ]

    return {
        "image": image_path.name,
        "width": width,
        "height": height,
        "dtype": original_dtype,
        "original_bit_depth": original_bit_depth,
        "unique_intensity_levels": int(unique_count),
        "minimum_intensity": minimum,
        "maximum_intensity": maximum,
        "missing_intensity_levels": int(missing_levels),
        "largest_intensity_gap": int(
            np.max(gaps) if len(gaps) else 0
        ),
        "mean_intensity": round(mean_intensity, 4),
        "std_intensity": round(std_intensity, 4),
        "minimum_pixel_percentage": round(
            minimum_percentage, 4
        ),
        "maximum_pixel_percentage": round(
            maximum_percentage, 4
        ),
        "top_pixel_values": top_values
    }


# ============================================================
# SPLIT ANALYSIS
# ============================================================

def analyze_split(split):

    image_paths = get_image_paths(split)

    print(f"\nAnalyzing {split.upper()}")
    print(f"Images found: {len(image_paths)}")

    records = []
    failed_images = []

    for index, image_path in enumerate(image_paths, start=1):

        result = analyze_image(image_path)

        if result is None:
            failed_images.append(image_path.name)
            continue

        result["split"] = split

        records.append(result)

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(image_paths)}"
            )

    return records, failed_images


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_summary(records):

    if not records:
        return {"image_count": 0}

    unique_levels = np.array([
        r["unique_intensity_levels"] for r in records
    ])

    means = np.array([
        r["mean_intensity"] for r in records
    ])

    stds = np.array([
        r["std_intensity"] for r in records
    ])

    bit_depths = Counter(
        r["original_bit_depth"] for r in records
    )

    dtypes = Counter(
        r["dtype"] for r in records
    )

    dimensions = Counter(
        f"{r['width']}x{r['height']}" for r in records
    )

    return {
        "image_count": len(records),
        "bit_depth_distribution": dict(bit_depths),
        "dtype_distribution": dict(dtypes),
        "dimension_distribution": dict(dimensions),
        "unique_levels": {
            "mean": round(float(np.mean(unique_levels)), 3),
            "median": round(float(np.median(unique_levels)), 3),
            "minimum": int(np.min(unique_levels)),
            "maximum": int(np.max(unique_levels))
        },
        "mean_intensity": {
            "mean": round(float(np.mean(means)), 4),
            "median": round(float(np.median(means)), 4)
        },
        "intensity_std": {
            "mean": round(float(np.mean(stds)), 4),
            "median": round(float(np.median(stds)), 4)
        },
        "images_with_fewer_than_100_levels": int(
            np.sum(unique_levels < 100)
        ),
        "images_with_fewer_than_200_levels": int(
            np.sum(unique_levels < 200)
        )
    }


# ============================================================
# SAVE IMAGE-LEVEL CSV
# ============================================================

def save_csv(records):

    csv_path = OUTPUT_DIR / "quantization_records.csv"

    fields = [
        "image",
        "split",
        "width",
        "height",
        "dtype",
        "original_bit_depth",
        "unique_intensity_levels",
        "minimum_intensity",
        "maximum_intensity",
        "missing_intensity_levels",
        "largest_intensity_gap",
        "mean_intensity",
        "std_intensity",
        "minimum_pixel_percentage",
        "maximum_pixel_percentage"
    ]

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                key: record[key] for key in fields
            })

    print(f"\nCSV saved: {csv_path}")


# ============================================================
# PLOT UNIQUE INTENSITY LEVELS
# ============================================================

def plot_unique_levels(records):

    data = []

    for split in SPLITS:

        values = [
            r["unique_intensity_levels"]
            for r in records
            if r["split"] == split
        ]

        data.append(values)

    plt.figure(figsize=(9, 5))

    plt.boxplot(
        data,
        tick_labels=["Train", "Validation", "Test"],
        showmeans=True
    )

    plt.title("Unique Grayscale Intensity Levels")
    plt.ylabel("Number of Unique Intensity Values")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "unique_intensity_levels.png",
        dpi=160
    )

    plt.close()


# ============================================================
# PLOT BIT-DEPTH DISTRIBUTION
# ============================================================

def plot_bit_depth(summary):

    splits = list(summary.keys())

    bit_depths = sorted({
        str(bit_depth)
        for split in splits
        for bit_depth in summary[split].get(
            "bit_depth_distribution", {}
        )
    })

    x = np.arange(len(splits))
    width = 0.25

    plt.figure(figsize=(8, 5))

    for index, bit_depth in enumerate(bit_depths):

        counts = [
            summary[split]["bit_depth_distribution"].get(
                bit_depth, 0
            )
            for split in splits
        ]

        plt.bar(
            x + index * width,
            counts,
            width,
            label=f"{bit_depth}-bit"
        )

    plt.xticks(
        x + width * max(0, len(bit_depths) - 1) / 2,
        [s.upper() for s in splits]
    )

    plt.title("Image Bit-Depth Distribution")
    plt.ylabel("Number of Images")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "bit_depth_distribution.png",
        dpi=160
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IMAGE INTENSITY QUANTIZATION ANALYSIS")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    all_records = []
    failed_report = {}
    summaries = {}

    for split in SPLITS:

        records, failed = analyze_split(split)

        all_records.extend(records)

        failed_report[split] = failed

        summaries[split] = calculate_summary(records)

    save_csv(all_records)

    plot_unique_levels(all_records)

    plot_bit_depth(summaries)

    report = {
        "dataset": "ARCADE Stenosis",
        "total_processed_images": len(all_records),
        "total_failed_images": sum(
            len(v) for v in failed_report.values()
        ),
        "split_summaries": summaries,
        "failed_images": failed_report
    }

    json_path = OUTPUT_DIR / "quantization_summary.json"

    with open(json_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nQUANTIZATION ANALYSIS SUMMARY")
    print("=" * 60)

    for split, summary in summaries.items():

        print(f"\n{split.upper()}")
        print(f"Images: {summary['image_count']}")
        print(
            "Bit depth:",
            summary["bit_depth_distribution"]
        )
        print(
            "Data types:",
            summary["dtype_distribution"]
        )
        print(
            "Unique intensity levels:",
            summary["unique_levels"]
        )
        print(
            "Images with fewer than 100 levels:",
            summary["images_with_fewer_than_100_levels"]
        )

    print("\nTotal failed images:", report["total_failed_images"])

    print("\nReports saved:")
    print(json_path)
    print(OUTPUT_DIR / "quantization_records.csv")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()
