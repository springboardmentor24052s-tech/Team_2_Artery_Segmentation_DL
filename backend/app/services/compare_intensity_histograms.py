
from pathlib import Path
import json

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
    BACKEND_ROOT / "results" / "intensity_histogram_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

BINS = 256


# ============================================================
# FIND IMAGES
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
# COLLECT INTENSITY DISTRIBUTIONS
# ============================================================

def collect_intensities(split):

    image_paths = get_image_paths(split)

    print(f"\nProcessing {split.upper()}: {len(image_paths)} images")

    histogram = np.zeros(BINS, dtype=np.float64)
    total_pixels = 0
    image_means = []
    failed_images = []

    for index, image_path in enumerate(image_paths, start=1):

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            failed_images.append(image_path.name)
            continue

        counts = np.bincount(
            image.ravel(),
            minlength=BINS
        )

        histogram += counts
        total_pixels += image.size

        image_means.append(
            float(np.mean(image) / 255.0)
        )

        if index % 100 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    if total_pixels == 0:
        raise RuntimeError(
            f"No readable images found in {split}"
        )

    # Normalize to probability distribution
    normalized_histogram = histogram / total_pixels

    # Cumulative distribution
    cdf = np.cumsum(normalized_histogram)

    return {
        "split": split,
        "image_count": len(image_means),
        "total_pixels": int(total_pixels),
        "histogram": normalized_histogram,
        "cdf": cdf,
        "mean_brightness": float(np.mean(image_means)),
        "failed_images": failed_images
    }


# ============================================================
# PLOT HISTOGRAMS
# ============================================================

def plot_histograms(results):

    intensity_values = np.arange(BINS) / 255.0

    plt.figure(figsize=(11, 6))

    for split in SPLITS:

        result = results[split]

        plt.plot(
            intensity_values,
            result["histogram"],
            label=split.upper(),
            linewidth=1.5
        )

    plt.title("Grayscale Intensity Distribution")
    plt.xlabel("Normalized Pixel Intensity (0–1)")
    plt.ylabel("Proportion of Pixels")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "combined_intensity_histogram.png",
        dpi=180
    )

    plt.close()


# ============================================================
# PLOT CUMULATIVE DISTRIBUTIONS
# ============================================================

def plot_cdfs(results):

    intensity_values = np.arange(BINS) / 255.0

    plt.figure(figsize=(11, 6))

    for split in SPLITS:

        result = results[split]

        plt.plot(
            intensity_values,
            result["cdf"],
            label=split.upper(),
            linewidth=1.8
        )

    plt.title("Cumulative Grayscale Intensity Distribution")
    plt.xlabel("Normalized Pixel Intensity (0–1)")
    plt.ylabel("Cumulative Proportion")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "combined_intensity_cdf.png",
        dpi=180
    )

    plt.close()


# ============================================================
# PLOT MEAN BRIGHTNESS
# ============================================================

def plot_mean_brightness(results):

    splits = SPLITS

    values = [
        results[split]["mean_brightness"]
        for split in splits
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(splits, values)

    plt.title("Mean Image Brightness by Dataset Split")
    plt.xlabel("Dataset Split")
    plt.ylabel("Mean Normalized Brightness")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "mean_brightness_comparison.png",
        dpi=180
    )

    plt.close()


# ============================================================
# SAVE NUMERICAL REPORT
# ============================================================

def save_report(results):

    report = {}

    for split in SPLITS:

        result = results[split]

        report[split] = {
            "image_count": result["image_count"],
            "total_pixels": result["total_pixels"],
            "mean_brightness": round(
                result["mean_brightness"], 6
            ),
            "failed_images": result["failed_images"]
        }

    # Pairwise Jensen-Shannon divergence
    # Calculated from the shared normalized histograms.
    def js_divergence(p, q):

        epsilon = 1e-12

        p = p + epsilon
        q = q + epsilon

        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log2(p / m))
        kl_qm = np.sum(q * np.log2(q / m))

        return float(0.5 * (kl_pm + kl_qm))

    divergences = {}

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test")
    ]

    for first, second in pairs:

        divergence = js_divergence(
            results[first]["histogram"],
            results[second]["histogram"]
        )

        divergences[f"{first}_vs_{second}"] = round(
            divergence, 6
        )

    report["pairwise_js_divergence"] = divergences

    output_path = OUTPUT_DIR / "intensity_distribution_report.json"

    with open(output_path, "w") as file:
        json.dump(report, file, indent=4)

    print(f"\nJSON report saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DATASET INTENSITY DISTRIBUTION ANALYSIS")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(DATASET_ROOT)

    results = {}

    for split in SPLITS:
        results[split] = collect_intensities(split)

    plot_histograms(results)
    plot_cdfs(results)
    plot_mean_brightness(results)
    save_report(results)

    print("\nINTENSITY DISTRIBUTION SUMMARY")
    print("=" * 60)

    for split in SPLITS:

        result = results[split]

        print(f"\n{split.upper()}")
        print(f"Images: {result['image_count']}")
        print(
            f"Mean brightness: "
            f"{result['mean_brightness']:.5f}"
        )
        print(
            f"Failed images: "
            f"{len(result['failed_images'])}"
        )

    print("\nPAIRWISE JS DIVERGENCE")

    report_path = OUTPUT_DIR / "intensity_distribution_report.json"

    with open(report_path) as file:
        report = json.load(file)

    for pair, value in report["pairwise_js_divergence"].items():
        print(f"{pair}: {value}")

    print("\nGraphs saved in:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()
