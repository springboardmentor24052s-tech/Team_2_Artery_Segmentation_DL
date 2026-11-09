
import json
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT / "backend" / "app" / "data"
    / "arcade" / "arcade" / "arcade" / "stenosis"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "backend" / "results"
    / "pixel_intensity_distribution"
)

SPLITS = ["train", "val", "test"]


def analyze_split(split):
    split_dir = DATASET_ROOT / split

    image_paths = [
        p for p in split_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and "annotations" not in p.parts
    ]

    histogram = np.zeros(256, dtype=np.int64)
    image_count = 0
    failed = []

    print(f"\nAnalyzing {split.upper()} ({len(image_paths)} images)")

    for index, path in enumerate(image_paths, start=1):
        try:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("L"))

            histogram += np.bincount(
                pixels.ravel(), minlength=256
            )
            image_count += 1

        except Exception as error:
            failed.append({
                "filename": path.name,
                "error": str(error)
            })

        if index % 100 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    total_pixels = int(histogram.sum())
    used_levels = np.where(histogram > 0)[0]

    percentages = (
        histogram / total_pixels * 100
        if total_pixels else np.zeros(256)
    )

    top_indices = np.argsort(histogram)[::-1][:15]

    summary = {
        "images_processed": image_count,
        "failed_images": len(failed),
        "total_pixels": total_pixels,
        "unique_intensity_levels": int(len(used_levels)),
        "percentage_of_256_levels_used": round(
            len(used_levels) / 256 * 100, 4
        ),
        "minimum_used_level": int(used_levels.min()) if len(used_levels) else None,
        "maximum_used_level": int(used_levels.max()) if len(used_levels) else None,
        "mean_intensity": round(
            float(np.average(np.arange(256), weights=histogram)), 5
        ) if total_pixels else None,
        "top_15_intensity_values": [
            {
                "intensity": int(i),
                "pixel_count": int(histogram[i]),
                "percentage": round(float(percentages[i]), 5)
            }
            for i in top_indices
        ],
        "histogram": histogram.tolist(),
        "percentage_distribution": percentages.tolist(),
        "failed": failed
    }

    return summary


def create_plots(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Absolute pixel frequency
    plt.figure(figsize=(14, 6))

    for split in SPLITS:
        histogram = results[split]["histogram"]
        plt.plot(range(256), histogram, label=split.upper())

    plt.title("Pixel Intensity Frequency by Dataset Split")
    plt.xlabel("Grayscale Intensity (0-255)")
    plt.ylabel("Number of Pixels")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "intensity_frequency.png", dpi=200)
    plt.close()

    # Plot 2: Normalized pixel distribution
    plt.figure(figsize=(14, 6))

    for split in SPLITS:
        percentages = results[split]["percentage_distribution"]
        plt.plot(range(256), percentages, label=split.upper())

    plt.title("Normalized Pixel Intensity Distribution")
    plt.xlabel("Grayscale Intensity (0-255)")
    plt.ylabel("Percentage of Pixels")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "normalized_intensity_distribution.png", dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for split in SPLITS:
        results[split] = analyze_split(split)

        summary = results[split]

        print("\n" + "=" * 60)
        print(f"{split.upper()} SUMMARY")
        print("=" * 60)

        for key in [
            "images_processed",
            "failed_images",
            "total_pixels",
            "unique_intensity_levels",
            "percentage_of_256_levels_used",
            "minimum_used_level",
            "maximum_used_level",
            "mean_intensity"
        ]:
            print(f"{key}: {summary[key]}")

        print("\nTop 15 intensity values:")
        for item in summary["top_15_intensity_values"]:
            print(item)

    create_plots(results)

    report_path = OUTPUT_DIR / "pixel_intensity_distribution_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 60)
    print("PIXEL INTENSITY ANALYSIS COMPLETED")
    print("=" * 60)
    print(f"Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
