
from pathlib import Path
import json

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

DATASET_ROOT = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "images"
)

SPLITS = ["train", "val", "test"]

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "pixel_intensity_parity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def analyze_split(split):

    image_dir = DATASET_ROOT / split

    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    even_pixels = 0
    odd_pixels = 0
    total_pixels = 0

    divisible_counts = {
        "divisible_by_2": 0,
        "divisible_by_4": 0,
        "divisible_by_8": 0,
        "divisible_by_16": 0
    }

    image_results = []

    for index, path in enumerate(paths, start=1):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        pixels = image.ravel().astype(np.int64)

        even = int(np.sum(pixels % 2 == 0))
        odd = int(np.sum(pixels % 2 == 1))

        count = pixels.size

        even_pixels += even
        odd_pixels += odd
        total_pixels += count

        for divisor in [2, 4, 8, 16]:

            divisible_counts[
                f"divisible_by_{divisor}"
            ] += int(np.sum(pixels % divisor == 0))

        unique_values = np.unique(pixels)

        image_results.append({
            "image": path.name,
            "unique_levels": int(len(unique_values)),
            "even_pixel_percentage": round(
                even / count * 100, 3
            ),
            "odd_pixel_percentage": round(
                odd / count * 100, 3
            ),
            "even_unique_levels": int(
                np.sum(unique_values % 2 == 0)
            ),
            "odd_unique_levels": int(
                np.sum(unique_values % 2 == 1)
            )
        })

        if index % 100 == 0:
            print(f"{split}: {index}/{len(paths)}")

    if total_pixels == 0:
        raise RuntimeError(f"No readable images in {split}")

    result = {
        "image_count": len(image_results),
        "total_pixels": total_pixels,
        "even_pixels": even_pixels,
        "odd_pixels": odd_pixels,
        "even_pixel_percentage": round(
            even_pixels / total_pixels * 100, 5
        ),
        "odd_pixel_percentage": round(
            odd_pixels / total_pixels * 100, 5
        ),
        "divisibility_percentage": {
            key: round(value / total_pixels * 100, 5)
            for key, value in divisible_counts.items()
        },
        "per_image": {
            "even_pixel_percentage_mean": round(
                float(np.mean([
                    r["even_pixel_percentage"]
                    for r in image_results
                ])), 5
            ),
            "odd_pixel_percentage_mean": round(
                float(np.mean([
                    r["odd_pixel_percentage"]
                    for r in image_results
                ])), 5
            ),
            "even_unique_levels_mean": round(
                float(np.mean([
                    r["even_unique_levels"]
                    for r in image_results
                ])), 5
            ),
            "odd_unique_levels_mean": round(
                float(np.mean([
                    r["odd_unique_levels"]
                    for r in image_results
                ])), 5
            )
        }
    }

    return result


def main():

    print("=" * 65)
    print("PIXEL INTENSITY PARITY ANALYSIS")
    print("=" * 65)

    results = {}

    for split in SPLITS:

        print(f"\nAnalyzing {split.upper()}...")

        results[split] = analyze_split(split)

    report_path = (
        OUTPUT_DIR / "pixel_intensity_parity_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(results, file, indent=4)

    print("\n" + "=" * 65)
    print("PIXEL INTENSITY PARITY RESULTS")
    print("=" * 65)

    for split, result in results.items():

        print(f"\n{split.upper()}")
        print("-" * 45)

        print(f"Images: {result['image_count']}")

        print(
            f"Even pixels: "
            f"{result['even_pixel_percentage']}%"
        )

        print(
            f"Odd pixels: "
            f"{result['odd_pixel_percentage']}%"
        )

        print("\nDivisibility:")

        for key, value in (
            result["divisibility_percentage"].items()
        ):
            print(f"{key}: {value}%")

        print("\nAverage per-image unique levels:")

        print(
            result["per_image"]
        )

    print("\nReport saved:")
    print(report_path)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
