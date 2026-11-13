
from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

ORIGINAL_DIR = (
    BACKEND_ROOT
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
    / "test"
)

YOLO_DIR = (
    BACKEND_ROOT
    / "results"
    / "stenosis_yolo_dataset"
    / "images"
    / "test"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "original_vs_yolo_image_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp"
}


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    values = np.unique(image)

    gaps = np.diff(values)

    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "unique_levels": int(len(values)),
        "mean_intensity": float(np.mean(image) / 255),
        "std_intensity": float(np.std(image) / 255),
        "gap_1_percentage": float(
            np.mean(gaps == 1) * 100
        ) if len(gaps) else 0,
        "gap_2_percentage": float(
            np.mean(gaps == 2) * 100
        ) if len(gaps) else 0,
        "minimum_intensity": int(values.min()),
        "maximum_intensity": int(values.max())
    }


# ============================================================
# MATCH AND COMPARE IMAGES
# ============================================================

def main():

    print("=" * 65)
    print("ORIGINAL VS YOLO TEST IMAGE COMPARISON")
    print("=" * 65)

    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(ORIGINAL_DIR)

    if not YOLO_DIR.exists():
        raise FileNotFoundError(YOLO_DIR)

    original_paths = {
        path.name: path
        for path in ORIGINAL_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "annotation" not in str(path.parent).lower()
    }

    yolo_paths = {
        path.name: path
        for path in YOLO_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    common_names = sorted(
        set(original_paths) & set(yolo_paths)
    )

    print(f"\nOriginal images: {len(original_paths)}")
    print(f"YOLO images: {len(yolo_paths)}")
    print(f"Matching filenames: {len(common_names)}")

    if not common_names:
        raise RuntimeError(
            "No matching image filenames found."
        )

    records = []
    pixel_mismatches = 0
    dimension_mismatches = 0

    for index, name in enumerate(common_names, start=1):

        original = cv2.imread(
            str(original_paths[name]),
            cv2.IMREAD_GRAYSCALE
        )

        prepared = cv2.imread(
            str(yolo_paths[name]),
            cv2.IMREAD_GRAYSCALE
        )

        if original is None or prepared is None:
            continue

        original_stats = analyze_image(
            original_paths[name]
        )

        prepared_stats = analyze_image(
            yolo_paths[name]
        )

        same_dimensions = (
            original.shape == prepared.shape
        )

        exact_pixel_match = False
        mean_absolute_difference = None
        maximum_absolute_difference = None

        if same_dimensions:

            difference = cv2.absdiff(
                original,
                prepared
            )

            exact_pixel_match = bool(
                np.array_equal(original, prepared)
            )

            mean_absolute_difference = float(
                np.mean(difference)
            )

            maximum_absolute_difference = int(
                np.max(difference)
            )

            if not exact_pixel_match:
                pixel_mismatches += 1

        else:
            dimension_mismatches += 1

        records.append({
            "image": name,
            "same_dimensions": same_dimensions,
            "exact_pixel_match": exact_pixel_match,
            "mean_absolute_pixel_difference": (
                mean_absolute_difference
            ),
            "maximum_pixel_difference": (
                maximum_absolute_difference
            ),
            "original_unique_levels": (
                original_stats["unique_levels"]
            ),
            "yolo_unique_levels": (
                prepared_stats["unique_levels"]
            ),
            "original_gap_1_percentage": (
                original_stats["gap_1_percentage"]
            ),
            "yolo_gap_1_percentage": (
                prepared_stats["gap_1_percentage"]
            ),
            "original_gap_2_percentage": (
                original_stats["gap_2_percentage"]
            ),
            "yolo_gap_2_percentage": (
                prepared_stats["gap_2_percentage"]
            )
        })

        if index % 50 == 0:
            print(
                f"Compared: {index}/{len(common_names)}"
            )

    if not records:
        raise RuntimeError(
            "No image pairs could be compared."
        )

    report = {
        "original_image_count": len(original_paths),
        "yolo_image_count": len(yolo_paths),
        "matching_images": len(common_names),
        "compared_images": len(records),
        "pixel_mismatch_images": pixel_mismatches,
        "dimension_mismatch_images": dimension_mismatches,
        "exact_pixel_match_images": sum(
            record["exact_pixel_match"]
            for record in records
        ),
        "mean_original_unique_levels": float(
            np.mean([
                record["original_unique_levels"]
                for record in records
            ])
        ),
        "mean_yolo_unique_levels": float(
            np.mean([
                record["yolo_unique_levels"]
                for record in records
            ])
        ),
        "mean_pixel_difference": float(
            np.mean([
                record["mean_absolute_pixel_difference"]
                for record in records
                if record["mean_absolute_pixel_difference"] is not None
            ])
        ),
        "images": records
    }

    report_path = (
        OUTPUT_DIR / "original_vs_yolo_test_report.json"
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
    print("COMPARISON RESULTS")
    print("=" * 65)

    print(f"Compared images: {len(records)}")

    print(
        f"Exact pixel matches: "
        f"{report['exact_pixel_match_images']}"
    )

    print(
        f"Pixel mismatch images: {pixel_mismatches}"
    )

    print(
        f"Dimension mismatch images: "
        f"{dimension_mismatches}"
    )

    print(
        f"Original unique levels mean: "
        f"{report['mean_original_unique_levels']:.2f}"
    )

    print(
        f"YOLO unique levels mean: "
        f"{report['mean_yolo_unique_levels']:.2f}"
    )

    print(
        f"Mean absolute pixel difference: "
        f"{report['mean_pixel_difference']:.5f}"
    )

    print("\nReport saved:")
    print(report_path)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
