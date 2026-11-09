
from pathlib import Path
import json
import csv
from collections import Counter

import cv2
import numpy as np


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

ANNOTATION_FILE = (
    TRAIN_DIR / "annotations" / "train.json"
)

OUTPUT_DIR = (
    BACKEND_ROOT
    / "results"
    / "train_mode_annotation_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

STENOSIS_CATEGORY_ID = 26


# ============================================================
# FIND IMAGE FILES
# ============================================================

def find_image_files():

    image_map = {}

    for path in TRAIN_DIR.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and "annotation" not in str(path.parent).lower()
        ):

            image_map[path.name] = path

    return image_map


# ============================================================
# READ IMAGE STORAGE MODE
# ============================================================

def get_image_mode(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        return None

    if image.ndim == 2:
        return "grayscale_mode"

    if image.ndim == 3 and image.shape[2] == 3:
        return "RGB_stored"

    if image.ndim == 3 and image.shape[2] == 4:
        return "RGB_stored"

    return None


# ============================================================
# LOAD COCO ANNOTATIONS
# ============================================================

def load_annotations():

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    categories = {
        category["id"]: category["name"]
        for category in data.get("categories", [])
    }

    if STENOSIS_CATEGORY_ID not in categories:
        raise ValueError(
            "Stenosis category ID 26 was not found in the COCO file."
        )

    stenosis_annotations = Counter()

    for annotation in data.get("annotations", []):

        if annotation.get("category_id") == STENOSIS_CATEGORY_ID:

            stenosis_annotations[
                annotation["image_id"]
            ] += 1

    return data, stenosis_annotations


# ============================================================
# BUILD IMAGE-WISE RECORDS
# ============================================================

def build_records():

    image_map = find_image_files()

    data, stenosis_annotations = load_annotations()

    records = []
    missing_images = []
    unreadable_images = []

    for image_info in data.get("images", []):

        image_id = image_info["id"]
        file_name = Path(image_info["file_name"]).name

        image_path = image_map.get(file_name)

        if image_path is None:
            missing_images.append(file_name)
            continue

        mode = get_image_mode(image_path)

        if mode is None:
            unreadable_images.append(file_name)
            continue

        annotation_count = stenosis_annotations.get(
            image_id, 0
        )

        records.append({
            "image_id": image_id,
            "image": file_name,
            "group": mode,
            "stenosis_annotations": annotation_count,
            "has_stenosis_annotation": (
                annotation_count > 0
            )
        })

    return records, missing_images, unreadable_images


# ============================================================
# GENERATE GROUP SUMMARY
# ============================================================

def summarize(records):

    summary = {}

    for group in ["RGB_stored", "grayscale_mode"]:

        group_records = [
            record
            for record in records
            if record["group"] == group
        ]

        counts = np.array(
            [
                record["stenosis_annotations"]
                for record in group_records
            ],
            dtype=int
        )

        total_images = len(counts)

        images_with_annotations = int(
            np.sum(counts > 0)
        )

        images_without_annotations = int(
            np.sum(counts == 0)
        )

        total_annotations = int(np.sum(counts))

        summary[group] = {
            "total_images": total_images,
            "images_with_stenosis": images_with_annotations,
            "images_without_stenosis": images_without_annotations,
            "total_stenosis_annotations": total_annotations,
            "annotation_percentage": round(
                images_with_annotations / total_images * 100,
                2
            ) if total_images else 0,
            "average_annotations_per_image": round(
                float(np.mean(counts)), 4
            ) if total_images else 0,
            "median_annotations_per_image": float(
                np.median(counts)
            ) if total_images else 0,
            "maximum_annotations_per_image": int(
                np.max(counts)
            ) if total_images else 0,
            "annotation_count_distribution": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(counts.tolist()).items()
                )
            }
        }

    return summary


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    output_path = (
        OUTPUT_DIR / "train_mode_annotation_records.csv"
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

    print("=" * 60)
    print("TRAINING IMAGE MODE ANNOTATION COMPARISON")
    print("=" * 60)

    print(f"\nTraining directory:\n{TRAIN_DIR}")
    print(f"\nAnnotation file:\n{ANNOTATION_FILE}")

    records, missing_images, unreadable_images = (
        build_records()
    )

    if not records:
        raise RuntimeError(
            "No images could be matched with COCO annotations."
        )

    summary = summarize(records)

    save_csv(records)

    report_path = (
        OUTPUT_DIR / "train_mode_annotation_summary.json"
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
    print("ANNOTATION COMPARISON RESULTS")
    print("=" * 60)

    print(f"\nMatched images: {len(records)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Unreadable images: {len(unreadable_images)}")

    for group, values in summary.items():

        print(f"\n{group}")
        print("-" * 40)

        for key, value in values.items():

            print(f"{key}: {value}")

    print("\nReports saved:")
    print(report_path)
    print(
        OUTPUT_DIR / "train_mode_annotation_records.csv"
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
