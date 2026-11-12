
import json
import cv2
import numpy as np

from pathlib import Path
from collections import defaultdict


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

# This script is assumed to be saved in the project root,
# one level above the backend folder.

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
)

SPLITS = ["train", "val", "test"]

IMAGE_SIZE = 512

print("\n========== VESSEL-STENOSIS GEOMETRY ANALYSIS ==========\n")

print("Project root:", PROJECT_ROOT)
print("Dataset path:", DATASET_ROOT)
print("Dataset exists:", DATASET_ROOT.exists())


# ============================================================
# 2. VALIDATE DATASET PATH
# ============================================================

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_ROOT}"
    )


# ============================================================
# 3. CREATE BINARY MASK FROM ANNOTATIONS
# ============================================================

def create_mask(annotations, image_id):

    mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    for annotation in annotations:

        if annotation["image_id"] != image_id:
            continue

        segmentation = annotation.get("segmentation", [])

        # Handle polygon segmentation.
        if isinstance(segmentation, list):

            for polygon in segmentation:

                if not isinstance(polygon, list) or len(polygon) < 6:
                    continue

                points = np.array(
                    polygon,
                    dtype=np.float32
                ).reshape(-1, 2)

                points = np.round(points).astype(np.int32)

                cv2.fillPoly(
                    mask,
                    [points],
                    1
                )

    return mask


# ============================================================
# 4. PROCESS TRAIN, VALIDATION AND TEST SPLITS
# ============================================================

for split in SPLITS:

    print(f"\n{'=' * 55}")
    print(f"PROCESSING {split.upper()} SPLIT")
    print(f"{'=' * 55}")

    syntax_file = (
        DATASET_ROOT
        / "syntax"
        / split
        / "annotations"
        / f"{split}.json"
    )

    stenosis_file = (
        DATASET_ROOT
        / "stenosis"
        / split
        / "annotations"
        / f"{split}.json"
    )

    # Check annotation files.
    if not syntax_file.exists():

        print("Syntax annotation file not found:", syntax_file)
        continue

    if not stenosis_file.exists():

        print("Stenosis annotation file not found:", stenosis_file)
        continue

    # Load JSON annotation files.
    with open(syntax_file, "r", encoding="utf-8") as f:
        syntax_data = json.load(f)

    with open(stenosis_file, "r", encoding="utf-8") as f:
        stenosis_data = json.load(f)

    # ========================================================
    # 5. SEPARATE VESSEL AND STENOSIS ANNOTATIONS
    # ========================================================

    # Category 26 represents stenosis in this dataset.
    vessel_annotations = [
        annotation
        for annotation in syntax_data.get("annotations", [])
        if annotation.get("category_id") != 26
    ]

    stenosis_annotations = [
        annotation
        for annotation in stenosis_data.get("annotations", [])
        if annotation.get("category_id") == 26
    ]

    vessel_by_image = defaultdict(list)

    stenosis_by_image = defaultdict(list)

    for annotation in vessel_annotations:

        vessel_by_image[annotation["image_id"]].append(
            annotation
        )

    for annotation in stenosis_annotations:

        stenosis_by_image[annotation["image_id"]].append(
            annotation
        )

    # Find images having both vessel and stenosis annotations.
    common_images = (
        set(vessel_by_image.keys())
        &
        set(stenosis_by_image.keys())
    )

    print("Syntax annotations:", len(vessel_annotations))
    print("Stenosis annotations:", len(stenosis_annotations))
    print("Common annotated images:", len(common_images))

    # ========================================================
    # 6. INITIALIZE ANALYSIS COUNTERS
    # ========================================================

    total_stenosis = 0

    overlapping_stenosis = 0

    non_overlapping_stenosis = 0

    overlap_percentages = []

    # ========================================================
    # 7. CHECK STENOSIS-TO-VESSEL OVERLAP
    # ========================================================

    for image_id in sorted(common_images):

        # Create vessel mask for the image.
        vessel_mask = create_mask(
            vessel_by_image[image_id],
            image_id
        )

        # Process each stenosis annotation separately.
        for annotation in stenosis_by_image[image_id]:

            single_stenosis_mask = create_mask(
                [annotation],
                image_id
            )

            stenosis_pixels = np.count_nonzero(
                single_stenosis_mask
            )

            if stenosis_pixels == 0:
                continue

            # Calculate intersection between masks.
            intersection = np.count_nonzero(
                (single_stenosis_mask > 0)
                &
                (vessel_mask > 0)
            )

            # Calculate spatial overlap percentage.
            overlap_percentage = (
                intersection / stenosis_pixels
            ) * 100

            total_stenosis += 1

            overlap_percentages.append(
                overlap_percentage
            )

            if intersection > 0:

                overlapping_stenosis += 1

            else:

                non_overlapping_stenosis += 1

    # ========================================================
    # 8. DISPLAY RESULTS
    # ========================================================

    print("\nStenosis annotations checked:", total_stenosis)

    print(
        "Annotations with vessel overlap:",
        overlapping_stenosis
    )

    print(
        "Annotations without vessel overlap:",
        non_overlapping_stenosis
    )

    if overlap_percentages:

        print(
            "Mean stenosis-to-vessel overlap (%):",
            round(float(np.mean(overlap_percentages)), 2)
        )

        print(
            "Median stenosis-to-vessel overlap (%):",
            round(float(np.median(overlap_percentages)), 2)
        )

        print(
            "Minimum overlap (%):",
            round(float(np.min(overlap_percentages)), 2)
        )

        print(
            "Maximum overlap (%):",
            round(float(np.max(overlap_percentages)), 2)
        )

    else:

        print("No valid stenosis annotations were available for overlap analysis.")


# ============================================================
# 9. ANALYSIS COMPLETED
# ============================================================

print("\n========== ANALYSIS COMPLETE ==========\n")

print(
    "NOTE: These overlap percentages measure spatial overlap "
    "between annotation masks."
)

print(
    "They are NOT clinical percentage stenosis measurements "
    "or severity classifications."
)
