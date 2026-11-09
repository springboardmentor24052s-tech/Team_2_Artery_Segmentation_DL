
import json
import csv
import random
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "results"
    / "stenosis_classification_dataset"
)

PATCH_SIZE = 128
CATEGORY_ID = 26

# Maximum negative patches generated per image
MAX_NEGATIVES_PER_IMAGE = 2

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

def create_output_directories():

    for split in ["train", "val", "test"]:

        for label in ["positive", "negative"]:

            directory = OUTPUT_ROOT / split / label

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


# --------------------------------------------------
# READ COCO ANNOTATIONS
# --------------------------------------------------

def load_annotations(split):

    annotation_path = (
        DATASET_ROOT
        / split
        / "annotations"
        / f"{split}.json"
    )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    with open(annotation_path, "r", encoding="utf-8") as file:

        data = json.load(file)

    images = {
        image["id"]: image
        for image in data["images"]
    }

    annotations = {}

    for annotation in data["annotations"]:

        if annotation["category_id"] != CATEGORY_ID:
            continue

        image_id = annotation["image_id"]

        annotations.setdefault(
            image_id,
            []
        ).append(annotation)

    return images, annotations


# --------------------------------------------------
# CROP AND RESIZE IMAGE PATCH
# --------------------------------------------------

def crop_patch(image, bbox):

    height, width = image.shape[:2]

    x, y, box_width, box_height = bbox

    # Add padding around the ground-truth box
    padding = 0.20 * max(box_width, box_height)

    center_x = x + box_width / 2
    center_y = y + box_height / 2

    side = max(box_width, box_height) + 2 * padding

    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))

    x2 = int(round(center_x + side / 2))
    y2 = int(round(center_y + side / 2))

    # Clip coordinates to image dimensions
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    patch = image[y1:y2, x1:x2]

    if patch.size == 0:
        return None

    patch = cv2.resize(
        patch,
        (PATCH_SIZE, PATCH_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return patch


# --------------------------------------------------
# GENERATE POSITIVE PATCHES
# --------------------------------------------------

def generate_positive_patches(
    split,
    image,
    image_id,
    annotations,
    image_filename,
    manifest
):

    count = 0

    for annotation_index, annotation in enumerate(annotations):

        bbox = annotation.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        patch = crop_patch(
            image,
            bbox
        )

        if patch is None:
            continue

        filename = (
            f"{image_id}_positive_{annotation_index}.png"
        )

        output_path = (
            OUTPUT_ROOT
            / split
            / "positive"
            / filename
        )

        if not cv2.imwrite(str(output_path), patch):
            raise RuntimeError(
                f"Failed to save patch: {output_path}"
            )

        manifest.append({
            "split": split,
            "filename": filename,
            "image_id": image_id,
            "source_image": image_filename,
            "label": 1,
            "class_name": "stenosis",
            "annotation_id": annotation["id"]
        })

        count += 1

    return count


# --------------------------------------------------
# GENERATE NEGATIVE PATCHES
# --------------------------------------------------

def generate_negative_patches(
    split,
    image,
    image_id,
    image_filename,
    annotations,
    manifest
):

    height, width = image.shape[:2]

    # Create a mask containing all ground-truth stenosis boxes
    occupied = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for annotation in annotations:

        x, y, box_width, box_height = annotation["bbox"]

        x1 = max(0, int(np.floor(x)))
        y1 = max(0, int(np.floor(y)))

        x2 = min(width, int(np.ceil(x + box_width)))
        y2 = min(height, int(np.ceil(y + box_height)))

        if x2 > x1 and y2 > y1:
            occupied[y1:y2, x1:x2] = 1

    # Find possible background regions
    # A candidate must be completely free from ground-truth boxes.
    candidate_size = PATCH_SIZE

    possible_locations = []

    for y in range(
        0,
        height - candidate_size + 1,
        32
    ):

        for x in range(
            0,
            width - candidate_size + 1,
            32
        ):

            region = occupied[
                y:y + candidate_size,
                x:x + candidate_size
            ]

            if np.count_nonzero(region) == 0:

                possible_locations.append((x, y))

    random.shuffle(possible_locations)

    selected = []

    # Select non-overlapping negative patches
    for x, y in possible_locations:

        overlaps = False

        for sx, sy in selected:

            if (
                abs(x - sx) < candidate_size
                and abs(y - sy) < candidate_size
            ):
                overlaps = True
                break

        if not overlaps:

            selected.append((x, y))

        if len(selected) >= MAX_NEGATIVES_PER_IMAGE:
            break

    count = 0

    for index, (x, y) in enumerate(selected):

        patch = image[
            y:y + candidate_size,
            x:x + candidate_size
        ]

        if patch.shape != (
            candidate_size,
            candidate_size
        ):
            continue

        filename = (
            f"{image_id}_negative_{index}.png"
        )

        output_path = (
            OUTPUT_ROOT
            / split
            / "negative"
            / filename
        )

        if not cv2.imwrite(str(output_path), patch):
            raise RuntimeError(
                f"Failed to save patch: {output_path}"
            )

        manifest.append({
            "split": split,
            "filename": filename,
            "image_id": image_id,
            "source_image": image_filename,
            "label": 0,
            "class_name": "non_stenosis",
            "annotation_id": ""
        })

        count += 1

    return count


# --------------------------------------------------
# PROCESS ONE DATASET SPLIT
# --------------------------------------------------

def process_split(split):

    print(f"\nProcessing {split.upper()} dataset...")

    images, annotations = load_annotations(split)

    manifest = []

    positive_count = 0
    negative_count = 0
    missing_images = 0

    for image_id, image_info in images.items():

        image_filename = image_info["file_name"]

        image_path = (
            DATASET_ROOT
            / split
            / "images"
            / image_filename
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            missing_images += 1
            print(f"Warning: Cannot read {image_path}")
            continue

        image_annotations = annotations.get(
            image_id,
            []
        )

        positive_count += generate_positive_patches(
            split=split,
            image=image,
            image_id=image_id,
            annotations=image_annotations,
            image_filename=image_filename,
            manifest=manifest
        )

        negative_count += generate_negative_patches(
            split=split,
            image=image,
            image_id=image_id,
            image_filename=image_filename,
            annotations=image_annotations,
            manifest=manifest
        )

    # Save metadata
    csv_path = OUTPUT_ROOT / f"{split}_labels.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "split",
            "filename",
            "image_id",
            "source_image",
            "label",
            "class_name",
            "annotation_id"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(manifest)

    print(f"{split.upper()} completed.")
    print("Positive patches:", positive_count)
    print("Negative patches:", negative_count)
    print("Total patches:", len(manifest))
    print("Unreadable images:", missing_images)
    print("Labels saved:", csv_path)

    return {
        "split": split,
        "positive": positive_count,
        "negative": negative_count,
        "total": len(manifest),
        "missing": missing_images
    }


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 55)
    print("STENOSIS CLASSIFICATION DATASET GENERATION")
    print("=" * 55)

    create_output_directories()

    results = []

    for split in ["train", "val", "test"]:

        result = process_split(split)

        results.append(result)

    print("\n" + "=" * 55)
    print("FINAL DATASET SUMMARY")
    print("=" * 55)

    for result in results:

        print(
            f"{result['split'].upper()}: "
            f"Positive={result['positive']}, "
            f"Negative={result['negative']}, "
            f"Total={result['total']}, "
            f"Unreadable={result['missing']}"
        )

    print("\nDataset generation completed.")
    print("Output directory:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
