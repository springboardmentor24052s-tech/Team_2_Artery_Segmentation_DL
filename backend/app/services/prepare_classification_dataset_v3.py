
import csv
import hashlib
import json
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
    / "stenosis_classification_dataset_v3"
)

PATCH_SIZE = 128
CATEGORY_ID = 26
RANDOM_SEED = 42

MAX_NEGATIVES_PER_IMAGE = 2
STRIDE = 32
ANNOTATION_MARGIN = 8

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------
# IMAGE HASHING
# --------------------------------------------------

def get_image_hash(image_path):
    hasher = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# --------------------------------------------------
# CHECK DUPLICATE IMAGES ACROSS SPLITS
# --------------------------------------------------

def find_cross_split_duplicates():

    hashes = {}
    duplicates = []

    for split in ["train", "val", "test"]:

        image_dir = DATASET_ROOT / split / "images"

        for image_path in sorted(image_dir.iterdir()):

            if not image_path.is_file():
                continue

            image_hash = get_image_hash(image_path)

            if image_hash in hashes:

                previous_split, previous_path = hashes[image_hash]

                if previous_split != split:
                    duplicates.append({
                        "hash": image_hash,
                        "first_split": previous_split,
                        "first_image": previous_path.name,
                        "duplicate_split": split,
                        "duplicate_image": image_path.name
                    })

            else:
                hashes[image_hash] = (split, image_path)

    return duplicates


# --------------------------------------------------
# LOAD COCO ANNOTATIONS
# --------------------------------------------------

def load_annotations(split):

    annotation_path = (
        DATASET_ROOT
        / split
        / "annotations"
        / f"{split}.json"
    )

    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    with open(annotation_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    images = {
        item["id"]: item
        for item in data["images"]
    }

    annotations = {}

    for annotation in data["annotations"]:

        if annotation.get("category_id") != CATEGORY_ID:
            continue

        annotations.setdefault(
            annotation["image_id"], []
        ).append(annotation)

    return images, annotations


# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

def create_directories():

    for split in ["train", "val", "test"]:

        for label in ["positive", "negative"]:
            directory = OUTPUT_ROOT / split / label
            directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# POSITIVE PATCH EXTRACTION
# --------------------------------------------------

def create_positive_patch(image, bbox):

    height, width = image.shape[:2]

    x, y, box_width, box_height = map(float, bbox)

    if box_width <= 0 or box_height <= 0:
        return None

    side = max(box_width, box_height) * 1.4

    center_x = x + box_width / 2
    center_y = y + box_height / 2

    x1 = max(0, int(round(center_x - side / 2)))
    y1 = max(0, int(round(center_y - side / 2)))

    x2 = min(width, int(round(center_x + side / 2)))
    y2 = min(height, int(round(center_y + side / 2)))

    if x2 <= x1 or y2 <= y1:
        return None

    patch = image[y1:y2, x1:x2]

    if patch.size == 0:
        return None

    return cv2.resize(
        patch,
        (PATCH_SIZE, PATCH_SIZE),
        interpolation=cv2.INTER_AREA
    )


# --------------------------------------------------
# CREATE OCCUPIED REGION MASK
# --------------------------------------------------

def create_occupied_mask(image, annotations):

    height, width = image.shape[:2]

    occupied = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for annotation in annotations:

        bbox = annotation.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x, y, box_width, box_height = map(float, bbox)

        x1 = max(
            0,
            int(np.floor(x)) - ANNOTATION_MARGIN
        )

        y1 = max(
            0,
            int(np.floor(y)) - ANNOTATION_MARGIN
        )

        x2 = min(
            width,
            int(np.ceil(x + box_width)) + ANNOTATION_MARGIN
        )

        y2 = min(
            height,
            int(np.ceil(y + box_height)) + ANNOTATION_MARGIN
        )

        if x2 > x1 and y2 > y1:
            occupied[y1:y2, x1:x2] = 1

    return occupied


# --------------------------------------------------
# NEGATIVE PATCH CANDIDATES
# --------------------------------------------------

def find_negative_candidates(image, occupied):

    height, width = image.shape[:2]

    candidates = []

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        STRIDE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        ):

            region = occupied[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            if np.any(region):
                continue

            patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                continue

            # Simple vessel-like structure ranking heuristic.
            # This is not a medical vessel detector.
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            enhanced = clahe.apply(patch)

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (15, 15)
            )

            blackhat = cv2.morphologyEx(
                enhanced,
                cv2.MORPH_BLACKHAT,
                kernel
            )

            score = float(np.mean(blackhat))

            candidates.append((x, y, score))

    return candidates


# --------------------------------------------------
# SELECT NON-OVERLAPPING NEGATIVE PATCHES
# --------------------------------------------------

def select_negative_candidates(candidates):

    # Select a high-response patch and a low-response patch.
    candidates = sorted(
        candidates,
        key=lambda item: item[2]
    )

    selected = []

    if not candidates:
        return selected

    ordered = [
        candidates[-1],
        candidates[0]
    ]

    for candidate in ordered:

        x, y, score = candidate

        overlaps = any(
            abs(x - sx) < PATCH_SIZE
            and abs(y - sy) < PATCH_SIZE
            for sx, sy, _ in selected
        )

        if not overlaps:
            selected.append(candidate)

        if len(selected) >= MAX_NEGATIVES_PER_IMAGE:
            break

    return selected


# --------------------------------------------------
# PROCESS ONE SPLIT
# --------------------------------------------------

def process_split(split, excluded_images):

    print(f"\nProcessing {split.upper()}...")

    images, annotations = load_annotations(split)

    manifest = []

    positive_count = 0
    negative_count = 0
    skipped_duplicates = 0
    unreadable_images = 0
    skipped_positive = 0

    image_dir = DATASET_ROOT / split / "images"

    for image_id, image_info in images.items():

        image_filename = image_info["file_name"]

        if (split, image_filename) in excluded_images:
            skipped_duplicates += 1
            continue

        image_path = image_dir / image_filename

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            unreadable_images += 1
            continue

        image_annotations = annotations.get(image_id, [])

        # Positive patches
        for index, annotation in enumerate(image_annotations):

            patch = create_positive_patch(
                image,
                annotation.get("bbox", [])
            )

            if patch is None:
                skipped_positive += 1
                continue

            filename = (
                f"{image_id}_positive_{index}.png"
            )

            output_path = (
                OUTPUT_ROOT
                / split
                / "positive"
                / filename
            )

            if not cv2.imwrite(str(output_path), patch):
                raise RuntimeError(
                    f"Could not save {output_path}"
                )

            manifest.append({
                "split": split,
                "filename": filename,
                "image_id": image_id,
                "source_image": image_filename,
                "label": 1,
                "class_name": "stenosis",
                "annotation_id": annotation["id"],
                "negative_type": ""
            })

            positive_count += 1

        # Weak negative patches
        occupied = create_occupied_mask(
            image,
            image_annotations
        )

        candidates = find_negative_candidates(
            image,
            occupied
        )

        selected = select_negative_candidates(candidates)

        for index, (x, y, score) in enumerate(selected):

            patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

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
                    f"Could not save {output_path}"
                )

            manifest.append({
                "split": split,
                "filename": filename,
                "image_id": image_id,
                "source_image": image_filename,
                "label": 0,
                "class_name": "unannotated_region_weak_negative",
                "annotation_id": "",
                "negative_type": "weak_negative",
                "vessel_response_score": score,
                "patch_x": x,
                "patch_y": y
            })

            negative_count += 1

    csv_path = OUTPUT_ROOT / f"{split}_labels.csv"

    fieldnames = [
        "split",
        "filename",
        "image_id",
        "source_image",
        "label",
        "class_name",
        "annotation_id",
        "negative_type",
        "vessel_response_score",
        "patch_x",
        "patch_y"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(manifest)

    result = {
        "split": split,
        "positive": positive_count,
        "negative": negative_count,
        "total": len(manifest),
        "skipped_duplicates": skipped_duplicates,
        "unreadable_images": unreadable_images,
        "skipped_positive": skipped_positive
    }

    print(result)

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("ARCADE STENOSIS CLASSIFICATION DATASET V3")
    print("=" * 60)

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            "V3 output folder is not empty. "
            "Rename or clear it before generating a fresh dataset."
        )

    create_directories()

    duplicates = find_cross_split_duplicates()

    print("\nCross-split duplicate images:")

    for duplicate in duplicates:
        print(duplicate)

    # Keep the first split occurrence and exclude later copies.
    # Split order: train, val, test.
    excluded_images = set()

    seen_hashes = {}

    for split in ["train", "val", "test"]:

        image_dir = DATASET_ROOT / split / "images"

        for image_path in sorted(image_dir.iterdir()):

            if not image_path.is_file():
                continue

            image_hash = get_image_hash(image_path)

            if image_hash in seen_hashes:
                excluded_images.add(
                    (split, image_path.name)
                )
            else:
                seen_hashes[image_hash] = (
                    split,
                    image_path.name
                )

    results = []

    for split in ["train", "val", "test"]:

        result = process_split(
            split,
            excluded_images
        )

        results.append(result)

    report = {
        "dataset": "ARCADE stenosis classification V3",
        "category_id": CATEGORY_ID,
        "patch_size": PATCH_SIZE,
        "negative_label_note": (
            "Negative patches are weak negatives: "
            "they are outside annotated stenosis bounding boxes "
            "and are not confirmed medically normal."
        ),
        "cross_split_duplicates_found": duplicates,
        "excluded_duplicate_images": [
            {
                "split": split,
                "image": filename
            }
            for split, filename in sorted(excluded_images)
        ],
        "split_results": results
    }

    report_path = OUTPUT_ROOT / "dataset_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("V3 DATASET GENERATION COMPLETED")
    print("=" * 60)

    for result in results:
        print(
            f"{result['split'].upper()}: "
            f"Positive={result['positive']}, "
            f"Weak negatives={result['negative']}, "
            f"Total={result['total']}"
        )

    print("\nReport:", report_path)
    print("Dataset:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
