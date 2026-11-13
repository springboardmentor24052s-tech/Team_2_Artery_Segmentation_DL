import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

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
    / "stenosis_classification_dataset_v2"
)

PATCH_SIZE = 128
CATEGORY_ID = 26

# Maximum negative patches per source image
MAX_NEGATIVES_PER_IMAGE = 2

# Candidate patches are considered every 32 pixels
CANDIDATE_STRIDE = 32

# Safety margin around stenosis bounding boxes
BOX_EXCLUSION_MARGIN = 8

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==========================================================
# DIRECTORY SETUP
# ==========================================================

def prepare_output_directory():

    # Rebuild only the v2 output directory.
    # Original dataset remains untouched.

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val", "test"]:

        for label in ["positive", "negative"]:

            (OUTPUT_ROOT / split / label).mkdir(
                parents=True,
                exist_ok=True
            )


# ==========================================================
# LOAD COCO ANNOTATIONS
# ==========================================================

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

    with open(
        annotation_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    images = {
        image["id"]: image
        for image in data["images"]
    }

    annotations_by_image = {}

    for annotation in data["annotations"]:

        if annotation.get("category_id") != CATEGORY_ID:
            continue

        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    return images, annotations_by_image


# ==========================================================
# POSITIVE PATCH GENERATION
# ==========================================================

def crop_positive_patch(image, bbox):

    height, width = image.shape[:2]

    if not bbox or len(bbox) != 4:
        return None

    x, y, box_width, box_height = map(float, bbox)

    if box_width <= 0 or box_height <= 0:
        return None

    padding = 0.20 * max(box_width, box_height)

    side = max(box_width, box_height) + (2 * padding)

    center_x = x + (box_width / 2)
    center_y = y + (box_height / 2)

    x1 = max(
        0,
        int(round(center_x - side / 2))
    )

    y1 = max(
        0,
        int(round(center_y - side / 2))
    )

    x2 = min(
        width,
        int(round(center_x + side / 2))
    )

    y2 = min(
        height,
        int(round(center_y + side / 2))
    )

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


def generate_positive_patches(
    split,
    image,
    image_id,
    image_filename,
    annotations,
    manifest
):

    count = 0

    for annotation_index, annotation in enumerate(annotations):

        patch = crop_positive_patch(
            image,
            annotation.get("bbox")
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
                f"Could not save positive patch: {output_path}"
            )

        manifest.append({

            "split": split,
            "filename": filename,
            "image_id": image_id,
            "source_image": image_filename,
            "label": 1,
            "class_name": "stenosis",
            "annotation_id": annotation.get("id", ""),
            "negative_type": "",
            "vessel_response_score": ""

        })

        count += 1

    return count


# ==========================================================
# BUILD EXCLUSION MASK
# ==========================================================

def build_exclusion_mask(image_shape, annotations):

    height, width = image_shape

    exclusion_mask = np.zeros(
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
            int(np.floor(x)) - BOX_EXCLUSION_MARGIN
        )

        y1 = max(
            0,
            int(np.floor(y)) - BOX_EXCLUSION_MARGIN
        )

        x2 = min(
            width,
            int(np.ceil(x + box_width)) + BOX_EXCLUSION_MARGIN
        )

        y2 = min(
            height,
            int(np.ceil(y + box_height)) + BOX_EXCLUSION_MARGIN
        )

        if x2 > x1 and y2 > y1:

            exclusion_mask[y1:y2, x1:x2] = 1

    return exclusion_mask


# ==========================================================
# APPROXIMATE VESSEL-LIKE RESPONSE
# ==========================================================

def make_line_kernel(size, angle_degrees):

    kernel = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    center = size // 2
    half_length = center - 1

    angle = np.deg2rad(angle_degrees)

    dx = int(round(half_length * np.cos(angle)))
    dy = int(round(half_length * np.sin(angle)))

    start_point = (
        center - dx,
        center - dy
    )

    end_point = (
        center + dx,
        center + dy
    )

    cv2.line(
        kernel,
        start_point,
        end_point,
        1,
        thickness=1
    )

    return kernel


def calculate_vessel_response(image):

    """
    Calculate a heuristic response for dark elongated structures
    using multi-scale morphological black-hat filters.

    This is only a patch-ranking method.
    It is not a validated vessel segmentation model.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    responses = []

    for kernel_size in (9, 15, 21):

        for angle in (0, 45, 90, 135):

            kernel = make_line_kernel(
                kernel_size,
                angle
            )

            blackhat = cv2.morphologyEx(
                enhanced,
                cv2.MORPH_BLACKHAT,
                kernel
            )

            responses.append(blackhat)

    response_map = np.maximum.reduce(
        responses
    ).astype(np.float32)

    return response_map


def calculate_patch_score(response_patch):

    return float(
        np.percentile(response_patch, 90)
    )


# ==========================================================
# FIND NEGATIVE CANDIDATE PATCHES
# ==========================================================

def find_negative_candidates(image, annotations):

    height, width = image.shape[:2]

    exclusion_mask = build_exclusion_mask(
        (height, width),
        annotations
    )

    response_map = calculate_vessel_response(image)

    candidates = []

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        CANDIDATE_STRIDE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            CANDIDATE_STRIDE
        ):

            exclusion_region = exclusion_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # Reject patches overlapping annotated stenosis regions

            if np.any(exclusion_region):
                continue

            response_patch = response_map[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            score = calculate_patch_score(
                response_patch
            )

            candidates.append({

                "x": x,
                "y": y,
                "score": score

            })

    return candidates


# ==========================================================
# SELECT NON-OVERLAPPING PATCHES
# ==========================================================

def select_non_overlapping(
    candidates,
    number_to_select,
    highest_first
):

    if number_to_select <= 0:
        return []

    ordered_candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=highest_first
    )

    selected = []

    for candidate in ordered_candidates:

        x = candidate["x"]
        y = candidate["y"]

        overlaps = False

        for chosen in selected:

            if (
                abs(x - chosen["x"]) < PATCH_SIZE
                and abs(y - chosen["y"]) < PATCH_SIZE
            ):

                overlaps = True
                break

        if not overlaps:

            selected.append(candidate)

        if len(selected) >= number_to_select:
            break

    return selected


def select_negative_patches(candidates):

    """
    Select a mixture of:

    1. High vessel-response candidates.
    2. Low vessel-response background-like candidates.

    These are automatically labelled negative because they are
    outside annotated stenosis boxes. They are not medically
    verified normal-vessel regions.
    """

    if not candidates:
        return []

    hard_count = (
        MAX_NEGATIVES_PER_IMAGE + 1
    ) // 2

    background_count = (
        MAX_NEGATIVES_PER_IMAGE - hard_count
    )

    hard_candidates = select_non_overlapping(
        candidates,
        hard_count,
        highest_first=True
    )

    hard_coordinates = {
        (item["x"], item["y"])
        for item in hard_candidates
    }

    remaining_candidates = [

        item for item in candidates

        if (item["x"], item["y"]) not in hard_coordinates

    ]

    background_candidates = select_non_overlapping(
        remaining_candidates,
        background_count,
        highest_first=False
    )

    selected = [

        (item, "hard_vessel_candidate")
        for item in hard_candidates

    ]

    selected.extend([

        (item, "low_vessel_response")
        for item in background_candidates

    ])

    # Fill remaining slots if insufficient candidates were found.

    if len(selected) < MAX_NEGATIVES_PER_IMAGE:

        already_selected = {

            (item["x"], item["y"])
            for item, _ in selected

        }

        remaining_candidates = [

            item for item in candidates

            if (item["x"], item["y"]) not in already_selected

        ]

        extra_candidates = select_non_overlapping(
            remaining_candidates,
            MAX_NEGATIVES_PER_IMAGE - len(selected),
            highest_first=True
        )

        selected.extend([

            (item, "additional_candidate")
            for item in extra_candidates

        ])

    return selected[:MAX_NEGATIVES_PER_IMAGE]


# ==========================================================
# GENERATE NEGATIVE PATCHES
# ==========================================================

def generate_negative_patches(
    split,
    image,
    image_id,
    image_filename,
    annotations,
    manifest
):

    candidates = find_negative_candidates(
        image,
        annotations
    )

    selected = select_negative_patches(
        candidates
    )

    count = 0

    for patch_index, (candidate, negative_type) in enumerate(selected):

        x = candidate["x"]
        y = candidate["y"]

        patch = image[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            continue

        filename = (
            f"{image_id}_negative_{patch_index}.png"
        )

        output_path = (
            OUTPUT_ROOT
            / split
            / "negative"
            / filename
        )

        if not cv2.imwrite(str(output_path), patch):

            raise RuntimeError(
                f"Could not save negative patch: {output_path}"
            )

        manifest.append({

            "split": split,
            "filename": filename,
            "image_id": image_id,
            "source_image": image_filename,
            "label": 0,
            "class_name": "non_stenosis_candidate",
            "annotation_id": "",
            "negative_type": negative_type,
            "vessel_response_score": round(
                candidate["score"],
                4
            )

        })

        count += 1

    return count


# ==========================================================
# PROCESS ONE DATASET SPLIT
# ==========================================================

def process_split(split):

    print(f"\nProcessing {split.upper()} split...")

    images, annotations_by_image = load_annotations(
        split
    )

    manifest = []

    positive_count = 0
    negative_count = 0
    unreadable_count = 0
    images_without_annotations = 0

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

            unreadable_count += 1

            print(
                f"Warning: Could not read image: {image_path}"
            )

            continue

        image_annotations = annotations_by_image.get(
            image_id,
            []
        )

        if not image_annotations:
            images_without_annotations += 1

        positive_count += generate_positive_patches(
            split=split,
            image=image,
            image_id=image_id,
            image_filename=image_filename,
            annotations=image_annotations,
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
        "vessel_response_score"
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

    print(f"\n{split.upper()} completed.")

    print("Positive patches:", positive_count)
    print("Negative patches:", negative_count)
    print("Total patches:", len(manifest))

    print(
        "Images without stenosis annotations:",
        images_without_annotations
    )

    print("Unreadable images:", unreadable_count)
    print("Manifest saved:", csv_path)

    return {

        "split": split,
        "positive": positive_count,
        "negative": negative_count,
        "total": len(manifest),
        "images_without_annotations": images_without_annotations,
        "unreadable": unreadable_count

    }


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    print("=" * 60)
    print("STENOSIS CLASSIFICATION DATASET V2")
    print("=" * 60)

    print("Source dataset:", DATASET_ROOT)
    print("Output dataset:", OUTPUT_ROOT)

    if not DATASET_ROOT.is_dir():

        raise FileNotFoundError(
            f"ARCADE stenosis dataset folder not found: {DATASET_ROOT}"
        )

    prepare_output_directory()

    results = []

    for split in ["train", "val", "test"]:

        result = process_split(split)

        results.append(result)

    print("\n" + "=" * 60)
    print("FINAL DATASET V2 SUMMARY")
    print("=" * 60)

    for result in results:

        print(
            f"{result['split'].upper()}: "
            f"Positive={result['positive']}, "
            f"Negative={result['negative']}, "
            f"Total={result['total']}, "
            f"Unreadable={result['unreadable']}"
        )

    print("\nDataset generation completed.")

    print("Output directory:", OUTPUT_ROOT)

    print(
        "\nIMPORTANT: Never use the test split "
        "for model training or threshold selection."
    )


if __name__ == "__main__":
    main()