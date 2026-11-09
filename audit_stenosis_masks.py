
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "backend/app/data/arcade/arcade/arcade"
MASK_ROOT = ROOT / "backend/results/arcade_masks"

SPLITS = ["train", "val", "test"]

for split in SPLITS:

    print(f"\n========== {split.upper()} ==========")

    annotation_file = (
        DATASET / f"stenosis/{split}/annotations/{split}.json"
    )

    mask_dir = MASK_ROOT / f"stenosis/{split}"

    with open(annotation_file, "r") as file:
        data = json.load(file)

    stenosis_ids = {
        category["id"]
        for category in data["categories"]
        if category["name"].lower() == "stenosis"
    }

    all_annotations = data["annotations"]

    stenosis_annotations = [
        ann for ann in all_annotations
        if ann["category_id"] in stenosis_ids
    ]

    print("Total images:", len(data["images"]))
    print("Total annotations:", len(all_annotations))
    print("Stenosis annotations:", len(stenosis_annotations))

    images_with_stenosis = {
        ann["image_id"] for ann in stenosis_annotations
    }

    images_without_stenosis = [
        image for image in data["images"]
        if image["id"] not in images_with_stenosis
    ]

    print(
        "Images with stenosis:",
        len(images_with_stenosis)
    )

    print(
        "Images without stenosis:",
        len(images_without_stenosis)
    )

    mismatches = 0
    missing_masks = 0
    checked = 0

    annotations_by_image = {}

    for ann in stenosis_annotations:
        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)

    # Check every generated mask against category-26 polygons.
    for image_info in data["images"]:

        filename = Path(image_info["file_name"]).stem
        mask_path = mask_dir / f"{filename}_mask.png"

        if not mask_path.is_file():
            missing_masks += 1
            continue

        generated_mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if generated_mask is None:
            missing_masks += 1
            continue

        expected_mask = np.zeros(
            (image_info["height"], image_info["width"]),
            dtype=np.uint8
        )

        for ann in annotations_by_image.get(
            image_info["id"], []
        ):

            for polygon in ann.get("segmentation", []):

                if len(polygon) < 6:
                    continue

                points = np.array(
                    polygon,
                    dtype=np.float32
                ).reshape(-1, 2)

                points = np.round(points).astype(np.int32)

                cv2.fillPoly(
                    expected_mask,
                    [points],
                    255
                )

        if generated_mask.shape != expected_mask.shape:
            mismatches += 1

        elif not np.array_equal(
            generated_mask > 0,
            expected_mask > 0
        ):
            mismatches += 1

        checked += 1

    print("Masks checked:", checked)
    print("Missing masks:", missing_masks)
    print("Masks differing from category-26 ground truth:", mismatches)

print("\nMask audit completed.")
