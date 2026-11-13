import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

SPLIT = "train"

JSON_PATH = ROOT / "syntax" / SPLIT / "annotations" / f"{SPLIT}.json"
MASK_DIR = MASK_ROOT / "syntax" / SPLIT


with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


annotations_by_image = {}

for annotation in data["annotations"]:
    annotations_by_image.setdefault(
        annotation["image_id"], []
    ).append(annotation)


def reconstruct_mask(image_info, annotations):
    height = image_info["height"]
    width = image_info["width"]

    mask = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        segmentation = annotation.get("segmentation", [])

        if not isinstance(segmentation, list):
            continue

        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue

            points = np.asarray(
                polygon, dtype=np.float32
            ).reshape(-1, 2)

            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)

            points = np.rint(points).astype(np.int32)

            cv2.fillPoly(mask, [points], 255)

    return mask


print("Checking original polygon alignment...\n")

checked = 0
mismatched = 0

for image_info in data["images"][:50]:

    filename = image_info["file_name"]
    image_id = image_info["id"]

    saved_path = MASK_DIR / f"{Path(filename).stem}_mask.png"

    saved_mask = cv2.imread(
        str(saved_path), cv2.IMREAD_GRAYSCALE
    )

    if saved_mask is None:
        print("Missing saved mask:", filename)
        continue

    annotations = annotations_by_image.get(image_id, [])

    reconstructed = reconstruct_mask(
        image_info, annotations
    )

    difference = np.count_nonzero(
        saved_mask != reconstructed
    )

    total = saved_mask.size
    mismatch_percent = difference / total * 100

    print(
        f"{filename}: "
        f"Different pixels={difference}, "
        f"Mismatch={mismatch_percent:.6f}%"
    )

    if difference > 0:
        mismatched += 1

    checked += 1


print("\nFINAL REPORT")
print("Images checked:", checked)
print("Images with differences:", mismatched)
print("Images matching:", checked - mismatched)