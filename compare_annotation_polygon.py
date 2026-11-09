
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

TASK = "syntax"
SPLIT = "train"
FILENAME = "922.png"

annotation_file = (
    ROOT / TASK / SPLIT / "annotations" / f"{SPLIT}.json"
)

with annotation_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

image_info = next(
    item for item in data["images"]
    if item["file_name"] == FILENAME
)

image_id = image_info["id"]

annotations = [
    item for item in data["annotations"]
    if item["image_id"] == image_id
]

height = image_info["height"]
width = image_info["width"]

# Independently redraw polygons from the JSON.
reference = np.zeros((height, width), dtype=np.uint8)

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
        cv2.fillPoly(reference, [points], 255)

generated_path = (
    MASK_ROOT / TASK / SPLIT / "922_mask.png"
)

generated = cv2.imread(
    str(generated_path),
    cv2.IMREAD_GRAYSCALE
)

if generated is None:
    raise FileNotFoundError(generated_path)

same = np.array_equal(reference > 0, generated > 0)

print("Image:", FILENAME)
print("Image ID:", image_id)
print("Annotations for image:", len(annotations))
print("Generated mask matches redrawn polygons:", same)
print(
    "Pixels differing:",
    int(np.count_nonzero((reference > 0) != (generated > 0)))
)
print(
    "Reference foreground pixels:",
    int(np.count_nonzero(reference))
)
print(
    "Generated foreground pixels:",
    int(np.count_nonzero(generated))
)
