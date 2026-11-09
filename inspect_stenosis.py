
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

TASK = "stenosis"
SPLIT = "train"

annotation_file = (
    ROOT / TASK / SPLIT / "annotations" / f"{SPLIT}.json"
)
image_dir = ROOT / TASK / SPLIT / "images"
mask_dir = MASK_ROOT / TASK / SPLIT

with annotation_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("Categories:")
for category in data.get("categories", []):
    print(category)

annotations_by_image = {}
for annotation in data["annotations"]:
    annotations_by_image.setdefault(
        annotation["image_id"], []
    ).append(annotation)

empty_images = []

for info in data["images"]:
    filename = info["file_name"]
    image_path = image_dir / filename
    mask_path = mask_dir / f"{Path(filename).stem}_mask.png"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Missing mask:", mask_path)
        continue

    if not np.any(mask):
        image_annotations = annotations_by_image.get(
            info["id"], []
        )
        empty_images.append(
            (filename, info["id"], len(image_annotations))
        )

print("\nEmpty-mask images:", len(empty_images))
for filename, image_id, annotation_count in empty_images:
    print(
        filename,
        "| image ID:", image_id,
        "| annotation count:", annotation_count
    )
