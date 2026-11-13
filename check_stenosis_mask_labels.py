
from pathlib import Path
import json

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks/stenosis/train")
ANNOTATION_FILE = ROOT / "stenosis/train/annotations/train.json"
IMAGE_DIR = ROOT / "stenosis/train/images"

with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Identify the actual stenosis category from the dataset.
stenosis_ids = {
    category["id"]
    for category in data["categories"]
    if category["name"].lower() == "stenosis"
}

if not stenosis_ids:
    raise RuntimeError("No category named 'stenosis' found in the JSON.")

print("Stenosis category IDs:", stenosis_ids)

images_by_id = {image["id"]: image for image in data["images"]}

annotations_by_image = {}
for annotation in data["annotations"]:
    annotations_by_image.setdefault(annotation["image_id"], []).append(
        annotation
    )

# Examine images that have explicit stenosis annotations.
positive_ids = {
    annotation["image_id"]
    for annotation in data["annotations"]
    if annotation["category_id"] in stenosis_ids
}

print("Images with explicit stenosis annotations:", len(positive_ids))

checked = 0

for image_id in sorted(positive_ids):
    info = images_by_id.get(image_id)
    if info is None:
        continue

    image_name = Path(info["file_name"]).name
    stem = Path(image_name).stem

    image_path = IMAGE_DIR / image_name
    mask_path = MASK_ROOT / f"{stem}_mask.png"

    if not image_path.exists() or not mask_path.exists():
        print("Missing image or mask:", image_name)
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Could not read mask:", mask_path)
        continue

    image_annotations = annotations_by_image.get(image_id, [])

    stenosis_count = sum(
        annotation["category_id"] in stenosis_ids
        for annotation in image_annotations
    )

    other_count = sum(
        annotation["category_id"] not in stenosis_ids
        for annotation in image_annotations
    )

    print(
        f"{image_name}: "
        f"stenosis annotations={stenosis_count}, "
        f"other annotations={other_count}, "
        f"mask foreground pixels={int((mask > 0).sum())}"
    )

    checked += 1

    if checked >= 10:
        break

print("\nImages checked:", checked)
