
from pathlib import Path
import json

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")

ANNOTATION_FILE = ROOT / "stenosis/val/annotations/val.json"
IMAGE_DIR = ROOT / "stenosis/val/images"
MASK_DIR = Path("backend/results/arcade_masks/stenosis/val")

with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    c["id"]: c["name"]
    for c in data["categories"]
}

stenosis_ids = {
    cid for cid, name in categories.items()
    if name.lower() == "stenosis"
}

annotations_by_image = {}

for annotation in data["annotations"]:
    if annotation["category_id"] in stenosis_ids:
        annotations_by_image.setdefault(
            annotation["image_id"], []
        ).append(annotation)

total = 0
mismatched = 0
missing = 0

print("VERIFYING ALL VALIDATION MASKS")
print("=" * 50)

for info in data["images"]:

    if info["id"] not in annotations_by_image:
        continue

    filename = info["file_name"]
    stem = Path(filename).stem

    image_path = IMAGE_DIR / filename
    mask_path = MASK_DIR / f"{stem}_mask.png"

    image = cv2.imread(
        str(image_path), cv2.IMREAD_GRAYSCALE
    )

    generated = cv2.imread(
        str(mask_path), cv2.IMREAD_GRAYSCALE
    )

    if image is None or generated is None:
        print("Missing image or mask:", filename)
        missing += 1
        continue

    height, width = image.shape

    reconstructed = np.zeros(
        (height, width), dtype=np.uint8
    )

    for annotation in annotations_by_image[info["id"]]:

        for polygon in annotation.get("segmentation", []):

            points = np.array(
                polygon, dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(points).astype(np.int32)

            cv2.fillPoly(
                reconstructed,
                [points],
                255
            )

    generated_binary = generated > 0
    reconstructed_binary = reconstructed > 0

    different = np.count_nonzero(
        generated_binary != reconstructed_binary
    )

    total += 1

    if different > 0:
        mismatched += 1
        print(
            f"MISMATCH: {filename} | "
            f"Different pixels: {different}"
        )

print("\nVALIDATION MASK AUDIT")
print("=" * 50)
print("Images checked:", total)
print("Mismatched masks:", mismatched)
print("Missing files:", missing)
print(
    "Matching masks:",
    total - mismatched
)
print("=" * 50)
