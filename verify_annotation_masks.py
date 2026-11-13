
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")
TASK = "syntax"
SPLIT = "train"

annotation_file = (
    ROOT / TASK / SPLIT / "annotations" / f"{SPLIT}.json"
)
image_dir = ROOT / TASK / SPLIT / "images"
mask_dir = MASK_ROOT / TASK / SPLIT

with annotation_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

image_by_id = {
    item["id"]: item for item in data["images"]
}

print("Images in annotation file:", len(image_by_id))
print("Annotations:", len(data["annotations"]))
print("Categories:", len(data.get("categories", [])))

# Check image dimensions and generated masks.
checked = 0

for image_info in data["images"][:20]:
    filename = image_info["file_name"]
    image_path = image_dir / filename
    mask_path = mask_dir / f"{Path(filename).stem}_mask.png"

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Missing image or mask:", filename)
        continue

    expected_size = (
        image_info["height"],
        image_info["width"],
    )

    print(
        filename,
        "| JSON size:", expected_size,
        "| actual image:", image.shape,
        "| mask:", mask.shape,
        "| foreground %:", round(np.mean(mask > 0) * 100, 2),
    )

    if image.shape != expected_size or mask.shape != image.shape:
        print("WARNING: Dimension mismatch:", filename)

    checked += 1

print("\nVerification sample completed:", checked, "images")
