
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

SPLIT = "train"
TASK = "stenosis"

json_path = ROOT / TASK / SPLIT / "annotations" / f"{SPLIT}.json"
mask_dir = MASK_ROOT / TASK / SPLIT

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    category["id"]: category["name"]
    for category in data.get("categories", [])
}

print("Category IDs:", categories)

stenosis_ids = {
    category_id
    for category_id, name in categories.items()
    if name.lower() == "stenosis"
}

print("Stenosis category IDs:", stenosis_ids)

image_by_id = {
    image["id"]: image for image in data["images"]
}

annotations_by_image = {}
for annotation in data["annotations"]:
    annotations_by_image.setdefault(
        annotation["image_id"], []
    ).append(annotation)

for filename in ["676.png", "738.png", "888.png", "218.png"]:
    image_info = next(
        (item for item in data["images"]
         if item["file_name"] == filename),
        None
    )

    if image_info is None:
        print("\nImage missing from JSON:", filename)
        continue

    image_annotations = annotations_by_image.get(
        image_info["id"], []
    )

    print(f"\nImage: {filename}")
    print("Total annotations:", len(image_annotations))
    print(
        "Annotation categories:",
        [
            (a.get("category_id"), categories.get(a.get("category_id")))
            for a in image_annotations
        ]
    )

    mask_path = mask_dir / f"{Path(filename).stem}_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Mask missing:", mask_path)
        continue

    print("Mask shape:", mask.shape)
    print("Foreground pixels:", int(np.count_nonzero(mask)))
