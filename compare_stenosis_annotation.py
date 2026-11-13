
from pathlib import Path
import json

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
ANNOTATION_FILE = ROOT / "stenosis/train/annotations/train.json"
IMAGE_DIR = ROOT / "stenosis/train/images"
MASK_DIR = Path("backend/results/arcade_masks/stenosis/train")

IMAGE_NAME = "1.png"

with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

image_info = next(
    (item for item in data["images"]
     if Path(item["file_name"]).name == IMAGE_NAME),
    None
)

if image_info is None:
    raise RuntimeError(f"{IMAGE_NAME} not found in annotation JSON.")

stenosis_ids = {
    category["id"]
    for category in data["categories"]
    if category["name"].lower() == "stenosis"
}

annotations = [
    annotation for annotation in data["annotations"]
    if annotation["image_id"] == image_info["id"]
    and annotation["category_id"] in stenosis_ids
]

image_path = IMAGE_DIR / IMAGE_NAME
mask_path = MASK_DIR / "1_mask.png"

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
generated_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or generated_mask is None:
    raise RuntimeError("Could not load image or generated mask.")

reference_mask = np.zeros(image.shape, dtype=np.uint8)

for annotation in annotations:
    for polygon in annotation.get("segmentation", []):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)

        cv2.fillPoly(
            reference_mask,
            [np.round(points).astype(np.int32)],
            255
        )

generated_binary = generated_mask > 0
reference_binary = reference_mask > 0

intersection = np.logical_and(
    generated_binary, reference_binary
).sum()

union = np.logical_or(
    generated_binary, reference_binary
).sum()

difference = np.logical_xor(
    generated_binary, reference_binary
).sum()

dice = (
    2 * intersection
    / (generated_binary.sum() + reference_binary.sum())
    if (generated_binary.sum() + reference_binary.sum()) > 0
    else 1.0
)

print("Image:", IMAGE_NAME)
print("Stenosis annotations:", len(annotations))
print("Generated mask pixels:", int(generated_binary.sum()))
print("Reconstructed annotation pixels:", int(reference_binary.sum()))
print("Different pixels:", int(difference))
print("Mask Dice:", round(float(dice), 6))

# Save an overlay to inspect alignment visually.
overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
overlay[reference_binary] = (0, 255, 0)
overlay[generated_binary] = (0, 0, 255)

both = np.logical_and(reference_binary, generated_binary)
overlay[both] = (0, 255, 255)

output_path = Path("backend/results/stenosis_mask_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), overlay)

print("Comparison image saved to:", output_path.resolve())
