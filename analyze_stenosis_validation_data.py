
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_DIR = Path("backend/results/arcade_masks/stenosis/val")
ANNOTATION_FILE = ROOT / "stenosis/val/annotations/val.json"

with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

annotated_ids = {
    ann["image_id"]
    for ann in data["annotations"]
    if ann["category_id"] == 26
}

percentages = []
missing_masks = 0

for info in data["images"]:

    if info["id"] not in annotated_ids:
        continue

    stem = Path(info["file_name"]).stem
    mask_path = MASK_DIR / f"{stem}_mask.png"

    if not mask_path.exists():
        missing_masks += 1
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        missing_masks += 1
        continue

    foreground = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]

    percentages.append(
        (foreground / total_pixels) * 100
    )

print("\nVALIDATION DATA ANALYSIS")
print("------------------------")
print("Images analyzed:", len(percentages))
print("Missing masks:", missing_masks)

if percentages:
    print("Average foreground percentage:",
          round(np.mean(percentages), 4), "%")
    print("Minimum foreground percentage:",
          round(np.min(percentages), 4), "%")
    print("Maximum foreground percentage:",
          round(np.max(percentages), 4), "%")

print("\nAnalysis completed.")
