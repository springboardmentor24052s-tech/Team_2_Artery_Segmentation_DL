
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_DIR = Path("backend/results/arcade_masks/stenosis/train")
ANNOTATION_FILE = ROOT / "stenosis/train/annotations/train.json"

with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

annotated_ids = {
    ann["image_id"]
    for ann in data["annotations"]
    if ann["category_id"] == 26
}

total_images = 0
empty_masks = 0
foreground_pixels = []
foreground_percentages = []

for info in data["images"]:

    if info["id"] not in annotated_ids:
        continue

    filename = Path(info["file_name"]).stem
    mask_path = MASK_DIR / f"{filename}_mask.png"

    if not mask_path.exists():
        print("Missing mask:", mask_path)
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Unreadable mask:", mask_path)
        continue

    foreground = int(np.sum(mask > 0))
    total_pixels = mask.shape[0] * mask.shape[1]
    percentage = (foreground / total_pixels) * 100

    total_images += 1
    foreground_pixels.append(foreground)
    foreground_percentages.append(percentage)

    if foreground == 0:
        empty_masks += 1

print("\nSTENOSIS TRAINING DATA ANALYSIS")
print("--------------------------------")
print("Images analyzed:", total_images)
print("Empty masks:", empty_masks)

if foreground_pixels:
    print("Average foreground pixels:",
          round(np.mean(foreground_pixels), 2))

    print("Minimum foreground pixels:",
          min(foreground_pixels))

    print("Maximum foreground pixels:",
          max(foreground_pixels))

    print("Average foreground percentage:",
          round(np.mean(foreground_percentages), 4), "%")

print("\nAnalysis completed.")
