
import json
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("backend/app/data/arcade/arcade/arcade")
SPLIT = ROOT / "syntax/train"
JSON_FILE = SPLIT / "annotations/train.json"

# Load the annotation file
with JSON_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Select the image we successfully located
image_info = next(
    (img for img in data["images"] if img["file_name"] == "922.png"),
    None
)

if image_info is None:
    raise ValueError("Image 922.png was not found in the JSON file.")

image_path = SPLIT / "images" / image_info["file_name"]
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

height, width = image.shape[:2]

# Create an empty binary mask
mask = np.zeros((height, width), dtype=np.uint8)

# Draw every polygon annotation belonging to this image
annotations = [
    ann for ann in data["annotations"]
    if ann["image_id"] == image_info["id"]
]

for ann in annotations:
    segmentation = ann.get("segmentation", [])

    # COCO polygon format
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if len(polygon) >= 6:
                points = np.array(
                    polygon, dtype=np.float32
                ).reshape(-1, 2)

                points = np.round(points).astype(np.int32)
                cv2.fillPoly(mask, [points], 255)

# Create an overlay for visual checking
overlay = image.copy()
overlay[mask > 0] = (0, 255, 0)

overlay = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

# Save previews outside the dataset
output_dir = Path("backend/results/mask_preview")
output_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output_dir / "original_922.png"), image)
cv2.imwrite(str(output_dir / "mask_922.png"), mask)
cv2.imwrite(str(output_dir / "overlay_922.png"), overlay)

print("Image:", image_path)
print("Image ID:", image_info["id"])
print("Annotations drawn:", len(annotations))
print("Mask foreground pixels:", int(np.count_nonzero(mask)))
print("Saved previews to:", output_dir.resolve())

cv2.imshow("Original Image", image)
cv2.imshow("Binary Segmentation Mask", mask)
cv2.imshow("Mask Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
