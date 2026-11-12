
import json
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

ANNOTATION_FILE = (
    BASE_DIR
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "stenosis"
    / "test"
    / "annotations"
    / "test.json"
)

OUTPUT_DIR = BASE_DIR / "results" / "ground_truth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_IMAGE = "28.png"
STENOSIS_CATEGORY_ID = 26

# ==========================================
# CHECK ANNOTATION FILE
# ==========================================

if not ANNOTATION_FILE.exists():
    raise FileNotFoundError(
        f"Annotation file not found: {ANNOTATION_FILE}"
    )

print("Annotation file found:", ANNOTATION_FILE)

# ==========================================
# LOAD COCO ANNOTATIONS
# ==========================================

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==========================================
# FIND TARGET IMAGE
# ==========================================

image_info = next(
    (
        img for img in data["images"]
        if img["file_name"] == TARGET_IMAGE
    ),
    None
)

if image_info is None:
    raise ValueError(
        f"{TARGET_IMAGE} not found in test.json"
    )

image_id = image_info["id"]
height = image_info["height"]
width = image_info["width"]

print("\nImage:", TARGET_IMAGE)
print("Image ID:", image_id)
print("Dimensions:", width, "x", height)

# ==========================================
# GENERATE GROUND-TRUTH MASK
# ==========================================

mask = np.zeros((height, width), dtype=np.uint8)

matched_annotations = 0
matched_polygons = 0

for annotation in data["annotations"]:

    if annotation["image_id"] != image_id:
        continue

    if annotation["category_id"] != STENOSIS_CATEGORY_ID:
        continue

    matched_annotations += 1

    segmentation = annotation.get("segmentation", [])

    for polygon in segmentation:

        if not isinstance(polygon, list) or len(polygon) < 6:
            continue

        points = np.array(
            polygon,
            dtype=np.float32
        ).reshape(-1, 2)

        points = np.round(points).astype(np.int32)

        cv2.fillPoly(mask, [points], 255)

        matched_polygons += 1

# ==========================================
# SAVE GROUND-TRUTH MASK
# ==========================================

output_path = OUTPUT_DIR / "28_stenosis_ground_truth.png"

if not cv2.imwrite(str(output_path), mask):
    raise RuntimeError("Failed to save ground-truth mask.")

# ==========================================
# DISPLAY RESULTS
# ==========================================

foreground_pixels = int(np.count_nonzero(mask))

print("\n========== GROUND TRUTH RESULTS ==========")
print("Image:", TARGET_IMAGE)
print("Image ID:", image_id)
print("Annotations:", matched_annotations)
print("Polygons:", matched_polygons)
print("Foreground pixels:", foreground_pixels)
print("Output:", output_path)
print("=========================================")

if matched_annotations == 0:
    print("WARNING: No stenosis annotations found for this image.")
