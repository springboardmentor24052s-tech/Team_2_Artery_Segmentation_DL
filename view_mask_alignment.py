
from pathlib import Path

import cv2
import numpy as np

# Dataset paths
ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

# Output folder
OUTPUT = Path("backend/results/mask_preview")

# Stenosis validation sample
TASK = "stenosis"
SPLIT = "val"
FILENAME = "100.png"

# Input image path
image_path = ROOT / TASK / SPLIT / "images" / FILENAME

# Ground-truth mask path
mask_path = (
    MASK_ROOT / TASK / SPLIT /
    f"{Path(FILENAME).stem}_mask.png"
)

# Load original image and mask
image = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE
)

mask = cv2.imread(
    str(mask_path),
    cv2.IMREAD_GRAYSCALE
)

# Check whether files loaded successfully
if image is None:
    raise FileNotFoundError(
        f"Original image not found: {image_path}"
    )

if mask is None:
    raise FileNotFoundError(
        f"Ground-truth mask not found: {mask_path}"
    )

# Check image and mask dimensions
if image.shape != mask.shape:
    raise ValueError(
        f"Image-mask dimensions do not match: "
        f"{image.shape} vs {mask.shape}"
    )

# Convert mask into binary format
mask = (mask > 0).astype(np.uint8)

# Create green overlay
overlay = cv2.cvtColor(
    image,
    cv2.COLOR_GRAY2BGR
)

overlay[mask == 1] = (
    0.5 * overlay[mask == 1]
    + 0.5 * np.array([0, 255, 0])
).astype(np.uint8)

# Create output directory
OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

# Save all three images
cv2.imwrite(
    str(OUTPUT / "stenosis_100_original.png"),
    image
)

cv2.imwrite(
    str(OUTPUT / "stenosis_100_ground_truth.png"),
    mask * 255
)

cv2.imwrite(
    str(OUTPUT / "stenosis_100_overlay.png"),
    overlay
)

# Display verification information
print("\nSTENOSIS MASK ALIGNMENT CHECK")
print("--------------------------------")
print("Task:", TASK)
print("Split:", SPLIT)
print("Filename:", FILENAME)
print("Image dimensions:", image.shape)
print("Mask dimensions:", mask.shape)
print("Ground-truth foreground pixels:", np.sum(mask))
print("Dimensions match: YES")

print("\nSaved files:")
print(OUTPUT / "stenosis_100_original.png")
print(OUTPUT / "stenosis_100_ground_truth.png")
print(OUTPUT / "stenosis_100_overlay.png")

print("\nMask alignment check completed.")
