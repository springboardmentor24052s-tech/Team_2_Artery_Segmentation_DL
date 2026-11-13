from pathlib import Path
import cv2
import numpy as np

from app.services.segmentation import predict_segmentation


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "arcade"
    / "arcade"
    / "arcade"
    / "syntax"
    / "val"
    / "images"
    / "1.png"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "syntax_sample_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Generate prediction
print("Loading syntax model and generating prediction...")

result = predict_segmentation(IMAGE_PATH)

original = result["original_image"]
mask = result["mask"]
probability = result["probability_map"]


# Create a color overlay
original_color = cv2.cvtColor(
    original,
    cv2.COLOR_GRAY2BGR
)

overlay = original_color.copy()
overlay[mask > 0] = (0, 255, 0)

combined = cv2.addWeighted(
    original_color,
    0.65,
    overlay,
    0.35,
    0
)


# Save outputs
cv2.imwrite(
    str(OUTPUT_DIR / "original.png"),
    original
)

cv2.imwrite(
    str(OUTPUT_DIR / "syntax_mask.png"),
    mask
)

cv2.imwrite(
    str(OUTPUT_DIR / "syntax_overlay.png"),
    combined
)


# Print prediction statistics
total_pixels = mask.size
foreground_pixels = int(np.count_nonzero(mask))

print("\nSYNTAX PREDICTION RESULTS")
print("-------------------------")
print("Image:", IMAGE_PATH.name)
print("Image dimensions:", original.shape)
print("Mask dimensions:", mask.shape)
print("Foreground pixels:", foreground_pixels)
print(
    "Foreground percentage:",
    round(foreground_pixels / total_pixels * 100, 2),
    "%"
)
print("Probability minimum:", float(probability.min()))
print("Probability maximum:", float(probability.max()))
print("Mean probability:", float(probability.mean()))

print("\nSaved images:")
print(OUTPUT_DIR)