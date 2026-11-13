
from pathlib import Path
import cv2
import numpy as np

from backend.app.services.segmentation import (
    predict_segmentation,
    predict_stenosis
)

# Input image
IMAGE_PATH = Path(
    "backend/app/data/arcade/arcade/arcade/syntax/val/images/1.png"
)

# Output directory
OUTPUT_DIR = Path("backend/results/model_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Run both models
print("Running Syntax U-Net...")
syntax_result = predict_segmentation(IMAGE_PATH)

print("Running Attention U-Net V3...")
stenosis_result = predict_stenosis(IMAGE_PATH)

# Save masks
cv2.imwrite(
    str(OUTPUT_DIR / "syntax_mask.png"),
    syntax_result["mask"]
)

cv2.imwrite(
    str(OUTPUT_DIR / "stenosis_mask.png"),
    stenosis_result["mask"]
)

# Create colored overlays for visual inspection
original = cv2.cvtColor(
    syntax_result["original_image"],
    cv2.COLOR_GRAY2BGR
)

syntax_overlay = original.copy()
syntax_overlay[syntax_result["mask"] > 0] = (0, 255, 0)

stenosis_overlay = original.copy()
stenosis_overlay[stenosis_result["mask"] > 0] = (0, 0, 255)

cv2.imwrite(
    str(OUTPUT_DIR / "syntax_overlay.png"),
    cv2.addWeighted(original, 0.6, syntax_overlay, 0.4, 0)
)

cv2.imwrite(
    str(OUTPUT_DIR / "stenosis_overlay.png"),
    cv2.addWeighted(original, 0.6, stenosis_overlay, 0.4, 0)
)

print("\nPrediction completed!")
print("Image dimensions:", syntax_result["width"], "x", syntax_result["height"])
print("Syntax foreground pixels:", np.count_nonzero(syntax_result["mask"]))
print("Stenosis foreground pixels:", np.count_nonzero(stenosis_result["mask"]))
print("Results saved in:", OUTPUT_DIR.resolve())
