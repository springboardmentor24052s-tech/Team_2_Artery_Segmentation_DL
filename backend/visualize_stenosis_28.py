
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR / "app" / "data" / "arcade" / "arcade"
    / "arcade" / "stenosis" / "test" / "images" / "28.png"
)

PRED_PATH = (
    BASE_DIR / "results" / "api_predictions"
    / "2d3f9e5602cf414e982bbbdcf41b9d42_stenosis_mask.png"
)

GT_PATH = (
    BASE_DIR / "results" / "ground_truth"
    / "28_stenosis_ground_truth.png"
)

OUTPUT_DIR = BASE_DIR / "results" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

original = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
pred = cv2.imread(str(PRED_PATH), cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(str(GT_PATH), cv2.IMREAD_GRAYSCALE)

if any(x is None for x in [original, pred, gt]):
    raise FileNotFoundError("Could not load one or more images.")

if original.shape != pred.shape or pred.shape != gt.shape:
    raise ValueError("Image dimensions do not match.")

pred = pred > 0
gt = gt > 0

# Convert original image to RGB
original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

# Ground truth displayed in green
gt_display = np.zeros_like(original_rgb)
gt_display[gt] = [0, 255, 0]

# Prediction displayed in red
pred_display = np.zeros_like(original_rgb)
pred_display[pred] = [255, 0, 0]

# Overlay:
# Green = ground truth only
# Red = prediction only
# Yellow = overlap
overlay = original_rgb.copy()

overlay[gt & ~pred] = [0, 255, 0]
overlay[pred & ~gt] = [255, 0, 0]
overlay[gt & pred] = [255, 255, 0]

# Add panel titles
panels = [
    ("Original Angiogram", original_rgb),
    ("Ground Truth", gt_display),
    ("V3 Prediction", pred_display),
    ("Comparison Overlay", overlay),
]

labeled_panels = []

for title, image in panels:
    image = cv2.resize(image, (512, 512))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    canvas = np.zeros((552, 512, 3), dtype=np.uint8)
    canvas[40:552, :] = image_bgr

    cv2.putText(
        canvas, title, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2
    )

    labeled_panels.append(canvas)

comparison = cv2.hconcat(labeled_panels)

output_path = OUTPUT_DIR / "28_v3_stenosis_comparison.png"

if not cv2.imwrite(str(output_path), comparison):
    raise RuntimeError("Failed to save comparison image.")

print("Comparison image created successfully.")
print("Saved at:", output_path)
