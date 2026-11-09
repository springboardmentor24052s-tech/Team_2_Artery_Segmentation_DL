
from pathlib import Path
import cv2
import numpy as np

from backend.app.services.segmentation import predict_stenosis

# Correct stenosis validation image and matching ground truth
image_path = Path(
    "backend/app/data/arcade/arcade/arcade/stenosis/val/images/1.png"
)

mask_path = Path(
    "backend/results/arcade_masks/stenosis/val/1_mask.png"
)

output_dir = Path("backend/results/correct_stenosis_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Check files
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

if not mask_path.exists():
    raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")

# Load original image and ground truth
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or ground_truth is None:
    raise ValueError("Could not read image or ground-truth mask.")

# Generate stenosis prediction
result = predict_stenosis(str(image_path))

# Extract mask and probability map from returned dictionary
prediction = result["mask"]
probability_map = result["probability_map"]

# Convert masks to binary
gt = ground_truth > 127
pred = np.asarray(prediction) > 0

if gt.shape != pred.shape:
    raise ValueError(
        f"Dimension mismatch: GT={gt.shape}, Prediction={pred.shape}"
    )

# Calculate metrics
intersection = np.logical_and(gt, pred).sum()
gt_pixels = gt.sum()
pred_pixels = pred.sum()

dice = (2 * intersection) / (gt_pixels + pred_pixels + 1e-8)

iou = intersection / (
    gt_pixels + pred_pixels - intersection + 1e-8
)

precision = intersection / (pred_pixels + 1e-8)
recall = intersection / (gt_pixels + 1e-8)

# Save prediction mask
prediction_mask = pred.astype(np.uint8) * 255

cv2.imwrite(
    str(output_dir / "stenosis_v3_prediction.png"),
    prediction_mask
)

# Save probability map for visualization
probability_visual = np.clip(
    probability_map * 255, 0, 255
).astype(np.uint8)

cv2.imwrite(
    str(output_dir / "stenosis_v3_probability.png"),
    probability_visual
)

# Create red prediction overlay
overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

overlay[pred] = (
    0.5 * overlay[pred] +
    0.5 * np.array([0, 0, 255])
).astype(np.uint8)

# Save comparison images
cv2.imwrite(
    str(output_dir / "stenosis_v3_overlay.png"),
    overlay
)

cv2.imwrite(
    str(output_dir / "ground_truth.png"),
    ground_truth
)

# Display results
print("\nCorrect Stenosis Validation Comparison")
print("--------------------------------------")
print("Model: Attention U-Net V3")
print("Input image:", image_path.name)
print("Ground truth:", mask_path.name)
print("Image dimensions:", image.shape)

print("\nPixel counts:")
print("Ground-truth pixels:", gt_pixels)
print("Predicted pixels:", pred_pixels)

print("\nEvaluation Metrics:")
print(f"Dice:      {dice:.4f}")
print(f"IoU:       {iou:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\nResults saved in:", output_dir)
