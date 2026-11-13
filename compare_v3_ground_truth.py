
from pathlib import Path
import cv2
import numpy as np

# Paths
original_path = Path(
    "backend/app/data/arcade/arcade/arcade/syntax/val/images/1.png"
)

ground_truth_path = Path(
    "backend/results/arcade_masks/stenosis/val/1_mask.png"
)

prediction_path = Path(
    "backend/results/model_comparison/stenosis_mask.png"
)

output_dir = Path("backend/results/ground_truth_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Read images
original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(str(ground_truth_path), cv2.IMREAD_GRAYSCALE)
prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)

if original is None or ground_truth is None or prediction is None:
    raise FileNotFoundError("One or more comparison images could not be loaded.")

if original.shape != ground_truth.shape or original.shape != prediction.shape:
    raise ValueError("Image dimensions do not match.")

# Convert masks to binary
gt = ground_truth > 0
pred = prediction > 0

# Calculate metrics
intersection = np.logical_and(gt, pred).sum()
union = np.logical_or(gt, pred).sum()

dice = (2 * intersection) / (gt.sum() + pred.sum() + 1e-8)
iou = intersection / (union + 1e-8)

precision = intersection / (pred.sum() + 1e-8)
recall = intersection / (gt.sum() + 1e-8)

# Create comparison panels
original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

gt_overlay = original_bgr.copy()
gt_overlay[gt] = (0, 255, 0)

pred_overlay = original_bgr.copy()
pred_overlay[pred] = (0, 0, 255)

comparison = np.hstack([
    original_bgr,
    cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR),
    cv2.addWeighted(original_bgr, 0.6, gt_overlay, 0.4, 0),
    cv2.addWeighted(original_bgr, 0.6, pred_overlay, 0.4, 0)
])

cv2.imwrite(str(output_dir / "comparison.png"), comparison)

print("\nGround Truth vs Attention U-Net V3")
print("-----------------------------------")
print("Ground truth pixels:", int(gt.sum()))
print("Predicted pixels:", int(pred.sum()))
print(f"Dice:      {dice:.4f}")
print(f"IoU:       {iou:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("Saved:", output_dir / "comparison.png")
