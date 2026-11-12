
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

PREDICTION_PATH = (
    BASE_DIR
    / "results"
    / "api_predictions"
    / "2d3f9e5602cf414e982bbbdcf41b9d42_stenosis_mask.png"
)

GROUND_TRUTH_PATH = (
    BASE_DIR
    / "results"
    / "ground_truth"
    / "28_stenosis_ground_truth.png"
)

# Check whether both files exist
for path in [PREDICTION_PATH, GROUND_TRUTH_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

# Read masks as grayscale
prediction = cv2.imread(str(PREDICTION_PATH), cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(str(GROUND_TRUTH_PATH), cv2.IMREAD_GRAYSCALE)

if prediction is None or ground_truth is None:
    raise ValueError("Could not read one or both mask images.")

if prediction.shape != ground_truth.shape:
    raise ValueError(
        f"Shape mismatch: prediction={prediction.shape}, "
        f"ground truth={ground_truth.shape}"
    )

# Convert masks to binary
prediction = prediction > 0
ground_truth = ground_truth > 0

# Calculate TP, FP, FN
tp = int(np.logical_and(prediction, ground_truth).sum())
fp = int(np.logical_and(prediction, np.logical_not(ground_truth)).sum())
fn = int(np.logical_and(np.logical_not(prediction), ground_truth).sum())

# Calculate Dice and IoU
dice_denominator = prediction.sum() + ground_truth.sum()
union = np.logical_or(prediction, ground_truth).sum()

dice = (2 * tp / dice_denominator) if dice_denominator else 1.0
iou = (tp / union) if union else 1.0

print("\n========== V3 STENOSIS EVALUATION ==========")
print("Image: 28.png")
print("Image dimensions:", prediction.shape)
print("Predicted foreground pixels:", int(prediction.sum()))
print("Ground-truth foreground pixels:", int(ground_truth.sum()))
print("--------------------------------------------")
print("True Positives (TP):", tp)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("--------------------------------------------")
print(f"Dice Score: {dice:.4f}")
print(f"IoU Score:  {iou:.4f}")
print("============================================")
