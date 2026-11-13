
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks/syntax/val")
PRED_ROOT = Path("backend/results/unet_predictions")
OUTPUT = Path("backend/results/prediction_comparisons")

OUTPUT.mkdir(parents=True, exist_ok=True)

samples = ["100.png", "101.png", "102.png"]

for filename in samples:
    stem = Path(filename).stem

    image_path = ROOT / "syntax/val/images" / filename
    mask_path = MASK_ROOT / f"{stem}_mask.png"
    pred_path = PRED_ROOT / f"{stem}_pred.png"

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

    if any(item is None for item in [image, mask, pred]):
        print("Missing file for:", filename)
        continue

    # Convert masks into binary arrays
    actual = mask > 0
    predicted = pred > 0

    # Calculate TP, FP and FN
    tp = np.sum(actual & predicted)
    fp = np.sum(~actual & predicted)
    fn = np.sum(actual & ~predicted)

    # Calculate evaluation metrics
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print(f"\n========== Image {filename} ==========")
    print(f"Reference foreground: {np.mean(actual) * 100:.2f}%")
    print(f"Predicted foreground: {np.mean(predicted) * 100:.2f}%")
    print(f"Dice Score: {dice:.4f}")
    print(f"IoU Score: {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Create visual comparison
    comparison = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Green = Reference, Red = Prediction
    comparison[actual] = (0, 255, 0)
    comparison[predicted] = (0, 0, 255)

    # Yellow = Correct overlap
    overlap = actual & predicted
    comparison[overlap] = (0, 255, 255)

    output_path = OUTPUT / f"{stem}_comparison.png"
    cv2.imwrite(str(output_path), comparison)

    print("Saved comparison:", output_path.resolve())

print("\nEvaluation complete.")
