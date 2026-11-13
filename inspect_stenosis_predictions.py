
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks/stenosis/val")
PRED_ROOT = Path("backend/results/stenosis_predictions")

OUTPUT = Path("backend/results/stenosis_detailed_inspection")
OUTPUT.mkdir(parents=True, exist_ok=True)

samples = ["100.png", "101.png", "102.png"]

for filename in samples:

    stem = Path(filename).stem

    image_path = ROOT / "stenosis/val/images" / filename
    mask_path = MASK_ROOT / f"{stem}_mask.png"
    pred_path = PRED_ROOT / f"{stem}_pred.png"

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

    if any(x is None for x in [image, mask, pred]):
        print("Missing file:", filename)
        print("Image:", image_path.exists())
        print("Mask:", mask_path.exists())
        print("Prediction:", pred_path.exists())
        continue

    mask = (mask > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    print("\nImage:", filename)
    print("Image shape:", image.shape)
    print("Ground truth pixels:", np.sum(mask))
    print("Predicted pixels:", np.sum(pred))

    # Convert original image to colour
    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Ground truth displayed in green
    actual = np.zeros_like(original)
    actual[mask == 1] = (0, 255, 0)

    # Prediction displayed in red
    predicted = np.zeros_like(original)
    predicted[pred == 1] = (0, 0, 255)

    # Overlay: green = actual, red = predicted,
    # yellow = overlapping regions
    overlay = original.copy()

    overlay[mask == 1] = (0, 255, 0)
    overlay[pred == 1] = (0, 0, 255)
    overlay[(mask == 1) & (pred == 1)] = (0, 255, 255)

    # Add panel titles
    panels = [original, actual, predicted, overlay]
    titles = [
        "Original Image",
        "Ground Truth (Green)",
        "V4 Prediction (Red)",
        "Overlay (Yellow = Match)"
    ]

    labelled_panels = []

    for panel, title in zip(panels, titles):
        panel = panel.copy()
        cv2.putText(
            panel, title, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2
        )
        labelled_panels.append(panel)

    combined = np.hstack(labelled_panels)

    output_path = OUTPUT / f"{stem}_detailed.png"
    cv2.imwrite(str(output_path), combined)

    print("Saved:", output_path.resolve())

print("\nDetailed inspection completed.")
