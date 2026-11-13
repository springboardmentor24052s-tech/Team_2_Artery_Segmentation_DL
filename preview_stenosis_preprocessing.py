
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
OUTPUT = Path("backend/results/preprocessing_preview")

OUTPUT.mkdir(parents=True, exist_ok=True)

samples = ["100.png", "101.png", "102.png"]

# Mild contrast enhancement
clahe = cv2.createCLAHE(
    clipLimit=1.2,
    tileGridSize=(8, 8)
)

for filename in samples:

    image_path = ROOT / "stenosis/val/images" / filename

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print("Missing image:", filename)
        continue

    # Step 1: Mild Gaussian denoising
    denoised = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    # Step 2: Gentle local contrast enhancement
    enhanced = clahe.apply(denoised)

    # Compare original and processed images
    comparison = np.hstack([
        image,
        denoised,
        enhanced
    ])

    output_path = OUTPUT / f"{Path(filename).stem}_preprocessing.png"

    cv2.imwrite(str(output_path), comparison)

    print("Saved:", output_path.resolve())

print("\nPreprocessing preview completed.")
