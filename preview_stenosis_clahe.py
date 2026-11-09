
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")
OUTPUT = Path("backend/results/clahe_preview")

OUTPUT.mkdir(parents=True, exist_ok=True)

samples = ["100.png", "101.png", "102.png"]

clahe = cv2.createCLAHE(
    clipLimit=2.0,
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

    enhanced = clahe.apply(image)

    comparison = np.hstack([
        image,
        enhanced
    ])

    output_path = OUTPUT / f"{Path(filename).stem}_clahe.png"

    cv2.imwrite(str(output_path), comparison)

    print("Saved:", output_path.resolve())

print("\nCLAHE preview completed.")
