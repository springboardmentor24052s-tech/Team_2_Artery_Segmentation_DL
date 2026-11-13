
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")

samples = [
    ("syntax", "train", "922.png"),
    ("stenosis", "train", "676.png"),
]

for task, split, filename in samples:
    image_path = ROOT / task / split / "images" / filename
    mask_path = (
        MASK_ROOT / task / split /
        f"{Path(filename).stem}_mask.png"
    )

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Could not load sample: {task}/{filename}")
        continue

    if image.shape[:2] != mask.shape:
        print(f"Dimension mismatch: {task}/{filename}")
        continue

    overlay = image.copy()
    overlay[mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    print(f"\nTask: {task}")
    print(f"Image: {filename}")
    print(f"Image dimensions: {image.shape}")
    print(f"Mask dimensions: {mask.shape}")
    print(f"Foreground pixels: {np.count_nonzero(mask)}")

    cv2.imshow(f"{task} - Original", image)
    cv2.imshow(f"{task} - Mask", mask)
    cv2.imshow(f"{task} - Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
