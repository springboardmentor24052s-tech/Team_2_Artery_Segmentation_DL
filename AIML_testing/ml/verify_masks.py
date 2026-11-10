
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT / "data" / "processed" / "stenosis" / "train" / "masks"

mask_files = sorted(MASK_DIR.glob("*.png"))

print("Total masks:", len(mask_files))

blank_masks = []
invalid_masks = []
pixel_values = set()

for path in mask_files:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        invalid_masks.append(path.name)
        continue

    pixel_values.update(np.unique(mask).tolist())

    if not np.any(mask > 0):
        blank_masks.append(path.name)

print("Blank masks:", len(blank_masks))
print("Unreadable masks:", len(invalid_masks))
print("Pixel values:", sorted(pixel_values))
print("Blank mask filenames:", blank_masks)

# Display one mask containing stenosis
for path in mask_files:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is not None and np.any(mask > 0):
        cv2.imshow("Sample Stenosis Mask", mask)
        print("Displayed mask:", path.name)
        print("Mask shape:", mask.shape)
        print("White pixel percentage:", round(
            100 * np.count_nonzero(mask) / mask.size, 4
        ))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
