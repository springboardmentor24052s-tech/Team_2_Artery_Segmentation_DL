
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("backend/app/data/arcade/arcade/arcade")

folders = {
    "Train": ROOT / "stenosis/train/images",
    "Validation": ROOT / "stenosis/val/images"
}

print("\nIMAGE INTENSITY ANALYSIS")
print("=" * 55)

for name, folder in folders.items():

    images = sorted(folder.glob("*.png"))

    means = []
    stds = []
    minimums = []
    maximums = []

    for path in images:

        image = cv2.imread(
            str(path), cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        image = image.astype(np.float32) / 255.0

        means.append(image.mean())
        stds.append(image.std())
        minimums.append(image.min())
        maximums.append(image.max())

    print(f"\n{name}")
    print("Images:", len(means))
    print("Average intensity:", round(float(np.mean(means)), 4))
    print("Average standard deviation:", round(float(np.mean(stds)), 4))
    print("Average minimum:", round(float(np.mean(minimums)), 4))
    print("Average maximum:", round(float(np.mean(maximums)), 4))

print("\nAnalysis completed.")
