
import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize

mask_path = Path(
    "backend/results/arcade_masks/syntax/test/1_mask.png"
)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(mask_path)

skeleton = skeletonize(mask > 0)

kernel = np.ones((3, 3), dtype=np.uint8)
kernel[1, 1] = 0

neighbor_count = cv2.filter2D(
    skeleton.astype(np.uint8),
    -1,
    kernel
)

# Identify junction candidate pixels.
junction_mask = (
    skeleton & (neighbor_count >= 3)
).astype(np.uint8)

# Group adjacent junction pixels.
count, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        junction_mask,
        connectivity=8
    )
)

print("JUNCTION REGION ANALYSIS")
print("Total junction candidate pixels:",
      np.count_nonzero(junction_mask))

print("Grouped junction regions:", count - 1)

for index in range(1, count):
    area = int(stats[index, cv2.CC_STAT_AREA])

    x = int(stats[index, cv2.CC_STAT_LEFT])
    y = int(stats[index, cv2.CC_STAT_TOP])
    width = int(stats[index, cv2.CC_STAT_WIDTH])
    height = int(stats[index, cv2.CC_STAT_HEIGHT])

    print({
        "region": index,
        "pixels": area,
        "bounding_box": (x, y, width, height),
        "centroid": (
            round(float(centroids[index][0]), 2),
            round(float(centroids[index][1]), 2)
        )
    })

print("\nJunction grouping completed.")
