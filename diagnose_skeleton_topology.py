
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

binary = mask > 0
skeleton = skeletonize(binary)

kernel = np.ones((3, 3), dtype=np.uint8)
kernel[1, 1] = 0

neighbor_count = cv2.filter2D(
    skeleton.astype(np.uint8),
    -1,
    kernel
)

values, counts = np.unique(
    neighbor_count[skeleton],
    return_counts=True
)

print("SKELETON TOPOLOGY DIAGNOSTICS")
print("Skeleton pixels:", np.count_nonzero(skeleton))
print("\nNeighbour-count distribution:")

for value, count in zip(values, counts):
    print(f"{value} neighbours: {count} pixels")

print("\nEndpoints:", np.count_nonzero(
    skeleton & (neighbor_count == 1)
))

print("Junction candidate pixels:", np.count_nonzero(
    skeleton & (neighbor_count >= 3)
))

print("Pixels with 2 neighbours:", np.count_nonzero(
    skeleton & (neighbor_count == 2)
))

output = (skeleton.astype(np.uint8) * 255)

cv2.imwrite(
    "backend/results/skeleton_topology_image1.png",
    output
)

print("\nSkeleton visualization saved.")
