
import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize

mask_path = Path(
    "backend/results/arcade_masks/syntax/test/1_mask.png"
)

output_path = Path(
    "backend/results/junction_regions_visualization.png"
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

junction_mask = (
    skeleton & (neighbor_count >= 3)
).astype(np.uint8)

endpoint_mask = (
    skeleton & (neighbor_count == 1)
)

count, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        junction_mask,
        connectivity=8
    )
)

# Black background with white skeleton.
visualization = np.zeros(
    (*skeleton.shape, 3),
    dtype=np.uint8
)

visualization[skeleton] = (255, 255, 255)

# Give every junction region a distinct colour.
rng = np.random.default_rng(42)

for index in range(1, count):
    color = tuple(
        int(v) for v in rng.integers(60, 256, size=3)
    )

    visualization[labels == index] = color

# Mark endpoints in green.
visualization[endpoint_mask] = (0, 255, 0)

cv2.imwrite(str(output_path), visualization)

print("Junction regions:", count - 1)
print("Endpoints:", np.count_nonzero(endpoint_mask))
print("Saved visualization:", output_path)
