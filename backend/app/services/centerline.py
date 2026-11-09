
import cv2
import numpy as np
from skimage.morphology import skeletonize


def extract_centerline(mask):
    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = mask > 0

    # Extract a one-pixel-wide vessel skeleton.
    skeleton = skeletonize(binary)

    centerline_mask = (skeleton * 255).astype(np.uint8)

    # Distance from each vessel pixel to the nearest background pixel.
    binary_uint8 = binary.astype(np.uint8)

    distance_map = cv2.distanceTransform(
        binary_uint8,
        cv2.DIST_L2,
        5
    )

    # Approximate local diameter at centerline pixels.
    centerline_distances = distance_map[skeleton]
    diameters = centerline_distances * 2.0

    diameter_values = diameters.tolist()

    if len(diameter_values) > 0:
        mean_diameter = float(np.mean(diameters))
        min_diameter = float(np.min(diameters))
        max_diameter = float(np.max(diameters))
    else:
        mean_diameter = 0.0
        min_diameter = 0.0
        max_diameter = 0.0

    # Count neighboring skeleton pixels for each centerline pixel.
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0

    neighbor_count = cv2.filter2D(
        (skeleton.astype(np.uint8)),
        -1,
        neighbor_kernel
    )

    endpoints = int(np.count_nonzero(skeleton & (neighbor_count == 1)))
    junctions = int(np.count_nonzero(skeleton & (neighbor_count >= 3)))

    return {
        "centerline_mask": centerline_mask,
        "centerline_pixel_count": int(np.count_nonzero(skeleton)),
        "endpoint_count": endpoints,
        "junction_count": junctions,
        "diameter_statistics": {
            "mean_diameter_pixels": round(mean_diameter, 2),
            "minimum_diameter_pixels": round(min_diameter, 2),
            "maximum_diameter_pixels": round(max_diameter, 2)
        },
        "diameter_values_pixels": [
            round(float(value), 2)
            for value in diameter_values
        ]
    }
