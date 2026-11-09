
import cv2
import numpy as np


def extract_stenosis_candidates(
    probability_map,
    binary_mask,
    min_area=10,
    max_area=None
):
    """
    Extract candidate regions from a stenosis probability map
    and its binary segmentation mask.

    Returns candidate bounding boxes and region features.
    """

    if probability_map is None or binary_mask is None:
        raise ValueError("Probability map and binary mask are required.")

    probability_map = np.asarray(probability_map, dtype=np.float32)
    binary_mask = np.asarray(binary_mask)

    if probability_map.shape != binary_mask.shape:
        raise ValueError("Probability map and mask dimensions must match.")

    if not np.isfinite(probability_map).all():
        raise ValueError("Probability map contains invalid values.")

    if min_area < 1:
        raise ValueError("min_area must be at least 1.")

    height, width = binary_mask.shape

    if max_area is None:
        max_area = height * width

    binary = (binary_mask > 0).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    candidates = []

    for component_id in range(1, count):
        x, y, box_width, box_height, area = stats[component_id]

        if area < min_area or area > max_area:
            continue

        region = labels == component_id
        probabilities = probability_map[region]

        candidates.append({
            "candidate_id": len(candidates) + 1,
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(box_width),
                "height": int(box_height)
            },
            "centroid": {
                "x": round(float(centroids[component_id][0]), 2),
                "y": round(float(centroids[component_id][1]), 2)
            },
            "area_pixels": int(area),
            "mean_probability": round(
                float(np.mean(probabilities)), 4
            ),
            "maximum_probability": round(
                float(np.max(probabilities)), 4
            )
        })

    return {
        "image_width": int(width),
        "image_height": int(height),
        "candidate_count": len(candidates),
        "candidates": candidates
    }
